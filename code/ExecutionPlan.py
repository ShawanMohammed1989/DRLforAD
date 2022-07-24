"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines a custom execution_plan, which is used in a Trainer.
The Trainer is a tune Trainable that is used by tune to rune trials that determine optimal hyperparameters
and train the model or evaluate its performance.
https://rllib.readthedocs.io/en/latest/tune-usage.html
"""

import logging
import os
import importlib
import ray
from ray.actor import ActorHandle

from ray.rllib.agents.impala.impala import make_learner_thread, \
    BroadcastUpdateLearnerWeights, record_steps_trained
from ray.rllib.agents.ppo.appo import APPOTrainer
from ray.rllib.execution.concurrency_ops import Concurrently, Enqueue, Dequeue
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import StoreToReplayBuffer, Replay
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.rollout_ops import SelectExperiences
from ray.rllib.utils.actors import create_colocated
from ray.util.iter import LocalIterator

from ray.rllib.execution.common import STEPS_TRAINED_COUNTER, _get_global_vars, _get_shared_metrics

from Policy import get_custom_policy_class
from VAEReplayBuffer import LocalVAEReplayBuffer
from VAEThread import make_vae_learner_thread

STEPS_TRAINED_COUNTER_VAE = "steps_trained_counter_vae"
STEPS_SAMPLED_COUNTER_VAE = "steps_samples_counter_vae"
NUM_VAE_UPDATES = "num_vae_updates"
LAST_VAE_UPDATE_TS = "last_vae_update_ts"

logger = logging.getLogger(__name__)


def execution_plan(workers, config):
    """
    This execution plan is a variation of the APPO execution plan provided by RLLib.
    If differs in that we do not only collect experiences from rollout workers and push them to the APPO learner queue,
    but we also put those experiences in a LocalVAEReplayBuffer from which our VAE Learner Thread samples.
    On every PPO training step, the VAE weights are copied to the policy network by a callable.

    :param workers: WorkerSet created by tune
    :param config: config file that we hand over to tune
    :return: Execution Plan operations
    """
    # Create parallel iterator with rollouts
    rollouts = ParallelRollouts(
        workers,
        mode="async",
        num_async=config["max_sample_requests_in_flight_per_worker"])
    
    # Augment with replay and concat to desired train batch size.
    train_batches = rollouts \
        .for_each(lambda batch: batch.decompress_if_needed()). \
        for_each(SelectExperiences(["default_policy"]))
    
    model_config = config.get('model', {}).get('custom_model_config', {})
    
    # PPO learner thread
    ppo_learner_thread = make_learner_thread(workers.local_worker(), config)
    ppo_learner_thread.start()
    
    if config.get('model', {}).get('custom_model_config', {}).get('ENABLE_VAE'):
        # We instantiate our VAE as a remote object to optimize by our vae_learner thread
        # We pull weights to the model used by our ordinary PPO thread
        
        num_replay_buffer_shards = 1  # TODO: See if this should be parameterized
        replay_actors = create_colocated(VAEReplayActor, [
            num_replay_buffer_shards,
            256,
            config.get('model', {}).get('custom_model_config', {}).get('VAE_REPLAY_BUFFER_SIZE') or 10000,
            config.get('model', {}).get('custom_model_config', {}).get('VAE_BATCH_SIZE') or 64],
                                         num_replay_buffer_shards)

        # Add batches to PPO Learner Queue
        ppo_enqueue_op = train_batches \
            .for_each(Enqueue(ppo_learner_thread.inqueue))
        # Also add those batches to the VAE Replay Buffer
        vae_store_op = ppo_enqueue_op \
            .for_each(StoreToReplayBuffer(actors=replay_actors))
    else:
        ppo_enqueue_op = train_batches \
            .for_each(Enqueue(ppo_learner_thread.inqueue))
    
    # We only need to update workers if there are remote workers.
    if workers.remote_workers():
        ppo_enqueue_op = ppo_enqueue_op.zip_with_source_actor() \
            .for_each(BroadcastUpdateLearnerWeights_CatchFails(
            ppo_learner_thread, workers,
            broadcast_interval=config["broadcast_interval"]))
    # This sub-flow updates the steps trained counter based on learner output.
    ppo_dequeue_op = Dequeue(
        ppo_learner_thread.outqueue, check=ppo_learner_thread.is_alive) \
        .for_each(record_steps_trained)
    
    # Callback for APPO to use to update KL, target network periodically.
    # The input to the callback is the learner fetches dict.
    if config["after_train_step"]:
        ppo_dequeue_op = ppo_dequeue_op.for_each(lambda t: t[1]).for_each(config["after_train_step"](workers, config))
    
    if config.get('model', {}).get('custom_model_config', {}).get('ENABLE_VAE'):
        # VAE learner thread
        vae_learner_thread = make_vae_learner_thread(workers, config)
        vae_learner_thread.start()
        ppo_dequeue_op = ppo_dequeue_op. \
            for_each(get_vae_network_updater_callable(vae_learner_thread.vae)(workers=workers,
                                                           vae_update_freq=model_config.get(
                                                               'PERCEPTION_UPDATE_FREQUENCY', 10)))
        
        # VAE sub-flow:
        # This sub-flow sends experiences to the learner.
        post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
        vae_enqueue_op = Replay(actors=replay_actors, num_async=1) \
            .for_each(lambda x: post_fn(x, workers, config)) \
            .for_each(Enqueue(vae_learner_thread.inqueue))
        # This sub-flow updates the steps trained counter based on learner output.
        vae_dequeue_op = Dequeue(
            vae_learner_thread.outqueue, check=vae_learner_thread.is_alive) \
            .for_each(record_steps_trained)
        # Merge all operations
        
        merged_enqueue_op = Concurrently(
            [ppo_enqueue_op, vae_enqueue_op], mode="async", output_indexes=[])
        
        merged_op = Concurrently(
            [ppo_dequeue_op, vae_store_op, merged_enqueue_op, vae_dequeue_op], mode="async", output_indexes=[0, 3])
        
        return StandardMetricsReporting(merged_op, workers, config) \
            .for_each(ppo_learner_thread.add_learner_metrics).for_each(vae_learner_thread.add_learner_metrics)
    else:
        merged_op = Concurrently(
            [ppo_enqueue_op, ppo_dequeue_op], mode="async", output_indexes=[1])
        
        return StandardMetricsReporting(merged_op, workers, config) \
            .for_each(ppo_learner_thread.add_learner_metrics)


def get_vae_network_updater_callable(vae):
    class UpdateVAENetwork:
        """Periodically call policy.set_vae_weights() on all PPO policies.

        Updates the LAST_TARGET_UPDATE_TS and NUM_TARGET_UPDATES counters in the
        local iterator context. The value of the last update counter is used to
        track when we should update the target next.
        """
        
        def __init__(self,
                     workers,
                     vae_update_freq,
                     by_steps_trained=False,
                     policies=frozenset([])):
            self.workers = workers
            self.target_update_freq = vae_update_freq
            self.policies = (policies or workers.local_worker().policies_to_train)
            self.vae = vae
            if by_steps_trained:
                self.metric = STEPS_TRAINED_COUNTER_VAE
            else:
                self.metric = STEPS_SAMPLED_COUNTER_VAE
        
        def __call__(self, _):
            metrics = LocalIterator.get_metrics()
            cur_ts = metrics.counters[self.metric]
            last_update = metrics.counters[LAST_VAE_UPDATE_TS]
            # if cur_ts - last_update > self.target_update_freq:
            to_update = self.policies
            sess = self.workers.local_worker().tf_sess
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, p_id: p_id in to_update and p.set_vae_weights(ray.get(self.vae.get_weights.remote()),
                                                                        sess))
            metrics.counters[NUM_VAE_UPDATES] += 1
            metrics.counters[LAST_VAE_UPDATE_TS] = cur_ts
    
    return UpdateVAENetwork


# This trainer adds functionality to the APPO(Impala) Training algorithm
CustomAPPOTrainer = APPOTrainer.with_updates(
    name="rllib-for-students_APPO-Trainer",
    get_policy_class=get_custom_policy_class,
    execution_plan=execution_plan
)


class LoadableCustomAPPOTrainerTrainer(CustomAPPOTrainer):
    def __init__(self, config, **kwargs):
        super(CustomAPPOTrainer, self).__init__(config, **kwargs)
        if config.get("SAVED_MODEL_WEIGHTS_FILE"):
            logger.info("Loading model weights.")
            self.workers.local_worker().get_policy().import_model_from_h5(os.path.join(config[""], config.get("SAVED_MODEL_WEIGHTS_FILE")))
        if config.get("SAVED_VAE_WEIGHTS_FILE"):
            logger.info("Loading VAE weights.")
            self.workers.local_worker().get_policy().import_model_from_h5(os.path.join(config[""], config.get("SAVED_VAE_WEIGHTS_FILE")))
        self.workers.sync_weights()


VAEReplayActor = ray.remote(num_cpus=1)(LocalVAEReplayBuffer)


# Update worker weights as they finish generating experiences.
class BroadcastUpdateLearnerWeights_CatchFails(BroadcastUpdateLearnerWeights):
    def __call__(self, item):
        actor, batch = item
        self.steps_since_broadcast += 1
        if (self.steps_since_broadcast >= self.broadcast_interval
                and self.learner_thread.weights_updated):
            self.weights = ray.put(self.workers.local_worker().get_weights())
            self.steps_since_broadcast = 0
            self.learner_thread.weights_updated = False
            # Update metrics.
            metrics = _get_shared_metrics()
            metrics.counters["num_weight_broadcasts"] += 1
        try:
            actor.set_weights.remote(self.weights, _get_global_vars())
        except AttributeError as e:
            logger.critical(e)
        # Also update global vars of the local worker.
        self.workers.local_worker().set_global_vars(_get_global_vars())
