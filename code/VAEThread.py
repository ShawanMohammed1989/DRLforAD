"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines a thread that is similar to an ordinary RLLib learner thread. (see rllib.agents.impala)
The main difference is that this thread owns a copy of the VAE used in the policy model.
This VAE is trained by this thread and periodically has it's weights copied over to the online policy model.
"""

import datetime
import logging
import threading
import importlib

import ray
from ray.rllib.execution.minibatch_buffer import MinibatchBuffer
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.window_stat import WindowStat
from six.moves import queue

logger = logging.getLogger(__name__)


def make_vae_learner_thread(workers, config):
    return VAELearnerThread(
        workers,
        minibatch_buffer_size=config["minibatch_buffer_size"],
        num_sgd_iter=config["num_sgd_iter"],
        learner_queue_size=config["learner_queue_size"],
        learner_queue_timeout=config["learner_queue_timeout"],
        config=config)


class VAELearnerThread(threading.Thread):
    """Background thread that learns on sample trajectories produced by Rollout Workers.

    This is for use with AsyncSamplesOptimizer.
    """

    def __init__(self, workers, minibatch_buffer_size, num_sgd_iter,
                 learner_queue_size, learner_queue_timeout, config):
        """Initialize the learner thread.

        Arguments:
            local_worker (RolloutWorker): process local rollout worker holding
                policies this thread will call learn_on_batch() on
            minibatch_buffer_size (int): max number of train batches to store
                in the minibatching buffer
            num_sgd_iter (int): number of passes to learn on per train batch
            learner_queue_size (int): max size of queue of inbound
                train batches to this thread
            learner_queue_timeout (int): raise an exception if the queue has
                been empty for this long in seconds
        """
        threading.Thread.__init__(self)
        self.learner_queue_size = WindowStat("size", 50)
        self.local_worker = workers.local_worker()
        self.inqueue = queue.Queue(maxsize=learner_queue_size)
        self.outqueue = queue.Queue()
        self.minibatch_buffer = MinibatchBuffer(
            inqueue=self.inqueue,
            size=minibatch_buffer_size,
            timeout=learner_queue_timeout,
            num_passes=num_sgd_iter,
            init_num_passes=num_sgd_iter)
        self.queue_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.load_timer = TimerStat()
        self.load_wait_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stats = {}
        self.images = {}
        self.stopped = False
        self.num_steps = 0

        self.configuration = config
        
        # Try to import a custom VAE, if not present, use VAE
        try:
            VAE = importlib.import_module("environments." + self.configuration["env_config"]["ENVIRONMENT_FOLDER_NAME"]).VAE
        except AttributeError:
            logger.warning("No class VAE defined in file VAE, in folder " +
                           self.configuration["env_config"]["ENVIRONMENT_FOLDER_NAME"] + ", proceeding with standard VAE...")
            from VAE import VAE

        @ray.remote(num_cpus=config.get('model', {}).get('custom_model_config', {}).get('VAE_NUM_CPUS'), num_gpus=config.get('model', {}).get('custom_model_config', {}).get('VAE_NUM_GPUS'))
        class remote_vae(VAE):
            pass
       
        self.vae = remote_vae.remote(config.get('model', {}).get('custom_model_config', {}),
                                workers.local_worker().get_policy().model.obs_space)
        
    def run(self):
        """
        Run this thread.
        """
        while not self.stopped:
            self.step()

    def step(self):
        """
        This method calls all the necessary methods for training the VAE and monitoring its progress.
        Any python thread (threading.Thread) must implement the step(), which is called repeatedly until termination.
        :return: None
        """
        with self.queue_timer:
            batch, _ = self.minibatch_buffer.get()
            
        with self.grad_timer:
            # This is where we invoke learn_on_batch() on our vae instead of a policy.
            step_id = self.vae.learn_on_batch.remote(batch)
            ray.get(step_id)
            
        images_outputs_id = self.vae.pop_past_image_outputs.remote()
        images_inputs_id = self.vae.pop_past_image_inputs.remote()
        stats_id = self.vae.pop_past_losses.remote()
        self.stats = ray.get(stats_id)
        input_images = ray.get(images_outputs_id)
        output_images = ray.get(images_inputs_id)
        self.num_steps += 1
        self.outqueue.put((batch.count, {**self.stats, "input_images": input_images, "output_images": output_images}))
        self.learner_queue_size.push(self.inqueue.qsize())

    def add_learner_metrics(self, result):
        """Add internal metrics to a trainer result dict."""
        result["info"].update(self.stats)
        return result
