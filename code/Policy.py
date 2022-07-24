"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines a custom policy that splits training between VAE training and ordinary PPO training.
and train the model.

Custom policies can be either subclassed directly:
https://rllib.readthedocs.io/en/latest/rllib-concepts.html
... or they can be "updated" versions of other polices, such as in:
https://github.com/ray-project/ray/blob/master/rllib/agents/sac/sac_tf_policy.py

The former possibility was chosen here. All functions defined in this file are used to replace the default functions
from the APPO policy with themselves. For a description of each functions arguments and expected returns, refer to the
original RLLib Policy.
"""

import logging
from typing import List, Type, Union

import gym
import numpy as np
from ray.rllib.agents.impala import vtrace_tf as vtrace
from ray.rllib.agents.impala.vtrace_tf_policy import _make_time_major
from ray.rllib.agents.ppo.appo import AsyncPPOTFPolicy, validate_config
from ray.rllib.agents.ppo.appo_tf_policy import TargetNetworkMixin
from ray.rllib.agents.ppo.appo_tf_policy import setup_late_mixins
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, ValueNetworkMixin
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.tf_ops import explained_variance

import tensorflow as tf

logger = logging.getLogger(__name__)


def appo_surrogate_loss(
        policy: Policy, model: ModelV2, dist_class: Type[TFActionDistribution],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """
    This is a slightly changed copy of the original appo_surrogate_loss function. Refer to the original for
    documentation. This copy has a comment indicating the change that me make.
    Furthermore, we store means and stds towards the end of this function.
    """
    model_out, _ = model.from_batch(train_batch)
    action_dist = dist_class(model_out, model)
    
    if isinstance(policy.action_space, gym.spaces.Discrete):
        is_multidiscrete = False
        output_hidden_shape = [policy.action_space.n]
    elif isinstance(policy.action_space,
                    gym.spaces.multi_discrete.MultiDiscrete):
        is_multidiscrete = True
        output_hidden_shape = policy.action_space.nvec.astype(np.int32)
    else:
        is_multidiscrete = False
        output_hidden_shape = 1
    
    # TODO: (sven) deprecate this when trajectory view API gets activated.
    def make_time_major(*args, **kw):
        return _make_time_major(policy, train_batch.get("seq_lens"), *args,
                                **kw)
    
    actions = train_batch[SampleBatch.ACTIONS]
    dones = train_batch[SampleBatch.DONES]
    rewards = train_batch[SampleBatch.REWARDS]
    behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]
    
    target_model_out, _ = policy.target_model.from_batch(train_batch)
    prev_action_dist = dist_class(behaviour_logits, policy.model)
    values = policy.model.value_function()
    values_time_major = make_time_major(values)
    
    policy.model_vars = policy.model.variables()
    policy.target_model_vars = policy.target_model.variables()
    
    if policy.is_recurrent():  # This is our change to the original. Since our seq_lens are tf.ones() we need to skip this
        max_seq_len = tf.reduce_max(train_batch["seq_lens"]) - 1
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
        mask = make_time_major(mask, drop_last=policy.config["vtrace"])
        
        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, mask))
    
    else:
        reduce_mean_valid = tf.reduce_mean
    
    if policy.config["vtrace"]:
        logger.debug("Using V-Trace surrogate loss (vtrace=True)")
        
        # Prepare actions for loss.
        loss_actions = actions if is_multidiscrete else tf.expand_dims(
            actions, axis=1)
        
        old_policy_behaviour_logits = tf.stop_gradient(target_model_out)
        old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)
        
        # Prepare KL for Loss
        mean_kl = make_time_major(
            old_policy_action_dist.multi_kl(action_dist), drop_last=True)
        
        unpacked_behaviour_logits = tf.split(
            behaviour_logits, output_hidden_shape, axis=1)
        unpacked_old_policy_behaviour_logits = tf.split(
            old_policy_behaviour_logits, output_hidden_shape, axis=1)
        
        # Compute vtrace on the CPU for better perf.
        with tf.device("/cpu:0"):
            vtrace_returns = vtrace.multi_from_logits(
                behaviour_policy_logits=make_time_major(
                    unpacked_behaviour_logits, drop_last=True),
                target_policy_logits=make_time_major(
                    unpacked_old_policy_behaviour_logits, drop_last=True),
                actions=tf.unstack(
                    make_time_major(loss_actions, drop_last=True), axis=2),
                discounts=tf.cast(~make_time_major(dones, drop_last=True),
                                  tf.float32) * policy.config["gamma"],
                rewards=make_time_major(rewards, drop_last=True),
                values=values_time_major[:-1],  # drop-last=True
                bootstrap_value=values_time_major[-1],
                dist_class=Categorical if is_multidiscrete else dist_class,
                model=model,
                clip_rho_threshold=tf.cast(
                    policy.config["vtrace_clip_rho_threshold"], tf.float32),
                clip_pg_rho_threshold=tf.cast(
                    policy.config["vtrace_clip_pg_rho_threshold"], tf.float32),
            )
        
        actions_logp = make_time_major(
            action_dist.logp(actions), drop_last=True)
        prev_actions_logp = make_time_major(
            prev_action_dist.logp(actions), drop_last=True)
        old_policy_actions_logp = make_time_major(
            old_policy_action_dist.logp(actions), drop_last=True)
        
        is_ratio = tf.clip_by_value(
            tf.math.exp(prev_actions_logp - old_policy_actions_logp), 0.0, 2.0)
        logp_ratio = is_ratio * tf.exp(actions_logp - prev_actions_logp)
        policy._is_ratio = is_ratio
        
        advantages = vtrace_returns.pg_advantages
        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages *
            tf.clip_by_value(logp_ratio, 1 - policy.config["clip_param"],
                             1 + policy.config["clip_param"]))
        
        action_kl = tf.reduce_mean(mean_kl, axis=0) \
            if is_multidiscrete else mean_kl
        mean_kl = reduce_mean_valid(action_kl)
        mean_policy_loss = -reduce_mean_valid(surrogate_loss)
        
        # The value function loss.
        delta = values_time_major[:-1] - vtrace_returns.vs
        value_targets = vtrace_returns.vs
        mean_vf_loss = 0.5 * reduce_mean_valid(tf.math.square(delta))
        
        # The entropy loss.
        actions_entropy = make_time_major(
            action_dist.multi_entropy(), drop_last=True)
        mean_entropy = reduce_mean_valid(actions_entropy)
    
    else:
        logger.debug("Using PPO surrogate loss (vtrace=False)")
        
        # Prepare KL for Loss
        mean_kl = make_time_major(prev_action_dist.multi_kl(action_dist))
        
        logp_ratio = tf.math.exp(
            make_time_major(action_dist.logp(actions)) -
            make_time_major(prev_action_dist.logp(actions)))
        
        advantages = make_time_major(train_batch[Postprocessing.ADVANTAGES])
        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages *
            tf.clip_by_value(logp_ratio, 1 - policy.config["clip_param"],
                             1 + policy.config["clip_param"]))
        
        action_kl = tf.reduce_mean(mean_kl, axis=0) \
            if is_multidiscrete else mean_kl
        mean_kl = reduce_mean_valid(action_kl)
        mean_policy_loss = -reduce_mean_valid(surrogate_loss)
        
        # The value function loss.
        value_targets = make_time_major(
            train_batch[Postprocessing.VALUE_TARGETS])
        delta = values_time_major - value_targets
        mean_vf_loss = 0.5 * reduce_mean_valid(tf.math.square(delta))
        
        # The entropy loss.
        mean_entropy = reduce_mean_valid(
            make_time_major(action_dist.multi_entropy()))
    
    # The summed weighted loss
    total_loss = mean_policy_loss + \
                 mean_vf_loss * policy.config["vf_loss_coeff"] - \
                 mean_entropy * policy.config["entropy_coeff"]
    
    # Optional additional KL Loss
    if policy.config["use_kl_loss"]:
        total_loss += policy.kl_coeff * mean_kl
    
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_kl = mean_kl
    policy._mean_vf_loss = mean_vf_loss
    policy._mean_entropy = mean_entropy
    policy._value_targets = value_targets
    
    # Store means and stds
    means, log_stds = tf.split(model_out, 2, axis=1)
    
    policy.means = dict()
    policy.stds = dict()
    
    for i in range(policy.config["model"]["custom_model_config"]["OUTPUTS"]):
        policy.means[i] = reduce_mean_valid(means[:, i])
        policy.stds[i] = reduce_mean_valid(log_stds[:, i])

    return total_loss


def stats(policy, train_batch):
    """
    This is an altered version of the original APPO stats function. It adds means and stds.
    """
    values_batched = _make_time_major(
        policy,
        train_batch.get("seq_lens"),
        policy.model.value_function(),
        drop_last=policy.config["vtrace"])
   
    action_distributions = dict()
    
    for i in range(policy.config["model"]["custom_model_config"]["OUTPUTS"]):
        action_distributions["model_action" + str(i) + "_mean"] = policy.means[i]
        action_distributions["model_action" + str(i) + "_mean"] = policy.stds[i]
    
    stats_dict = {
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "policy_loss": policy._mean_policy_loss,
        "entropy": policy._mean_entropy,
        "var_gnorm": tf.linalg.global_norm(policy.model.trainable_variables()),
        "vf_loss": policy._mean_vf_loss,
        "vf_explained_var": explained_variance(
            tf.reshape(policy._value_targets, [-1]),
            tf.reshape(values_batched, [-1])),
        # Means and stds of action stributions
        "action_distributions": action_distributions
    }
    
    if policy.config["vtrace"]:
        is_stat_mean, is_stat_var = tf.nn.moments(policy._is_ratio, [0, 1])
        stats_dict["mean_IS"] = is_stat_mean
        stats_dict["var_IS"] = is_stat_var
    
    if policy.config["use_kl_loss"]:
        stats_dict["kl"] = policy._mean_kl
        stats_dict["KL_Coeff"] = policy.kl_coeff
    
    return stats_dict


def build_custom_model(policy, obs_space, action_space, config):
    """
    Build our custom model two times for APPO. The model is retreived from the ModelCatalog, to which is is registered
    in the main function. Both models get an action space dimension that is fit to CarRacing-v0.
    :param policy: The policy object that will call this method with itself as an argument
    :param obs_space: Our environment observation space dictionary
    :param action_space: The original action space
    :param config: configuration object
    :return: The online model
    """
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])
    # In Car Racing, action space is Box(3,) -> logit_dim is 6
    
    # APPO needs two models
    
    logger.warning("Training model is " + str(config["model"]["custom_model"]))
    
    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        logit_dim,
        config["model"],
        name='online model',
        framework="tf2")
    
    # Target model is updated periodically
    policy.target_model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        logit_dim,
        config["model"],
        name='target model',
        framework="tf2")
    
    return policy.model


def compute_and_clip_gradients(policy, optimizer, loss):
    """
    Gradient computation for PPO is performed according to the RLLib APPO gradient computation and clipping algorithm.
    All gradients are stored as a variable of the policy.
    :param policy: The policy object that will call this method with itself as an argument
    :param optimizer: The PPO optimizer
    :param loss: The PPO loss
    :return: Combined PPO und VAE grads and vars
    """
    
    """
    Policy Gradient computation, separately for actor and critic:
    1. Get trainable variables from model
    2. compute gradients
    3. clip gradients
    4. store clipped gradients for later application
    """
    # Include encoder variables, but do not propagate gradients if VAE is used
    actor_variables = policy.model.actor_model.trainable_variables + \
                      policy.model.vae_encoder.trainable_variables + \
                      policy.model.actor_critic_shared_model.trainable_variables
    actor_grads_and_vars = optimizer.compute_gradients(loss, actor_variables)
    actor_grads = [g for (g, v) in actor_grads_and_vars]
    actor_clipped_grads, _ = tf.clip_by_global_norm(actor_grads, policy.config["grad_clip"])
    actor_clipped_grads_and_vars = list(zip(actor_clipped_grads, actor_variables))
    policy.actor_clipped_grads_and_vars = [(g, v) for (g, v) in actor_clipped_grads_and_vars
                                           if g is not None]
    
    # Include encoder variables, but do not propagate gradients if VAE is used
    critic_variables = policy.model.critic_model.trainable_variables + \
                       policy.model.vae_encoder.trainable_variables + \
                       policy.model.actor_critic_shared_model.trainable_variables
    critic_grads_and_vars = optimizer.compute_gradients(loss, critic_variables)
    critic_grads = [g for (g, v) in critic_grads_and_vars]
    critic_clipped_grads, _ = tf.clip_by_global_norm(critic_grads, policy.config["grad_clip"])
    critic_clipped_grads_and_vars = list(zip(critic_clipped_grads, critic_variables))
    policy.critic_clipped_grads_and_vars = [(g, v) for (g, v) in critic_clipped_grads_and_vars
                                            if g is not None]
    
    # save these for later use in apply_gradients
    grads_and_vars = (policy.critic_clipped_grads_and_vars +
                      policy.actor_clipped_grads_and_vars)
    
    return grads_and_vars


def apply_gradients(policy, optimizer, grads_and_vars):
    """
    Apply the gradients that have been stored in the policy object in 'compute_and_clip_gradients' with their
    respective optimizers.
    :param policy: The policy object that will call this method with itself as an argument
    :param optimizer: The PPO optimizer
    :param grads_and_vars: The combined VAE and PPO grads and vars
    :return: All apply operations
    """
    # Apply gradients separately from policy variables, as we have different optimizers
    actor_ops = policy.actor_optimizer.apply_gradients(policy.actor_clipped_grads_and_vars)
    critic_ops = policy.critic_optimizer.apply_gradients(policy.critic_clipped_grads_and_vars)
    
    return tf.group([actor_ops, critic_ops])


class OptimizerMixin:
    """
    We need two optimizers for our different losses and their respective learning rates.
    """
    
    def __init__(self, config):
        self.critic_optimizer = tf.optimizers.Adam(
            learning_rate=config['model']['custom_model_config'].get('CRITIC_LR') or 0.00001)
        self.actor_optimizer = tf.optimizers.Adam(
            learning_rate=config['model']['custom_model_config'].get('ACTOR_LR') or 0.00002)
        self.vae_stats = {}
        self.past_input_images = []
        self.past_output_images = []


class VAEMixin:
    def set_vae_weights(self, weights, session):
        """
        A custom remote function that is meant to be called by the UpdateVAENetwork callable in the
        training execution plan.
        :param weights: np.array containing pretrained VAE weights
        :param session: tf.Session from our local rollout_worker
        :return: None
        """
        encoder_weights, decoder_weights = weights
        
        # We need to the session and graph that our rollout_worker uses to create our model
        if session:
            with session.as_default():
                with session.graph.as_default():
                    self.model.vae_encoder.set_weights(encoder_weights)
                    self.model.vae_decoder.set_weights(decoder_weights)
        else:
            # If there is no session, we are in eager mode and still want to set the weights
            self.model.vae_encoder.set_weights(encoder_weights)
            self.model.vae_decoder.set_weights(decoder_weights)


def after_init_fn(policy, obs_space, action_space, config):
    """
    Before our policy and its losses are initiated, we need to call the Mixin constructors, such that variables
    that the losses depend on are initiated first.
    :param policy: The policy object that will call this method with itself as an argument
    :param obs_space: Our environment observation space dictionary
    :param action_space: The original CarRacing-v0 action space
    :param config: configuration object
    :return: None
    """
    validate_config(config)
    # Call standard APPO late mixin fn
    setup_late_mixins(policy, obs_space, action_space, config)
    # initialise our Mixin before we initialise our policy initialisation
    VAEMixin.__init__(policy)


def before_init_fn(policy, obs_space, action_space, config):
    """
    :param policy: The policy object that will call this method with itself as an argument
    :param obs_space: Our environment observation space dictionary
    :param action_space: The original CarRacing-v0 action space
    :param config: configuration object
    :return: None
    """
    OptimizerMixin.__init__(policy, config)


# See the documentation of the with_updates() function and the SAC example linked at the top of this file
# for a better understanding
CustomPolicy = AsyncPPOTFPolicy.with_updates(
    name="Modified_PPO_Policy",
    make_model=build_custom_model,
    gradients_fn=compute_and_clip_gradients,
    apply_gradients_fn=apply_gradients,
    stats_fn=stats,
    mixins=[
        LearningRateSchedule, KLCoeffMixin, TargetNetworkMixin,
        ValueNetworkMixin, VAEMixin, OptimizerMixin
    ],
    before_init=before_init_fn,
    after_init=after_init_fn,
    loss_fn=appo_surrogate_loss
)


def get_custom_policy_class(config):
    return CustomPolicy
