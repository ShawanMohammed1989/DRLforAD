"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines a VAE model and it's functions required for training alognside with a replaybuffer tailored to
replay PPO experiences. The states of these PPO experiences are used by the VAEThread to train the VAE model.
Stats are written to tensorboard periodically and separately from other stats in RLLib.
"""

import logging
import numpy as np
import os
import importlib
import ray
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

import tensorflow as tf
# TODO: We do not know why we have to manually enable eager execution for the general VAE here, ...
#  ... but vor the CarRacing VAE we do not. Find out why and fix discrepancies.
tf.compat.v1.enable_eager_execution()

from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))

logger = logging.getLogger(__name__)


class VAE:
    """
	A class containing a VAE network with it's optimizer, loss function, training function and containers and
	functions to handle stats.
	"""
    
    def __init__(self, model_config, obs_space):
        self.configuration = model_config
        self.obs_space = obs_space
        
        # Try to import a custom VAE, if not present, use VAE
        try:
            build_networks = importlib.import_module("environments." + self.configuration["ENVIRONMENT_FOLDER_NAME"]).build_networks
        except AttributeError:
            logger.warning("No build_networks defined in file ArtificialNeuralNetworks.py, in folder " +
                           self.configuration["ENVIRONMENT_FOLDER_NAME"] + ", proceeding with standard VAE...")
            from ArtificialNeuralNetwork import build_networks
        
        # We only need the VAE encoder and decoder here, actor and critic networks are omitted
        self.encoder, self.decoder, _, _, _ = build_networks(model_config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.get('VAE_LR') or 0.0001)
        
        # We store past stats in lists to pop them from when being logged
        self.past_image_inputs = []
        self.past_image_outputs = []
        
        self.input_image_pops = 0
        self.output_image_pops = 0
        self.past_stats_pops = 0
        
        self.past_stats = {
            "vae_logvar": [],
            "vae_mean": [],
            "vae_logvar_sum": [],
            "vae_mean_sum": [],
            "vae_reconstruction_loss": [],
            "kl_loss_angelo": [],
            "total_loss": []
        }
        
        self.writer = tf.summary.create_file_writer(os.path.join(model_config["logdir"], "vae_log"))
    
    def pop_past_losses(self):
        # Pop past losses on every read
        past_stats = self.past_stats
        self.past_stats = self.past_stats = {
            "vae_logvar": [],
            "vae_mean": [],
            "vae_logvar_sum": [],
            "vae_mean_sum": [],
            "vae_reconstruction_loss": [],
            "kl_loss_angelo": [],
            "total_loss": []
        }
        
        with self.writer.as_default():
            for stat_name, stats in past_stats.items():
                for stat in stats:
                    tf.summary.scalar(stat_name, stat, step=self.past_stats_pops)
                    self.writer.flush()
                    self.past_stats_pops += 1
        return past_stats
    
    def pop_past_image_inputs(self):
        images = self.past_image_inputs
        self.past_image_inputs = []
        
        with self.writer.as_default():
            for image in images:
                tf.summary.image("past_image_inputs", image, step=self.input_image_pops)
                self.writer.flush()
                self.input_image_pops += 1
        return images
    
    def pop_past_image_outputs(self):
        images = self.past_image_outputs
        self.past_image_outputs = []
        
        with self.writer.as_default():
            for image in images:
                tf.summary.image("past_image_outputs", image, step=self.input_image_pops)
                self.writer.flush()
                self.output_image_pops += 1
        return images
    
    def append_to_past_stats(self, vae_logvar, vae_mean, vae_logvar_sum, vae_mean_sum,
                             vae_reconstruction_loss, kl_loss_angelo, total_loss):
        
        # Append each loss or stat to respective list
        self.past_stats['vae_logvar'].append(np.mean(vae_logvar))
        self.past_stats['vae_mean'].append(np.mean(vae_mean))
        self.past_stats['vae_logvar_sum'].append(np.mean(vae_logvar_sum))
        self.past_stats['vae_mean_sum'].append(np.mean(vae_mean_sum))
        self.past_stats['vae_reconstruction_loss'].append(np.mean(vae_reconstruction_loss))
        self.past_stats['kl_loss_angelo'].append(np.mean(kl_loss_angelo))
        self.past_stats['total_loss'].append(np.mean(total_loss))
    
    def get_weights(self):
        """
		Get the trainable weights from encoder and decoder.
		Use this function with .remote() when sharing weights over the redis object store.
		:return: list(weights)
		"""
        return [self.encoder.get_weights(), self.decoder.get_weights()]
    
    def set_weights(self, weights):
        """
		Set the trainable weights from encoder and decoder.
		Use this function with .remote() when copying weights over the redis object store.
		"""
        encoder_weights, decoder_weights = weights
        self.encoder.set_weights(encoder_weights)
        self.decoder.set_weights(decoder_weights)
    
    def compute_and_apply_vae_gradients(self, samples):
        
        with tf.GradientTape() as tape:
            image = tf.saturate_cast(samples, dtype=tf.float32)
            mean, log_var, _ = self.encoder([image])
            
            # Reparameterize
            eps_ = tf.random.normal(shape=[mean.shape[1]])
            vae_z = eps_ * tf.exp(log_var * .5) + mean
            
            image_output = self.decoder([vae_z])
            
            # Build the VAE loss and additional stats
            # Build the VAE loss and additional stats
            vae_logvar, vae_mean, vae_logvar_sum, vae_mean_sum, vae_reconstruction_loss, kl_loss_angelo, total_loss = \
                compute_vae_losses_from_network_outputs(image_output,
                                                        image,
                                                        log_var,
                                                        mean)
        
        # Store stats for tensorboard logging
        self.append_to_past_stats(vae_logvar, vae_mean, vae_logvar_sum, vae_mean_sum,
                                  vae_reconstruction_loss, kl_loss_angelo, total_loss)
        
        # self.past_image_inputs.append(tf.expand_dims(image[0], axis=0))
        # self.past_image_outputs.append(tf.expand_dims(image_output[0], axis=0))
        
        # Compute and apply gradients
        vae_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, vae_variables)
        self.optimizer.apply_gradients(zip(grads, vae_variables))
        return grads, {}
    
    def learn_on_batch(self, batch):
        """
		Trains the VAE(s) on batches of experiences.

		:param batch: batch to train on
		:return: gradient info
		"""
        info_out = {}
        if isinstance(batch, MultiAgentBatch):
            for samples in batch.policy_batches.values():
                tensorobs = tf.convert_to_tensor(samples['obs'])
                # If observations need to be unpacked here, see CarRacing VAE for solution and fix
                self.compute_and_apply_vae_gradients(tensorobs)
        elif isinstance(batch, SampleBatch):
            # unflatten inputs first, because retracing occurs when call arguments are not tensors
            tensorobs = tf.convert_to_tensor(batch['obs'])
            self.compute_and_apply_vae_gradients(tensorobs)
        else:
            raise ValueError("Batches for VAE to train on have to be instances of SampleBatch or MultiAgentBatch")
        return info_out


@tf.function
def compute_vae_losses_from_network_outputs(image_output,
                                            image_input,
                                            vae_logvar,
                                            vae_mean):
    """
	A custom loss function that provides an additional loss for the VAE, on top of the policy loss for PPO.
	For better understanding, refer to Kemal's, Angelo's or my master thesis document.
	"""
    K = tf.keras.backend
    
    vae_logvar = vae_logvar
    vae_mean = vae_mean
    vae_logvar_sum = K.sum(vae_logvar)
    vae_mean_sum = K.sum(vae_mean)
    
    # Build a sum of all reconstruction losses, image is preprocessed above
    vae_reconstruction_loss = tf.cast(tf.reduce_sum(tf.losses.mean_squared_error(image_output, image_input), axis=[1, 2]),
                         dtype="float32")  # cast to float32 in order to prevent errors
    
    # Build the KL Loss
    kl_loss_angelo = -tf.reduce_mean(vae_logvar - vae_mean ** 2 - tf.exp(vae_logvar), axis=1)
    
    # Combine losses
    total_loss = tf.reduce_mean(tf.add(vae_reconstruction_loss, kl_loss_angelo))
    return vae_logvar, vae_mean, vae_logvar_sum, vae_mean_sum, vae_reconstruction_loss, kl_loss_angelo, total_loss
