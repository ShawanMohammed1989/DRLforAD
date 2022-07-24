"""
Master Thesis project by Artur Niederfahrenhorst
This file defines an RLlib custom model.
https://rllib.readthedocs.io/en/latest/rllib-training.html
"""

import tensorflow as tf
import importlib
import logging
import os

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import restore_original_dimensions

from .Preprocessing import preprocess_images
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork

logger = logging.getLogger(__name__)


class CustomModel(TFModelV2):
    """
    A TFModelV2 Policy model that uses the neural network structure proposed in Angelo's project.
    See: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.configuration = model_config['custom_model_config']
        # This VAE model's VAE encodes and decodes only an image so we have to build our networks accordingly
        
        # Try to import a custom VAE, if not present, use VAE
        try:
            build_networks = importlib.import_module("environments." + self.configuration["ENVIRONMENT_FOLDER_NAME"]).build_networks
        except AttributeError:
            logger.warning("No build_networks defined in file ArtificialNeuralNetworks.py, in folder " +
                           self.configuration["ENVIRONMENT_FOLDER_NAME"] + ", proceeding with standard VAE...")
            from ArtificialNeuralNetwork import build_networks

        
        # Lazy initialisations:
        # These two lazy initialisations have to be done to write future images to tensorboard
        self.last_vae_image_output = tf.Variable(tf.zeros(shape=[1]), dtype=tf.float32,
                                                 shape=[1],
                                                 name='output_image_dummy')
        self.last_vae_image_input = tf.Variable(tf.zeros(shape=[1]), dtype=tf.float32,
                                                shape=[1],
                                                name='input_image_dummy')
        
        self.register_variables([self.last_vae_image_output, self.last_vae_image_input])
        
        self.last_value_output = None  # Lazy init for most recent value of a forward pass
        
        self.vae_encoder, self.vae_decoder, self.actor_critic_shared_model, self.actor_model, self.critic_model = build_networks(
            self.configuration)
        
        for model in [self.vae_encoder, self.vae_decoder, self.actor_critic_shared_model, self.actor_model,
                      self.critic_model]:
            model.summary()
            self.register_variables(model.variables)
            
        if self.configuration.get("SAVED_MODEL_WEIGHTS_FILE"):
            raise NotImplementedError("This feature has not been implemented")
            self.import_model_from_h5(os.path.join(os.path.dirname(__file__), "..", "saved_models", self.configuration.get("SAVED_MODEL_WEIGHTS_FILE")))
        if self.configuration.get("SAVED_VAE_WEIGHTS_FILE"):
            raise NotImplementedError("This feature has not been implemented")
            self.import_model_from_h5(os.path.join(os.path.dirname(__file__), "..", "saved_models", self.configuration.get("SAVED_VAE_WEIGHTS_FILE")))
    
    def forward(self, input_dict, state, seq_lens):
        """
        Implements the forward pass.
        See: https://ray.readthedocs.io/en/latest/rllib-models.html

        :param input_dict: {"obs”, “obs_flat”, “prev_action”, “prev_reward”, “is_training”}
        :param state: None
        :param seq_lens: None
        :return: (ouputs, state), outsputs of size [BATCH, num_outputs]
        """
        image = tf.saturate_cast(input_dict["obs"]['image'], tf.float32)
        image = preprocess_images(image, self.configuration)
        
        mean, _, vae_feature_map_extraction = self.vae_encoder(
            [tf.cast(image, dtype='float32'), tf.cast(input_dict['obs']['speed'], dtype='float32'),
             tf.cast(input_dict['obs']['gyro'], dtype='float32'),
             tf.cast(input_dict['obs']['steering'], dtype='float32')])
        
        self.last_vae_image_input = image
        if self.configuration['ENABLE_VAE']:
            mean = tf.stop_gradient(mean)
            vae_feature_map_extraction = tf.stop_gradient(vae_feature_map_extraction)
        
        self.last_vae_image_output = self.vae_decoder([mean])  # We "throw other outputs away"
        
        shared_model_output = self.actor_critic_shared_model([vae_feature_map_extraction])
        
        logits = tf.concat(self.actor_model([shared_model_output]), axis=1, name="Concat_logits")
        self.last_value_output = tf.reshape(self.critic_model([shared_model_output]), [-1])
        
        return logits, []  # [] is empty state
    
    def value_function(self):
        """
        Use the last computed value from the forward pass operation. (see function self.forward())
        """
        return self.last_value_output
    
    def import_from_h5(self, h5_file: str) -> None:
        raise NotImplementedError
        # TODO: Implement


class RNNModel(RecurrentNetwork, CustomModel):
    """
    This model is used to utilize the recurrency approach of RLlib.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
    def forward_rnn(self, inputs, state, seq_lens):
        input_dict = restore_original_dimensions(inputs, self.obs_space)
        
        image = preprocess_images(input_dict['obs']['image'], self.configuration)
        image = tf.cast(image, tf.float32)
        
        state = tf.stop_gradient(state)
        
        mean, _, vae_feature_map_extraction = self.vae_encoder(
            [tf.cast(image, dtype='float32'), tf.cast(input_dict['obs']['speed'], dtype='float32'),
             tf.cast(input_dict['obs']['gyro'], dtype='float32'),
             tf.cast(input_dict['obs']['steering'], dtype='float32')])  # _ are log_vars
        
        # If VAE training is enabled, we do not want to optimize the VAE with the PPO loss
        if self.configuration['ENABLE_VAE']:
            mean = tf.stop_gradient(mean)
            vae_feature_map_extraction = tf.stop_gradient(vae_feature_map_extraction)
        
        # Save the last image output, we can use it for tensorboard if we want
        self.last_vae_image_output = self.vae_decoder(mean)
        
        shared_model_output = self.actor_critic_shared_model([vae_feature_map_extraction])
        
        # The following code implements the RNN logic
        
        batch_size = tf.shape(vae_feature_map_extraction)[0]
        
        # Only relevant for WINDOWRNN architecture
        window_size = self.configuration.get("WINDOW_SIZE", 4)
        
        # From here on, we do our forward pass recurrently
        
        # TODO: Implement
        
        raise NotImplementedError
        
        return logits, new_critic_state + new_actor_state
        
    @override(TFModelV2)
    def get_initial_state(self):
        raise NotImplementedError
        # TODO: Implement

    def import_from_h5(self, h5_file: str) -> None:
        raise NotImplementedError
        # TODO: Implement
