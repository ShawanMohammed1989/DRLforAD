"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines a artificial neural networks that are used by our custom RLLib model to learn on an environment.
"""

import os
import warnings
import logging
import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer

logger = tf.get_logger()
logger.setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def build_networks(configuration):
    """
	Build a joined network for actor and critic.
	Return the network and it's sub-networks.
	The network structure is explained in Kemal's thesis.

	Inputs are:
	- observation_input

	Outputs are:
	- actions mean
	- actions var
	- vae_decoder_output
	- vae_mean
	- vae_logvar
	- vae_z
	"""
    
    # VAE Encoder:
    
    # VAE Encoder Image Input
    image_input = tf.keras.layers.Input(shape=list(configuration['STATE_DIM']))
    
    # First set all of the potentially concatenated inputs to None, so that they we later not be added to the model
    layer = image_input
    
    for i in range(configuration['VAE_CNN_LAYERS']):
        layer = tf.keras.layers.Conv2D(filters=configuration['VAE_CNN_FILTERS'][i],
                                       kernel_size=configuration['VAE_CNN_KERNEL_SIZE'][i],
                                       strides=configuration['VAE_CNN_STRIDES'][i],
                                       activation=tf.nn.relu,
                                       name='vae_shared' + '_SharedCNNLayer_' + str(i),
                                       kernel_initializer=configuration['INITIALIZER'],
                                       padding=configuration['VAE_CNN_PADDING'][i],
                                       kernel_regularizer=configuration['REGULARIZER'])(layer)
    
    vae_feature_extraction_layer = layer
    layer = tf.keras.layers.Flatten()(layer)
    
    # Create mean and logvar from encoder output (from feature extraction layer), this is our latent vector
    vae_mean = tf.keras.layers.Dense(units=configuration['VAE_LATENT_DIM'],
                                     kernel_initializer=configuration['INITIALIZER'],
                                     name='Mean',
                                     kernel_regularizer=configuration['REGULARIZER'])(layer)
    vae_logvar = tf.keras.layers.Dense(units=configuration['VAE_LATENT_DIM'],
                                       kernel_initializer=configuration['INITIALIZER'],
                                       name='Logvar', activation=tf.nn.softplus,
                                       kernel_regularizer=configuration['REGULARIZER'])(layer)
    
    # VAE Decoder:
    
    # VAE Decoder Latent Space Input
    decoder_z_input = tf.keras.layers.Input(shape=[50])
    
    layer = tf.keras.layers.Dense(units=vae_feature_extraction_layer.shape[1] *
                                        vae_feature_extraction_layer.shape[2] *
                                        vae_feature_extraction_layer.shape[3],
                                  name='Dense_Layer_1',
                                  kernel_initializer=configuration['INITIALIZER'],
                                  kernel_regularizer=configuration['REGULARIZER'])(decoder_z_input)
    layer = tf.keras.layers.Reshape([vae_feature_extraction_layer.shape[1],
                                     vae_feature_extraction_layer.shape[2],
                                     vae_feature_extraction_layer.shape[3]])(layer)
    # Build layers to transpose-convolute back to original input dimension
    for i1 in range(configuration['VAE_CNN_LAYERS']):
        shifted_i1 = - configuration['VAE_CNN_LAYERS'] + i1
        layer = tf.keras.layers.Conv2DTranspose(filters=configuration['VAE_CNN_DECODER_FILTERS'][shifted_i1],
                                                kernel_size=configuration['VAE_CNN_DECODER_KERNEL_SIZE'][shifted_i1],
                                                strides=configuration['VAE_CNN_DECODER_STRIDES'][shifted_i1],
                                                activation=tf.nn.relu,
                                                padding=configuration['VAE_CNN_DECODER_PADDING'][shifted_i1],
                                                kernel_initializer=configuration['INITIALIZER'],
                                                kernel_regularizer=configuration['REGULARIZER'])(layer)

    layer = tf.keras.layers.Conv2DTranspose(filters=3,
                                            kernel_size=3,
                                            strides=1,
                                            padding="same",
                                            kernel_initializer=configuration['INITIALIZER'],
                                            kernel_regularizer=configuration['REGULARIZER'])(layer)
    
    image_output = tf.math.tanh(layer)[:, :, :, 0:3]
    concat_dim = 3  # Dimension where a value is expected to be concatenated
    # PPO Actor and Critic:
    
    # PPO Actor and Critic Shared Layers
    
    actor_critic_feature_map_input = tf.keras.layers.Input(shape=[vae_feature_extraction_layer.shape[1],
																  vae_feature_extraction_layer.shape[2],
																  vae_feature_extraction_layer.shape[3]])
    
    layer = actor_critic_feature_map_input
    for i in range(configuration['SHARED_CNN_LAYERS']):
        layer = tf.keras.layers.Conv2D(filters=configuration['SHARED_CNN_FILTERS'][i],
                                       kernel_size=configuration['SHARED_CNN_KERNEL_SIZE'][i],
                                       strides=configuration['SHARED_CNN_STRIDES'][i],
                                       activation=tf.nn.relu,
                                       name='Actor_Critic' + '_SharedCNNLayer_' + str(i),
                                       kernel_initializer=configuration['INITIALIZER'],
                                       padding='same',
                                       kernel_regularizer=configuration['REGULARIZER'])(layer)
    
    actor_critic_shared_cnn_output = layer
    
    # PPO Actor Critic Inputs
    
    actor_feature_map_input = tf.keras.layers.Input(shape=[actor_critic_shared_cnn_output.shape[1],
                                                           actor_critic_shared_cnn_output.shape[2],
                                                           actor_critic_shared_cnn_output.shape[3]])
    critic_feature_map_input = tf.keras.layers.Input(shape=[actor_critic_shared_cnn_output.shape[1],
                                                           actor_critic_shared_cnn_output.shape[2],
                                                           actor_critic_shared_cnn_output.shape[3]])
    
    # Build lists of states, since we work with keras GRU cells
    # Since not every layer is stateful, we need only as many states as we have stateful layers
    
    # Critic state lists
    critic_cnn_gru_state_inputs = [(a * b or None) and tf.keras.layers.Input(shape=[a * b]) for a, b in
                                   zip(configuration.get("CRITIC_GRU_STATE_SIZES"),
                                       configuration.get("CRITIC_CNN_GRU_LAYERS"))]
    
    critic_fc_gru_state_inputs = [(a * b or None) and tf.keras.layers.Input(shape=[a * b]) for a, b in
                                  zip(configuration.get("CRITIC_GRU_STATE_SIZES"),
                                      configuration.get("CRITIC_FC_GRU_LAYERS"))]
    
    # Actor state lists
    actor_cnn_gru_state_inputs = [(a * b or None) and tf.keras.layers.Input(shape=[a * b]) for a, b in
                                  zip(configuration.get("ACTOR_GRU_STATE_SIZES"),
                                      configuration.get("ACTOR_CNN_GRU_LAYERS"))]
    
    actor_fc_gru_state_inputs = [(a * b or None) and tf.keras.layers.Input(shape=[a * b]) for a, b in
                                 zip(configuration.get("ACTOR_GRU_STATE_SIZES"),
                                     configuration.get("ACTOR_FC_GRU_LAYERS"))]
    
    value, critic_gru_fc_state_outputs, critic_gru_cnn_state_outputs = build_critic_net(x=critic_feature_map_input,
                                                                                        configuration=configuration,
                                                                                        critic_cnn_gru_states=critic_cnn_gru_state_inputs,
                                                                                        critic_fc_gru_states=critic_fc_gru_state_inputs)
    
    action_layer, actor_gru_fc_state_outputs, actor_gru_cnn_state_outputs = build_actor_net(x=actor_feature_map_input,
                                                                                            configuration=configuration,
                                                                                            actor_cnn_gru_states=actor_cnn_gru_state_inputs,
                                                                                            actor_fc_gru_states=actor_fc_gru_state_inputs)
    
    # Define Models
    encoder_inputs = [i for i in [image_input] if i is not None]
    encoder = tf.keras.Model(encoder_inputs,
	                         [vae_mean, vae_logvar, vae_feature_extraction_layer],
	                         name='VAE_Encoder')
    
    decoder_outputs = [i for i in [image_output] if
                       i is not None]
    decoder = tf.keras.Model([decoder_z_input],
                             decoder_outputs,
                             name='VAE_Decoder')
    
    actor_critic_shared = tf.keras.Model([actor_critic_feature_map_input],
                                         [actor_critic_shared_cnn_output],
                                         name='Actor_Critic_Shared_Model')
    
    # Reduce our state inputs
    actor_cnn_gru_state_inputs = [i for i in actor_cnn_gru_state_inputs if i is not None]
    actor_fc_gru_state_inputs = [i for i in actor_fc_gru_state_inputs if i is not None]
    actor = tf.keras.Model([actor_feature_map_input, actor_cnn_gru_state_inputs, actor_fc_gru_state_inputs],
                           [action_layer, actor_gru_cnn_state_outputs, actor_gru_fc_state_outputs],
                           name='Actor_Model')
    
    # Reduce our state inputs
    critic_cnn_gru_state_inputs = [i for i in critic_cnn_gru_state_inputs if i is not None]
    critic_fc_gru_state_inputs = [i for i in critic_fc_gru_state_inputs if i is not None]
    critic = tf.keras.Model([critic_feature_map_input, critic_cnn_gru_state_inputs, critic_fc_gru_state_inputs],
                            [value, critic_gru_cnn_state_outputs, critic_gru_fc_state_outputs],
                            name='Critic_Model')
    
    return encoder, decoder, actor_critic_shared, actor, critic


def build_actor_net(x, configuration, actor_cnn_gru_states, actor_fc_gru_states):
    # These two lists tell us which layers should be stateful
    actor_rnn_cnn_layers = configuration.get("ACTOR_CNN_GRU_LAYERS") or [0] * 64
    actor_rnn_fc_layers = configuration.get("ACTOR_FC_GRU_LAYERS") or [0] * 64
    
    # Append actor CNN layers to graph
    gru_cnn_state_outputs = []
    for i in range(configuration['ACTOR_CNN_LAYERS']):
        actor_cnn_layer_params = {"filters": configuration['ACTOR_CNN_FILTERS'][i],
                                  "kernel_size": configuration['ACTOR_CNN_KERNEL_SIZE'][i],
                                  "strides": configuration['ACTOR_CNN_STRIDES'][i],
                                  "activation": tf.nn.relu,
                                  "name": "ActorCNNLayer_CNN_" + str(i),
                                  "kernel_initializer": configuration['INITIALIZER'],
                                  "kernel_regularizer": configuration['REGULARIZER']}
        
        if actor_rnn_cnn_layers[i]:
            raise NotImplementedError("Currently, there is no Conv2D GRU Layer available. "
                                      "The available LSTM layer has not been tested.")
            x = tf.keras.layers.ConvLSTM2D(**actor_cnn_layer_params)(x)
        else:
            x = tf.keras.layers.Conv2D(**actor_cnn_layer_params)(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    # Append actor FC layers to graph
    gru_fc_state_outputs = []
    recurrent_layer_counter = 0
    for i in range(configuration['ACTOR_FC_LAYERS']):
        actor_layer_params = {"units": configuration['ACTOR_FC_UNITS'][i],
                              "activation": tf.nn.relu,
                              "name": 'ActorLayer_Dense_' + str(i),
                              "kernel_initializer": configuration['INITIALIZER'],
                              "kernel_regularizer": configuration['REGULARIZER']}
        
        if actor_rnn_fc_layers[i]:
            actor_layer_params.update({"name": "ActorGRULayer_DenseGRU_" + str(i),
                                       "units": configuration["ACTOR_GRU_STATE_SIZES"][i]})
            state = actor_fc_gru_states[recurrent_layer_counter]
            
            x, state = tf.keras.layers.GRUCell(**actor_layer_params)(x, state)
            
            recurrent_layer_counter += 1
            gru_fc_state_outputs.append(state)
        else:
            x = tf.keras.layers.Dense(**actor_layer_params)(x)
    
    x = tf.keras.layers.Dense(configuration["OUTPUTS"] * 2,
                              name='ActorLayer_last',
                              kernel_initializer=normc_initializer(0.01),
                              kernel_regularizer=configuration['REGULARIZER'])(x)
    
    return x, gru_fc_state_outputs, gru_cnn_state_outputs


def build_critic_net(x, configuration, critic_cnn_gru_states, critic_fc_gru_states):
    # These two lists tell us which layers should be stateful
    critic_rnn_cnn_layers = configuration.get("CRITIC_CNN_GRU_LAYERS") or [0] * 64
    critic_rnn_fc_layers = configuration.get("CRITIC_FC_GRU_LAYERS") or [0] * 64
    
    # Append critic CNN layers to graph
    gru_cnn_state_outputs = []
    for i in range(configuration['CRITIC_CNN_LAYERS']):
        critic_cnn_layer_params = {"filters": configuration['CRITIC_CNN_FILTERS'][i],
                                   "kernel_size": configuration['CRITIC_CNN_KERNEL_SIZE'][i],
                                   "strides": configuration['CRITIC_CNN_STRIDES'][i],
                                   "activation": tf.nn.relu,
                                   "name": "CriticCNNLayer_CNN_" + str(i),
                                   "kernel_initializer": configuration['INITIALIZER'],
                                   "kernel_regularizer": configuration['REGULARIZER']}
        
        if critic_rnn_cnn_layers[i]:
            raise NotImplementedError("Currently, there is no Conv2D GRU Layer available. "
                                      "The available LSTM layer has not been tested.")
            x = tf.keras.layers.ConvLSTM2D(**critic_cnn_layer_params)(x)
        
        else:
            x = tf.keras.layers.Conv2D(**critic_cnn_layer_params)(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    # Append critic FC layers to graph
    gru_fc_state_outputs = []
    recurrent_layer_counter = 0
    for i in range(configuration['CRITIC_FC_LAYERS']):
        critic_layer_params = {"units": configuration['CRITIC_FC_UNITS'][i],
                               "activation": tf.nn.relu,
                               "name": 'CriticLayer_Dense_' + str(i),
                               "kernel_initializer": configuration['INITIALIZER'],
                               "kernel_regularizer": configuration['REGULARIZER']}
        
        if critic_rnn_fc_layers[i]:
            critic_layer_params.update({"name": "CriticGRULayer_DenseGRU_" + str(i),
                                        "units": configuration["CRITIC_GRU_STATE_SIZES"][i]})
            state = critic_fc_gru_states[recurrent_layer_counter]
            
            x, state = tf.keras.layers.GRUCell(**critic_layer_params)(x, state)
            
            recurrent_layer_counter += 1
            gru_fc_state_outputs.append(state)
        else:
            x = tf.keras.layers.Dense(**critic_layer_params)(x)
    
    value_func = tf.keras.layers.Dense(1, name='value', kernel_initializer=configuration['INITIALIZER'],
                                       kernel_regularizer=configuration['REGULARIZER'])(x)
    
    return value_func, gru_fc_state_outputs, gru_cnn_state_outputs








