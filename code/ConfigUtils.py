"""
rllib for students at the ICE by Artur Niederfahrenhorst
yaml config files suitable for this project can
be read and written with these methods.
"""

import yaml
import tensorflow as tf
import ray.tune as tune
import logging

logger = logging.getLogger("config_logger")

PATH_GENERAL_CONFIG = "config/general_config.yaml"
PATH_RAY_CONFIG = "config/ray_config.yaml"
PATH_RLLIB_CONFIG = "config/rllib_config.yaml"
PATH_TUNE_CONFIG = "config/tune_config.yaml"


def add_state_dimension(config_):
	"""
	Calculate the state dimension which depends on the image dimensions.
	:return: state dimension
	"""
	state_dim = config_.get('OBSERVATION_DIMENSION')
	if state_dim:
		if config_.get('CROPIMAGE'):
			state_dim = tuple([config_['TO_Y'] - config_['FROM_Y'],
			                   config_['TO_X'] - config_['FROM_X'], state_dim[2]])
		else:
			state_dim = tuple(config_['OBSERVATION_DIMENSION'])

		config_['STATE_DIM'] = state_dim
	
	return config_


def turn_parameters_into_objects(config_):
	"""
	Turn the config parameters into an objects.
	:param config_: config object
	:return:processed config
	"""
	# Layer weight initializers
	if config_.get('INITIALIZER') == "glorot_uniform":
		config_['INITIALIZER'] = tf.initializers.glorot_uniform()
	elif config_.get('INITIALIZER') == "he_normal":
		config_['INITIALIZER'] = tf.initializers.he_normal()
	
	# Layer weight regularizers
	if config_.get('REGULARIZER') == "l1_l2":
		config_['REGULARIZER'] = tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-2)
	elif config_.get('REGULARIZER') == "l1":
		config_['REGULARIZER'] = tf.keras.regularizers.l1(1e-4)
	elif config_.get('REGULARIZER') == "l2":
		config_['REGULARIZER'] = tf.keras.regularizers.l2(1e-2)
	else:
		config_['REGULARIZER'] = None
	
	return config_


def extract_tunable_parameters(config_):
	"""
	Turn the config parameters into tune parameters.
	:param config_: config object
	:return:processed config
	"""
	
	def turn_into_tune_parameter(name, value):
		if name.startswith("LOGUNIFORM_"):
			return {name.replace("LOGUNIFORM_", ""): tune.loguniform(value[0], value[1])}
		elif name.startswith("UNIFORM_"):
			return {name.replace("UNIFORM_", ""): tune.uniform(value)}
		elif name.startswith("GRID_SEARCH_"):
			return {name.replace("GRID_SEARCH_", ""): tune.grid_search(value)}
		else:
			return {}
	
	new_config = {}
	
	for name, value in config_.items():
		new_config.update(turn_into_tune_parameter(name, value))
	
	return new_config


def read_config(path):
	"""
	Parse a config file and preprocess it.
	:param path: config file path
	:return: preprocessed config
	"""
	
	stream = open(path, 'r')
	config_ = dict(yaml.safe_load(stream))
	return config_


def read_general_config(path):
	"""
	Parse the config file and preprocess it.
	:param path: config file path
	:return: preprocessed config
	"""
	
	config_ = read_config(path)
	
	recurrent_layers_in_model = sum(config_.get("CRITIC_CNN_GRU_LAYERS", None) +
	                                config_.get("CRITIC_FC_GRU_LAYERS", None) +
	                                config_.get("ACTOR_CNN_GRU_LAYERS", None) +
	                                config_.get("ACTOR_FC_GRU_LAYERS", None))
	# Check config and add state dimension
	if recurrent_layers_in_model and (config_.get("NETWORK_ARCHITECTURE", None) == "SEQUENTIAL"):
		raise ValueError("SEQUENTIAL network does not support config reccurent layers")
	
	config_ = turn_parameters_into_objects(config_)
	
	return config_


def read_tune_config(path):
	"""
	Parse a config file.
	:param path: config file path
	:return: preprocessed config
	"""
	config_ = read_config(path)
	
	tunable_parameters_config = extract_tunable_parameters(config_)
	
	return tunable_parameters_config, config_


def read_environment_config(path):
	"""
	Parse a config file and preprocess it.
	:param path: config file path
	:return: preprocessed config
	"""
	
	stream = open(path, 'r')
	config_ = dict(yaml.safe_load(stream))
	config_ = add_state_dimension(config_)
	config_ = turn_parameters_into_objects(config_)
	return config_


def read_ray_rllib_general_configs():
	"""
	Read all configs and check for collisions.
	
	"""
	config = {}
	get_intersecting_keys = lambda a: set(a.keys()) & set(config.keys())
	intersecting_keys = []
	
	# Read new config file, intersect with last one, update intersecting_keys list and overall config
	def intersect_and_update(_config):
		for key in get_intersecting_keys(_config):
			logger.warning("Configuration key " + key + " found, but define already before. Consider cleaning up!")
		config.update(_config)
	
	tmp = read_general_config(PATH_GENERAL_CONFIG)
	intersect_and_update(tmp)
	
	tmp = read_config(PATH_RAY_CONFIG)
	intersect_and_update(tmp)
	
	tmp = read_config(PATH_RLLIB_CONFIG)
	intersect_and_update(tmp)
	
	# Try to read the environment config
	try:
		tmp = read_environment_config("environments/" + config["ENVIRONMENT_FOLDER_NAME"] + "/environment_config.yaml")
		intersect_and_update(tmp)
	except KeyError:
		logger.warning("Please specify ENVIRONMENT_CONFIG_NAME correctly, preferably in the general_config.yaml.")
		exit(1)
	
	return config


def write_config(config_, path):
	"""
	Write a yaml config file.
	:param config_: config object
	:param path: path to write config to
	:return: None
	"""
	with open(path, 'w') as file:
		yaml.dump(config_, file)
