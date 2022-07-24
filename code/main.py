"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines the main entry point for our framework.
"""
import warnings
import logging
import tensorflow as tf
import ray
import ray.rllib.agents.ppo.appo as appo
import argparse
import importlib
import os
import sys
import datetime

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import AsyncHyperBandScheduler

from ray.rllib.models.preprocessors import NoPreprocessor
from ConfigUtils import read_ray_rllib_general_configs, read_tune_config
from CustomModel import CustomModel, RNNModel
from ExecutionPlan import LoadableCustomAPPOTrainerTrainer

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = tf.get_logger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
# Use these arguments for parameters that are different across multiple machines/instances of this program
# Use the general_config.yaml for parameters that are the same for all machines/instances
parser.add_argument("--tune", action="store_true", help="If set, run in parameter tuning mode (see tune_config.yaml)")
parser.add_argument("--resume-trials-from-dir", type=str, default=None,
                    help="Load trials to resume tuning after after a crash")
parser.add_argument("--debug", action="store_true", default=False,
                    help="Run ray in local mode (inside one process) to enable better debugging. ")
parser.add_argument("--experiment-name", type=str, default="unnamed_trial",
                    help="The name of the experiment")

if __name__ == '__main__':
    # Get configs and parameters
    args = parser.parse_args()
    
    default_config = appo.DEFAULT_CONFIG.copy()
    merged_configs = read_ray_rllib_general_configs()
    
    ray.init(num_gpus=merged_configs.get("NUM_GPUS", 0) + merged_configs.get("VAE_NUM_GPUS", 0),
             num_cpus=merged_configs.get("NUM_CPUS", 0), object_store_memory=merged_configs.get("OBJECT_STORE_MEMORY"))
    
    try:
        environment = importlib.import_module("environments." + merged_configs["ENVIRONMENT_FOLDER_NAME"]).CustomEnvironment
    except Exception as e:
        print(e)
        logger.critical("CustomEnvironment not properly defined at " + merged_configs["ENVIRONMENT_FOLDER_NAME"])
        exit(1)

    try:
        CustomModel = importlib.import_module("environments." + merged_configs["ENVIRONMENT_FOLDER_NAME"]).CustomModel
    except:
        logger.info("No CustomModel defined at " + merged_configs["ENVIRONMENT_FOLDER_NAME"])
        
    try:
        RNNModel = importlib.import_module("environments." + merged_configs["ENVIRONMENT_FOLDER_NAME"]).RNNModel
    except:
        logger.info("No RNNModel defined at " + merged_configs["ENVIRONMENT_FOLDER_NAME"])
        
    simuator_interface_import_string = merged_configs.get('SHARED_SIMULATOR_INTERFACE')
    if simuator_interface_import_string:
        try:
            SimulatorInterface = importlib.import_module("environments." + merged_configs["ENVIRONMENT_FOLDER_NAME"]).SimulatorInterface
            @ray.remote
            class RemoteSimulatorInterface(SimulatorInterface):
                pass
            simulator_interface_handle = RemoteSimulatorInterface.remote(merged_configs)
            merged_configs['SIMULATOR_INTERFACE_HANDLE'] = simulator_interface_handle
        except AttributeError:
            logger.warning("No Simulator defined in folder " +
                           merged_configs["ENVIRONMENT_FOLDER_NAME"] + ". Proceeding without.")
    
    # Register our models in the model catalogue
    ModelCatalog.register_custom_model("SEQUENTIAL_MODEL", CustomModel)
    ModelCatalog.register_custom_model("RNN_MODEL", RNNModel)
    
    logdir = os.path.join(os.path.dirname(__file__), "..", "results")
    if args.debug:
        logdir = os.path.join(logdir, "debug", "run_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    elif args.tune:
        logdir = os.path.join(logdir, "tune", "run_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    else:
        logdir = os.path.join(logdir, "experiment",
        args.experiment_name or "run_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        
    merged_configs["logdir"] = logdir
    merged_configs["gamma"] = merged_configs.get("GAMMA", 0.99)
    merged_configs["lambda"] = merged_configs.get("LAMBDA", 1.0)
    merged_configs["epsilon"] = merged_configs.get("EPSILON", 0.1)
    
    training_config = {
        "env": environment,
        "model": {
            "custom_model": "SEQUENTIAL_MODEL",
            # This config is parsed by RLLib. Since only certain sub-dictionaries are allowed to have unknown keys,
            # we just put all our items in the model config.
            "custom_model_config": merged_configs,
            "max_seq_len": merged_configs['MAX_TIME_SEQUENCE_LENGTH'],
            "dim": merged_configs.get('RLLIB_PREPROCESSOR_DIM'),
            "grayscale": merged_configs.get('GRAYSCALE', False),
            "framestack": merged_configs.get('FRAMESTACK', False),
            "zero_mean": merged_configs.get("MEANSHIFT", True),
        },
        "preprocessor_pref": "deepmind",
        "num_workers": merged_configs.get("NUMBER_OF_WORKERS"),
        "env_config": merged_configs,
        "num_gpus": merged_configs.get("NUM_GPUS", 0),
        "train_batch_size": merged_configs['TRAIN_BATCH_SIZE'],
        "rollout_fragment_length": merged_configs['ROLLOUT_FRAGMENT_LENGTH'],
        "min_iter_time_s": merged_configs['MIN_ITER_TIME_S'],
        "log_level": merged_configs['LOG_LEVEL'],
        "framework": merged_configs['FRAMEWORK'],
        "vtrace": merged_configs['VTRACE'],
    }
    
    config = {**default_config, **training_config}
    
    scheduler = None  # Scheduler will default to FIFO, if None is chosen by tune
    
    
    # We use this creator to track our experiments and put them all in one folder
    trial_counter = 0
    def trial_name_str_creator(trial):
        global trial_counter
        trial_counter = trial_counter + 1
        return args.experiment_name + "_trial_no_" + str(trial_counter)

    if args.debug:
        # If none of the above "special" modes are active, simply train
        
        config["num_workers"] = 1
        config["log_level"] = logging.DEBUG
        
        
        tune.run(
            LoadableCustomAPPOTrainerTrainer,
            local_dir=logdir,
            log_to_file=True,
            name="",
            config=config
        )
        
    else:
        # The folowing code sets up a trials scheduler according to the tune_config.yaml
        tunable_parameters, tune_config = read_tune_config("config/tune_config.yaml")
        
        if args.tune:
            config['env_config'] = {**config['env_config'], **tunable_parameters}
            
            scheduler = AsyncHyperBandScheduler(
                time_attr='training_iteration',
                metric='episode_reward_mean',
                mode='max',
                max_t=tune_config.get('ASYNC_HP_MAX_T'),
                grace_period=tune_config.get('ASYNC_HP_GRACE_PERIOD'),
                reduction_factor=tune_config.get('ASYNC_HP_REDUCTION_FACTOR'),
                brackets=tune_config.get('ASYNC_HP_BRACKETS')
            )
            
            stop = {
                "training_iteration": config.get('STOP_AT_TRAINING_ITER_X', sys.maxsize),
                "timesteps_total": config.get('STOP_AT_TOTAL_TIMESTEPS_X', sys.maxsize),
                "episode_reward_mean": config.get('STOP_AT_MEAN_REWARD_X', sys.maxsize),
            }
            
            # Run our tune trial(s)
            tune.run(
                LoadableCustomAPPOTrainerTrainer,
                local_dir=logdir,
                queue_trials=False,
                scheduler=scheduler,
                config=config,
                num_samples=tune_config.get('NUM_SAMPLES'),
                stop=stop,
                trial_name_creator=trial_name_str_creator,
                trial_dirname_creator=trial_name_str_creator,
                restore=args.resume_trials_from_dir
            )
        
        # Evaluation mode. This performs a series of trial trainings that can then be evaluated
        else:
            # Run some experiment to average statistics over
            analysis = tune.run(
                LoadableCustomAPPOTrainerTrainer,
                local_dir=logdir,
                queue_trials=False,
                num_samples=tune_config['EXP_NUM_EXPERIMENTS'],
                name="rllib-for-students-evaluation",
                config=config,
                stop={"episodes_total": tune_config['EXP_NUM_TRAINING_EPISODES']},
                trial_name_creator=trial_name_str_creator,
                trial_dirname_creator=trial_name_str_creator,
                restore=args.resume_trials_from_dir
            )
            
            # config = {**config, **{"evaluation_config": {"explore": False}}}  # Test this. It does probably not take effect
            
            #
            # for trial_index, analysis_ in enumerate(tune_runs):
            # 	trial = analysis_.trials[0]
            # 	trainer = CustomAPPOTrainer(config=config)
            # 	trainer.restore(trial.checkpoint.value)
            # 	output_dir = "~/ray_results/experiments/" + args.trial_name + "/rollout" + str(trial_index)
            # 	os.makedirs(output_dir, exist_ok=True)
            # 	with RolloutSaver(
            # 			output_dir + "/evaluation", str(trial_index),
            # 			use_shelve=False,
            # 			write_update_file=False,
            # 			target_episodes=merged_configs['EXP_NUM_EVAL_EPISODES'],
            # 			save_info=True) as saver:
            # 		rollout(trainer, args.env, 999999999, merged_configs['EXP_NUM_EVAL_EPISODES'], no_render=False)
            # 	path = trainer.save(checkpoint_dir=output_dir)
            # 	trainer.stop()
            # 	print("Model parameters saved at" + path)
        
    ray.shutdown()
