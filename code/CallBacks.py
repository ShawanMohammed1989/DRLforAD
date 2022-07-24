"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines callbacks for our tune trials.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
"""

import pprint

pp = pprint.PrettyPrinter(indent=4)


def on_episode_start(info):
    hist_data = info["episode"].hist_data
    user_data = info["episode"].user_data

    user_data["action_prob"] = []
    user_data["action1"] = []
    user_data["action2"] = []
    user_data["action3"] = []
    user_data["advantages"] = []
    user_data["behaviour_logits"] = []
    user_data["dones"] = []
    user_data["rewards"] = []
    user_data["seq_lens"] = []
    user_data["value_targets"] = []
    user_data["vf_preds"] = []

    user_data["last_vae_image_output"] = []
    user_data["last_vae_image_input"] = []

    hist_data["action_prob"] = []
    hist_data["action1"] = []
    hist_data["action2"] = []
    hist_data["action3"] = []
    hist_data["advantages"] = []
    hist_data["behaviour_logits"] = []
    hist_data["dones"] = []
    hist_data["rewards"] = []
    hist_data["seq_lens"] = []
    hist_data["value_targets"] = []
    hist_data["vf_preds"] = []
    hist_data["last_vae_input"] = []
    hist_data["last_vae_output"] = []


def on_episode_step(info):
    episode = info["episode"]
    episode.user_data["action_prob"] = []
    last_actions = episode.last_action_for()
    episode.user_data["action1"].append(last_actions[0])
    episode.user_data["action2"].append(last_actions[1])
    episode.user_data["action3"].append(last_actions[2])
    episode.user_data["advantages"] = []
    episode.user_data["behaviour_logits"] = []
    episode.user_data["dones"] = []
    episode.user_data["seq_lens"] = []
    episode.user_data["value_targets"] = []
    episode.user_data["vf_preds"] = []


def on_episode_end(info):
    hist_data = info["episode"].hist_data
    user_data = info["episode"].user_data
    hist_data["action_prob"] = user_data['action_prob']
    hist_data["action1"] = user_data["action1"]
    hist_data["action2"] = user_data["action2"]
    hist_data["action3"] = user_data["action3"]
    hist_data["advantages"] = user_data['advantages']
    hist_data["behaviour_logits"] = user_data['behaviour_logits']
    hist_data["dones"] = user_data['dones']
    hist_data["rewards"] = user_data['rewards']
    hist_data["seq_lens"] = user_data['seq_lens']
    hist_data["value_targets"] = user_data['value_targets']
    hist_data["vf_preds"] = user_data['vf_preds']

    user_data["reward"] = info['episode'].total_reward


def on_sample_end(info):
    print("returned sample batch of size {}".format(info["samples"].count))


def on_train_result(info):
    pass


def on_postprocess_traj(info):
    episode = info["episode"]
    batch = info["post_batch"]  # note: you can mutate this
    print("postprocessed {} steps".format(batch.count))


callbacks = {
    "on_episode_start": on_episode_start,
    "on_episode_end": on_episode_end,
    "on_episode_step": on_episode_step,
    "on_sample_end": on_sample_end,
    "on_train_result": on_train_result,
    "on_postprocess_traj": on_postprocess_traj,
}
