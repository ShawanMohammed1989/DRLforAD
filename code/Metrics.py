"""
rllib for students at the ICE by Artur Niederfahrenhorst
The content of this file is derived from ray.rllib.execution.metric_ops.
It offers slightly modified metrics reporting, such that we can see all rewards.
# TODO: Review this file
"""

from typing import Any, List

from ray.rllib.execution.metric_ops import OncePerTimestepsElapsed, OncePerTimeInterval
from ray.util.iter import LocalIterator
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, _get_shared_metrics
from ray.rllib.evaluation.worker_set import WorkerSet


def CustomMetricsReporting(
        train_op: LocalIterator[Any],
        workers: WorkerSet,
        config: dict,
        selected_workers: List["ActorHandle"] = None) -> LocalIterator[dict]:
    """Operator to periodically collect and report custom metrics.

    Arguments:
        train_op (LocalIterator): Operator for executing training steps.
            We ignore the output values.
        workers (WorkerSet): Rollout workers to collect metrics from.
        config (dict): Trainer configuration, used to determine the frequency
            of stats reporting.
        selected_workers (list): Override the list of remote workers
            to collect metrics from.

    Returns:
        A local iterator over training results.
    """

    output_op = train_op \
        .filter(OncePerTimestepsElapsed(config["timesteps_per_iteration"])) \
        .filter(OncePerTimeInterval(config["min_iter_time_s"])) \
        .for_each(CollectCustomMetrics(
            workers, min_history=config["metrics_smoothing_episodes"],
            timeout_seconds=config["collect_metrics_timeout"],
            selected_workers=selected_workers))
    return output_op


class CollectCustomMetrics:
    """Callable that collects rewards only from workers.
    The metrics are smoothed over a given history window.

    This should be used with the .for_each() operator. For a higher level
    API, consider using StandardMetricsReporting instead.
    """

    def __init__(self,
                 workers,
                 min_history=100,
                 timeout_seconds=180,
                 selected_workers=None):
        self.workers = workers
        self.episode_history = []
        self.to_be_collected = []
        self.min_history = min_history
        self.timeout_seconds = timeout_seconds
        self.selected_workers = selected_workers

    def __call__(self, _):
        # Collect worker metrics.
        episodes, self.to_be_collected = collect_episodes(
            self.workers.local_worker(),
            self.selected_workers or self.workers.remote_workers(),
            self.to_be_collected,
            timeout_seconds=self.timeout_seconds)
        orig_episodes = list(episodes)
        missing = self.min_history - len(episodes)
        if missing > 0:
            episodes.extend(self.episode_history[-missing:])
            assert len(episodes) <= self.min_history
        self.episode_history.extend(orig_episodes)
        self.episode_history = self.episode_history[-self.min_history:]
        res = summarize_episodes(episodes, orig_episodes)

        # Add in iterator metrics.
        metrics = _get_shared_metrics()
        timers = {}
        counters = {}
        info = {}
        info.update(metrics.info)
        for k, counter in metrics.counters.items():
            counters[k] = counter
        for k, timer in metrics.timers.items():
            timers["{}_time_ms".format(k)] = round(timer.mean * 1000, 3)
            if timer.has_units_processed():
                timers["{}_throughput".format(k)] = round(
                    timer.mean_throughput, 3)
        res.update({
            "num_healthy_workers": len(self.workers.remote_workers()),
            "timesteps_total": metrics.counters[STEPS_SAMPLED_COUNTER],
        })
        res["timers"] = timers
        res["info"] = info
        res["info"].update(counters)

        episode_rewards = []
        for episode in episodes:
            episode_rewards.append(episode.episode_reward)

        res["info"].update({"episode_rewards": episode_rewards})

        return res
