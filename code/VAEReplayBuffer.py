"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines a LocalReplayBuffer that is very similar to RLLib's LocalReplayBuffer class but deepcopies batches
and also only uses observations from batches, since we only need those in order to train the VAE.
"""

import collections
import copy

from ray.rllib import SampleBatch
from ray.rllib.execution.replay_buffer import LocalReplayBuffer, ReplayBuffer, _ALL_POLICIES, warn_replay_buffer_size
from ray.rllib.policy.sample_batch import MultiAgentBatch, DEFAULT_POLICY_ID

"""
These are the keys for elements of an APPO experience. If values for these keys are needed, they may be extracted in
add() method of the VAEReplayBuffer
"""
BATCH_KEYS = ['t', 'eps_id', 'agent_index', 'obs', 'actions', 'rewards', 'prev_actions', 'prev_rewards', 'dones',
			  'infos', 'action_prob', 'action_logp', 'action_dist_inputs', 'vf_preds', 'unroll_id', 'advantages',
			  'value_targets']


class LocalVAEReplayBuffer(LocalReplayBuffer):
	"""A replay buffer shard.

	Ray actors are single-threaded, so for scalability multiple replay actors
	may be created to increase parallelism."""
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Overwrite regular replay buffer with our own
		def new_buffer():
			return VAEReplayBuffer(self.buffer_size)  # TODO: Maybe change later to prioritization
		
		self.replay_buffers = collections.defaultdict(new_buffer)
	
	def add_batch(self, batch, recursive_calls=0):
		"""
		The ordinary LocalReplayBuffer method add_batch() fails to copy the batch from time to time because the batch
		varies in while the copy is taking place. In order to not get the
		"dictionary changes size during iteration" error of its super-class, we recursively try to copy.
		:param batch: SampleBatch
		:param recursive_calls: int, leave at 0
		:return: None
		"""
		# deepcopy so that we do not get "dictionary changes size during iteration" error in super class
		try:
			copy_batch = copy.deepcopy(batch)  # TODO: Test copy instead of deepcopy
		except RuntimeError as e:
			if recursive_calls > 100:
				raise e
			return self.add_batch(batch, recursive_calls=recursive_calls + 1)
		
		return super().add_batch(copy_batch)
	
	def replay(self):
		"""
		A replay method that does not pass the "beta" parameter to the sample method
		:return: MultiAgentBatch
		"""
		if self._fake_batch:
			fake_batch = SampleBatch(self._fake_batch)
			return MultiAgentBatch({
				DEFAULT_POLICY_ID: fake_batch
			}, fake_batch.count)
		
		if self.num_added < self.replay_starts:
			return None
		
		with self.replay_timer:
			if self.replay_mode == "lockstep":
				return self.replay_buffers[_ALL_POLICIES].sample(
					self.replay_batch_size)
			else:
				samples = {}
				for policy_id, replay_buffer in self.replay_buffers.items():
					s_ = replay_buffer.sample(self.replay_batch_size)
					samples[policy_id] = SampleBatch(s_)
				return MultiAgentBatch(samples, self.replay_batch_size)


class VAEReplayBuffer(ReplayBuffer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		self._num_sampled = 0
		
	def add(self, item, weight):
		item = SampleBatch({"obs": item.data["obs"]})
		warn_replay_buffer_size(item=item, num_items=self._maxsize / item.count)
		assert item.count > 0, item
		self._num_timesteps_added += item.count
		self._num_timesteps_added_wrap += item.count
		
		if self._next_idx >= len(self._storage):
			self._storage.append(item)
			self._est_size_bytes += item.size_bytes()
		else:
			self._storage[self._next_idx] = item
		
		# Wrap around storage as a circular buffer once we hit maxsize.
		if self._num_timesteps_added_wrap >= self._maxsize:
			self._eviction_started = True
			self._num_timesteps_added_wrap = 0
			self._next_idx = 0
		else:
			self._next_idx += 1
		
		if self._eviction_started:
			self._evicted_hit_stats.push(self._hit_count[self._next_idx])
			self._hit_count[self._next_idx] = 0
