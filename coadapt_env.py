import copy

from gymnasium.spaces import Box
import gymnasium as gym
import numpy as np

import locomo_envs
from utils import BestEpisodesVideoRecorder


class CoadaptEnv:
    """ Wraps chosen environment so it can work with Coadapation algorithm """
    def __init__(self, config):
        self._config = config
        self._render = self._config['env']['render']
        self._record_video = self._config['env']['record_video']

        try:
            env_name = self._config['env']['env_name']
        except KeyError as e:
            raise KeyError("Missing 'env_name' config parameter!")

        self._env = gym.make(env_name, render_mode='rgb_array')

        self._design_size = self._env.unwrapped.design_size
        self._config_numpy = np.array(self._env.unwrapped.design)
        self.design_params_bounds = [(0.8, 2.0)] * self._design_size

        self.init_sim_params = [
            [1.0] * 6,
            [1.41, 0.96, 1.97, 1.73, 1.97, 1.17],
            [1.52, 1.07, 1.11, 1.97, 1.51, 0.99],
            [1.08, 1.18, 1.39, 1.76 , 1.85, 0.92],
            [0.85, 1.54, 0.97, 1.38, 1.10, 1.49],
        ]
        self.observation_space = Box(-np.inf, np.inf,
                                     shape=[self._env.observation_space.shape[0] + self._design_size],
                                     dtype=np.float32) #env.observation_space
        self.action_space = self._env.action_space
        self._initial_state = self._env.reset()

        if self._record_video:
            self._video_recorder = BestEpisodesVideoRecorder(path=config['data_folder_experiment'], max_videos=5)

        # Which dimensions in the state vector are design parameters?
        self._design_dims = list(range(self.observation_space.shape[0] - self._design_size, self.observation_space.shape[0]))
        assert len(self._design_dims) == self._design_size
    
    def render(self):
        pass

    def step(self, action):
        info = {}
        state, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        # state = state[0]
        state = np.append(state, self._config_numpy)
        info['orig_action_cost'] = 0.1 * np.mean(np.square(action))
        info['orig_reward'] = reward

        if self._record_video:
            self._video_recorder.step(env=self._env, state=state, reward=reward, done=done)

        # return state, reward, False, False, info
        return state, reward, False, False, info

    def reset(self):
        state, info = self._env.reset()
        self._initial_state = state
        state = np.append(state, self._config_numpy)

        if self._record_video:
            self._video_recorder.reset(env=self._env, state=state, reward=0, done=False)

        return state, info

    def set_new_design(self, vec):
        self._env.reset(options={'design': vec})
        self._config_numpy = np.array(vec)

        if self._record_video:
            self._video_recorder.increase_folder_counter()

    def get_random_design(self):
        optimized_params = np.random.uniform(low=0.8, high=2.0, size=self._design_size)
        return optimized_params

    def get_current_design(self):
        return copy.copy(self._env.unwrapped.design)

    def get_design_dimensions(self):
        return copy.copy(self._design_dims)
