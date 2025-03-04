import gymnasium as gym
import mujoco as mj
import numpy as np
import time
from PIL import Image
from utils import BestEpisodesVideoRecorder

import locomo_envs 


ds = [
    [1.0] * 6,
    [1.41, 0.96, 1.97, 1.73, 1.97, 1.17],
    [1.52, 1.07, 1.11, 1.97, 1.51, 0.99],
    [1.08, 1.18, 1.39, 1.76 , 1.85, 0.92],
    [0.85, 1.54, 0.97, 1.38, 1.10, 1.49],
]

# env = gym.make('HalfCheetahCA-v0', ctrl_cost_weight=0.1, render_mode='rgb_array', xml_file='./n_half_cheetah.xml')
# env = gym.make('HalfCheetahCA-v0', ctrl_cost_weight=0.1, render_mode='human', xml_file='./half_cheetah.xml') 
# env = gym.make('HalfCheetahCA-v0', render_mode='human') 
env = gym.make('HalfCheetahCA-v0', render_mode='rgb_array') 
# env = gym.make('HalfCheetah-v5', render_mode='rgb_array') 
# env = gym.make('Walker2d-v5', render_mode='human') 
# env = gym.make('HalfCheetahCA-v0', render_mode='rgb_array', xml_file='./half_cheetah.xml') 
# print(type(env.unwrapped.design()))
# observation, info = env.reset(options={'design': ds[2], 'random': False})
# observation, info = env.reset()

# image = env.render()
# im = Image.fromarray(image)
# im.show()
# trigger = lambda t: True
# env = gym.wrappers.RecordVideo(env=env, video_folder='./Vids', episode_trigger=trigger)
recorder = BestEpisodesVideoRecorder(path='./NewVids', max_videos=3)

for ep in range(1):
    observation, info = env.reset(options={'design': ds[1], 'random': True})
    # state = np.append(observation, np.array(env.unwrapped.design))
    # recorder.reset(env, state, 0, False)
    for i in range(200):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        state = np.append(observation, np.array(env.unwrapped.design))
        recorder.step(env, state, reward, terminated)

        # print(reward)
        # print(observation)
recorder.reset(env, state, reward, terminated)
env.close()