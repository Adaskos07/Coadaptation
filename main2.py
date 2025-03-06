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

env = gym.make('HalfCheetahCA-v0', render_mode='human') 
# env = gym.make('HalfCheetahCA-v0', render_mode='rgb_array') 
# env = gym.make('HalfCheetah-v5', render_mode='rgb_array') 
# env = gym.make('Walker2d-v5', render_mode='human') 
# print(type(env.unwrapped.design()))
# observation, info = env.reset(options={'design': ds[2], 'random': False})
# observation, info = env.reset()

# image = env.render()
# im = Image.fromarray(image)
# im.show()
# trigger = lambda t: True
# env = gym.wrappers.RecordVideo(env=env, video_folder='./Vids', episode_trigger=trigger)
# recorder = BestEpisodesVideoRecorder(path='./NewVids', max_videos=3)

# for ep in range(1):

observation, info = env.reset(options={'design': ds[1], 'random': False})
print(env.unwrapped.design)
print(env.unwrapped.design_size)


exit()
# state = np.append(observation, np.array(env.unwrapped.design))
# recorder.reset(env, state, 0, False)
for i in range(200):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    state = np.append(observation, np.array(env.unwrapped.design))
    # recorder.step(env, state, reward, terminated)

    # print(reward)
    # print(observation)

# observation, info = env.reset(options={'design': ds[4], 'random': True})
observation, info = env.reset()
print(env.unwrapped.design)
# state = np.append(observation, np.array(env.unwrapped.design))
for i in range(200):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    state = np.append(observation, np.array(env.unwrapped.design))



# recorder.reset(env, state, reward, terminated)
env.close()