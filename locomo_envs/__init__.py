import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='HalfCheetahCA-v0',
    entry_point='locomo_envs.envs.half_cheetah_ca2:HalfCheetahCAEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    # nondeterministic = True,
)

register(
    id='Walker2dCA-v0',
    entry_point='locomo_envs.envs.walker2d_ca:Walker2dCAEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    # nondeterministic = True,
)