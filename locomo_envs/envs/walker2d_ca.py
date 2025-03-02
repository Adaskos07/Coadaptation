from typing import Tuple, Dict, Union
import numpy as np
from gymnasium import utils

from locomo_envs.envs.mujoco_ca_env2 import MujocoCAEnv
from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

Walker2dEnv.__bases__ = (MujocoCAEnv, utils.EzPickle)


class Walker2dCAEnv(Walker2dEnv):
    def __init__(self,
                 xml_file: str = "walker2d_v5.xml",
                 frame_skip: int = 4,
                 default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
                 forward_reward_weight: float = 1,
                 ctrl_cost_weight: float = 0.001,
                 healthy_reward: float = 1,
                 terminate_when_unhealthy: bool = True,
                 healthy_z_range: Tuple[float, float] = (0.8, 2.0),
                 healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
                 reset_noise_scale: float = 0.005,
                 exclude_current_positions_from_observation: bool = True,
                 **kwargs):
        super().__init__(xml_file,
                         frame_skip,
                         default_camera_config,
                         forward_reward_weight,
                         ctrl_cost_weight,
                         healthy_reward,
                         terminate_when_unhealthy,
                         healthy_z_range,
                         healthy_angle_range,
                         reset_noise_scale,
                         exclude_current_positions_from_observation,
                         **kwargs)
    
    def change_spec(self, options):
        if options is None:
            return False

        height = 1.0
        bth_r, bsh_r, bfo_r = 1.0, 1.0, 1.0
        fth_r, fsh_r, ffo_r = 1.0, 1.0, 1.0
        if options['random']:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = np.random.uniform(low=0.5, high=1.5, size=6)
        elif options['design']:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = options['design']
            height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r + .106 * fsh_r +  .07 * ffo_r)
            height *= 2.0 + 0.01
        else:
            return False