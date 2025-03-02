__credits__ = ["Kallinteris-Andreas", "Rushiv Arora", "Adam Sulak"]
from copy import deepcopy, copy
from typing import Dict, Union
import numpy as np

from gymnasium import utils
from locomo_envs.envs.mujoco_ca_env import MujocoCAEnv
from gymnasium.spaces import Box

import mujoco


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahCAEnv(MujocoCAEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoCAEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {"x_position": x_position_after, "x_velocity": x_velocity, **reward_info}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        qpos = self.init_qpos 
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
        }
    
    def change_spec(self, options):
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
            return

        # dspec = copy(self.init_dspec)
        # dspec = mujoco.MjSpec.from_file(self.fullpath)
        dspec = self.init_dspec
        torso = dspec.worldbody.first_body()
        torso.pos = (0, 0, height)

        # back
        back_thigh = torso.bodies[0]
        back_shin = back_thigh.first_body()
        back_foot = back_shin.first_body()

        back_thigh.first_geom().pos = (.1*bth_r, 0, -.13*bth_r)
        back_thigh.first_geom().size = (.046, .145*bth_r, 0)

        back_shin.pos = (.16*bth_r, 0, -.25*bth_r)
        back_shin.first_geom().pos = (-.14*bsh_r, 0, -.07*bsh_r)
        back_shin.first_geom().size = (.046, .15*bsh_r, 0)

        back_foot.pos = (-.28*bsh_r, 0, -.14*bsh_r)
        back_foot.first_geom().pos = (.03*bfo_r, 0, -.097*bfo_r)
        back_foot.first_geom().size = (.046, .094*bfo_r, 0)

        # front
        front_thigh = torso.bodies[1]
        front_shin = front_thigh.first_body()
        front_foot = front_shin.first_body()

        front_thigh.first_geom().pos = (-.07*fth_r, 0, -.12*fth_r)
        front_thigh.first_geom().size = (.046, .133*fth_r, 0)

        front_shin.pos = (-.14*fth_r, 0, -.24*fth_r)
        front_shin.first_geom().pos = (.065*fsh_r, 0, -.09*fsh_r)
        front_shin.first_geom().size = (.046, .106*fsh_r, 0)

        front_foot.pos = (.13*fsh_r, 0, -.18*fsh_r)
        front_foot.first_geom().pos = (.045*ffo_r, 0, -.07*ffo_r)
        front_foot.first_geom().size = (.046, .07*ffo_r, 0)

        # dspec not spec!
        # self.model = dspec.compile()
        # self.model.vis.global_.offwidth = self.width
        # self.model.vis.global_.offheight = self.height
        # self.data = mujoco.MjData(self.model)

        self.model, self.data = dspec.recompile(self.model, self.data)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        # self.mujoco_renderer.model = self.model
        # self.mujoco_renderer.data = self.data


        # self.mujoco_renderer.scn = mujoco.MjvScene(self.model, 1000)
        # self.mujoco_renderer.viewer = self.mujoco_renderer._get_viewer('human')
        # self.init_qpos = self.data.qpos.ravel().copy()
        # self.init_qvel = self.data.qvel.ravel().copy()
        # self.model, _ = self.dspec.recompile(self.model, self.data)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()


