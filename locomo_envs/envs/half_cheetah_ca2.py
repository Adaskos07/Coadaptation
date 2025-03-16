import numpy as np
from gymnasium import utils

from locomo_envs.envs.mujoco_ca_env2 import MujocoCAEnv
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

HalfCheetahEnv.__bases__ = (MujocoCAEnv, utils.EzPickle)

class HalfCheetahCAEnv(HalfCheetahEnv):
    def __init__(self,
                 xml_file = "half_cheetah.xml",
                 frame_skip = 5,
                 default_camera_config = DEFAULT_CAMERA_CONFIG,
                 forward_reward_weight = 1, ctrl_cost_weight = 0.1,
                 reset_noise_scale = 0.1,
                 exclude_current_positions_from_observation = True,
                 **kwargs):
        super().__init__(xml_file,
                         frame_skip,
                         default_camera_config,
                         forward_reward_weight,
                         ctrl_cost_weight,
                         reset_noise_scale,
                         exclude_current_positions_from_observation,
                         **kwargs)

        self._design = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.bounds = (0.5, 1.5)
        # self.bounds = (0.8, 2)

    @property
    def design(self):
        return self._design
    
    @property
    def design_size(self):
        return len(self._design)
    
    # def _get_obs(self):
    #     position = self.data.qpos.flatten()
    #     velocity = self.data.qvel.flatten()

    #     if self._exclude_current_positions_from_observation:
    #         position = position[1:]

    #     # observation = np.concatenate((position, velocity)).ravel()
    #     observation = position.ravel()
    #     return observation

    def change_spec(self, options):
        if options is None:
            return False

        height = 1.0
        bth_r, bsh_r, bfo_r = 1.0, 1.0, 1.0
        fth_r, fsh_r, ffo_r = 1.0, 1.0, 1.0
        if options.get('random'):
            self._design = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=6)
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = self._design
        # elif options.get('design'):
        elif options.get('design') is not None:
            self._design = options['design']
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = self._design
            height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r + .106 * fsh_r +  .07 * ffo_r)
            height *= 2.0 + 0.01
        else:
            return False

        # dspec = copy(self.init_dspec)
        # dspec = mujoco.MjSpec.from_file(self.fullpath)
        # dspec = self.init_dspec
        torso = self.dspec.worldbody.first_body()
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

        return True
