from typing import Tuple, Optional
import mujoco
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


class MujocoCAEnv(MujocoEnv):
    def __init__(self,
                 model_path,
                 frame_skip,
                 observation_space,
                 render_mode = None,
                 width = ...,
                 height = ...,
                 camera_id = None,
                 camera_name = None,
                 default_camera_config = None,
                 max_geom = 1000,
                 visual_options = ...):
        super().__init__(model_path,
                         frame_skip,
                         observation_space,
                         render_mode,
                         width,
                         height,
                         camera_id,
                         camera_name,
                         default_camera_config,
                         max_geom,
                         visual_options)

    def _initialize_simulation(self)-> Tuple["mujoco.MjModel", "mujoco.MjData", "mujoco.MjSpec"]:
        """
        Initialize MuJoCo simulation data structures `mjModel`, `mjData` and `mjSpec`.
        """
        # name 'spec' is taken by gym.Env
        dspec = mujoco.MjSpec.from_file(self.fullpath)
        # DELING WITH PARENT CLASS THAT DEOSNT HAVE DSPEC
        self.dspec = dspec
        model = dspec.compile()

        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data
    
    def _update_model_and_renderer(self):
        self.model, self.data = self.dspec.recompile(self.model, self.data)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # renderer might throw segfault on wayland when running .close()
        self.mujoco_renderer.model = self.model
        self.mujoco_renderer.data = self.data
        self.mujoco_renderer._get_viewer(self.render_mode)
        self.mujoco_renderer.viewer.model = self.model
        self.mujoco_renderer.viewer.data = self.data
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        update_made = self.change_spec(options)
        if update_made:
            self._update_model_and_renderer()
        else:
            mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info
    
    # added methods to overwrite
    def change_spec(self, options):
        """
        Method for changing underlying mujoco MjSpec object named self.dspec
        """
        raise NotImplementedError

    @property 
    def design(self):
        raise NotImplementedError