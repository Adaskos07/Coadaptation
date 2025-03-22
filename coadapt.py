import os
import time
import csv

import numpy as np
import torch

import utils

from coadapt_env import CoadaptEnv
from RL.soft_actor import SoftActorCritic
from RL.evoreplay import EvoReplayLocalGlobalStart
from DO.pso_batch import PSO_batch
from DO.pso_sim import PSO_simulation


def select_design_opt_alg(alg_name):
    """ Selects the design optimization method.

    Args:
        alg_name: String which states the design optimization method. Can be
            `pso_batch` or `pso_sim`.

    Returns:
        The class of a design optimization method.

    Raises:
        ValueError: If the string alg_name is unknown.
    """
    if alg_name == "pso_batch":
        return PSO_batch
    elif alg_name == "pso_sim":
        return PSO_simulation
    else:
        raise ValueError("Design Optimization method not found.")

def select_rl_alg(rl_name):
    """ Selectes the reinforcement learning method.

    Args:
        rl_name: Name (string) of the rl method.

    Returns:
        The class of a reinforcement learning method.

    Raises:
        ValueError: If the string rl_name is unknown.
    """
    if rl_name == 'SoftActorCritic':
        return SoftActorCritic
    else:
        raise ValueError('RL method not fund.')


class Coadaptation:
    """ Co-Adaptaton algorithm class."""

    def __init__(self, config):
        self._config = config
        utils.move_to_cuda(self._config)

        self._episode_length = self._config['steps_per_episodes']
        self._reward_scale = self._config['reward_scale']

        self._env = CoadaptEnv(config=self._config)

        self._replay = EvoReplayLocalGlobalStart(self._env,
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7))

        self._rl_alg_class = select_rl_alg(self._config['rl_method'])

        self._networks = self._rl_alg_class.create_networks(env=self._env, config=config)

        self._rl_alg = self._rl_alg_class(config=self._config, env=self._env , replay=self._replay, networks=self._networks)

        self._do_alg_class = select_design_opt_alg(self._config['design_optim_method'])
        self._do_alg = self._do_alg_class(config=self._config, replay=self._replay, env=self._env)

        # if self._config['use_cpu_for_rollout']:
        #     utils.move_to_cpu()
        # else:
        #     utils.move_to_cuda(self._config)
        # # TODO this is a temp fix - should be cleaned up, not so hppy with it atm
        # self._policy_cpu = self._rl_alg_class.get_policy_network(SoftActorCritic.create_networks(env=self._env, config=config)['individual'])
        utils.move_to_cuda(self._config)

        self._last_single_iteration_time = 0
        self._design_counter = 0
        self._episode_counter = 0
        self._data_design_type = 'Initial'

    def initialize_episode(self):
        """ Initializations required before the first episode.

        Should be called before the first episode of a new design is
        executed. Resets variables such as _data_rewards for logging purposes
        etc.

        """
        # self._rl_alg.initialize_episode(init_networks = True, copy_from_gobal = True)
        self._rl_alg.episode_init()
        self._replay.reset_species_buffer()

        self._data_rewards = []
        self._episode_counter = 0

    def single_iteration(self):
        """ A single iteration.

        Makes all necessary function calls for a single iterations such as:
            - Collecting training data
            - Executing a training step
            - Evaluate the current policy
            - Log data

        """
        print("Time for one iteration: {}".format(time.time() - self._last_single_iteration_time))
        self._last_single_iteration_time = time.time()
        self._replay.set_mode("species")
        self.collect_training_experience()
        # TODO Change here to train global only after five designs
        train_pop = self._design_counter > 3
        if self._episode_counter >= self._config['initial_episodes']:
            self._rl_alg.single_train_step(train_ind=True, train_pop=train_pop)
        self._episode_counter += 1
        self.execute_policy()
        self.save_logged_data()
        self.save_networks()

    def collect_training_experience(self):
        """ Collect training data.

        This function executes a single episode in the environment using the
        exploration strategy/mechanism and the policy.
        The data, i.e. state-action-reward-nextState, is stored in the replay
        buffer.
        """
        state, _ = self._env.reset()
        nmbr_of_steps = 0
        done = False

        if self._episode_counter < self._config['initial_episodes']:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['population'])
        else:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['individual'])
        # self._policy_cpu = utils.copy_network(network_to=self._policy_cpu, network_from=policy_gpu_ind, config=self._config, force_cpu=self._config['use_cpu_for_rollout'])
        self._policy_cpu = policy_gpu_ind

        if self._config['use_cpu_for_rollout']:
            utils.move_to_cpu()
        else:
            utils.move_to_cuda(self._config)

        while not(done) and nmbr_of_steps <= self._episode_length:
            nmbr_of_steps += 1
            action, _ = self._policy_cpu.get_action(state)
            # new_state, reward, done, info = self._env.step(action)
            new_state, reward, truncated, terminated, _ = self._env.step(action)
            done = truncated or terminated
            # TODO this has to be fixed _variant_spec
            reward = reward * self._reward_scale
            terminal = np.array([done])
            reward = np.array([reward])
            self._replay.add_sample(observation=state, action=action,
                                    reward=reward, next_observation=new_state,
                                    terminal=terminal)
            state = new_state
        self._replay.terminate_episode()
        utils.move_to_cuda(self._config)

    def execute_policy(self):
        """ Evaluates the current deterministic policy.

        Evaluates the current policy in the environment by unrolling a single
        episode in the environment.
        The achieved cumulative reward is logged.
        """
        state, _ = self._env.reset()
        done = False
        reward_ep = 0.0
        reward_original = 0.0
        action_cost = 0.0
        nmbr_of_steps = 0

        if self._episode_counter < self._config['initial_episodes']:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['population'])
        else:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['individual'])
        # self._policy_cpu = utils.copy_network(network_to=self._policy_cpu, network_from=policy_gpu_ind, config=self._config, force_cpu=self._config['use_cpu_for_rollout'])
        self._policy_cpu = policy_gpu_ind

        if self._config['use_cpu_for_rollout']:
            utils.move_to_cpu()
        else:
            utils.move_to_cuda(self._config)

        while not(done) and nmbr_of_steps <= self._episode_length:
            nmbr_of_steps += 1
            # action, _ = self._policy_cpu.get_action(state, deterministic=True)
            action_dist, _ = self._policy_cpu.get_action(state)
            # action = action_dist.mean # makes it deterministic
            # print(action_dist)
            action = action_dist
            new_state, reward, truncated, terminated, info = self._env.step(action)
            done = truncated or terminated
            action_cost += info['orig_action_cost']
            reward_ep += float(reward)
            reward_original += float(info['orig_reward'])
            state = new_state
        utils.move_to_cuda(self._config)
        # Do something here to log the results
        self._data_rewards.append(reward_ep)

    def save_networks(self):
        """ Saves the networks on the disk.
        """
        if not self._config['save_networks']:
            return

        checkpoints_pop = {}
        for key, net in self._networks['population'].items():
            checkpoints_pop[key] = net.state_dict()

        checkpoints_ind = {}
        for key, net in self._networks['individual'].items():
            checkpoints_ind[key] = net.state_dict()

        checkpoint = {
            'population' : checkpoints_pop,
            'individual' : checkpoints_ind,
        }
        file_path = os.path.join(self._config['data_folder_experiment'], 'checkpoints')
        if not os.path.exists(file_path):
          os.makedirs(file_path)
        torch.save(checkpoint, os.path.join(file_path, 'checkpoint_design_{}.chk'.format(self._design_counter)))

    def load_networks(self, path):
        """ Loads netwokrs from the disk.
        """
        model_data = torch.load(path) #, map_location=ptu.device)

        model_data_pop = model_data['population']
        for key, net in self._networks['population'].items():
            params = model_data_pop[key]
            net.load_state_dict(params)

        model_data_ind = model_data['individual']
        for key, net in self._networks['individual'].items():
            params = model_data_ind[key]
            net.load_state_dict(params)

    def save_logged_data(self):
        """ Saves the logged data to the disk as csv files.

        This function creates a log-file in csv format on the disk. For each
        design an individual log-file is creates in the experient-directory.
        The first row states if the design was one of the initial designs
        (as given by the environment), a random design or an optimized design.
        The second row gives the design parameters (eta). The third row
        contains all subsequent cumulative rewards achieved by the policy
        throughout the reinforcement learning process on the current design.
        """
        file_path = self._config['data_folder_experiment']
        current_design = self._env.get_current_design()

        with open(os.path.join(file_path,'data_design_{}.csv'.format(self._design_counter)),
                  'w') as fd:
            cwriter = csv.writer(fd)
            cwriter.writerow(['Design Type:', self._data_design_type])
            cwriter.writerow(current_design)
            cwriter.writerow(self._data_rewards)

    def run(self):
        """ Runs the Fast Evolution through Actor-Critic RL algorithm.

        First the initial design loop is executed in which the rl-algorithm
        is exeuted on the initial designs. Then the design-optimization
        process starts.
        It is possible to have different numbers of iterations for initial
        designs and the design optimization process.
        """
        iterations_init = self._config['iterations_init']
        iterations = self._config['iterations']
        design_cycles = self._config['design_cycles']
        exploration_strategy = self._config['exploration_strategy']

        self._intial_design_loop(iterations_init)
        self._training_loop(iterations, design_cycles, exploration_strategy)

    def _training_loop(self, iterations, design_cycles, exploration_strategy):
        """ The trianing process which optimizes designs and policies.

        The function executes the reinforcement learning loop and the design
        optimization process.

        Args:
            iterations: An integer stating the number of iterations/episodes
                to be used per design during the reinforcement learning loop.
            design_cycles: Integer stating how many designs to evaluate.
            exploration_strategy: String which describes which
                design exploration strategy to use. Is not used at the moment,
                i.e. only the (uniform) random exploration strategy is used.
        """
        self.initialize_episode()
        # TODO fix the following
        initial_state, _ = self._env._env.reset()

        self._data_design_type = 'Optimized'

        optimized_params = self._env.get_random_design()
        q_network = self._rl_alg_class.get_q_network(self._networks['population'])
        policy_network = self._rl_alg_class.get_policy_network(self._networks['population'])
        optimized_params = self._do_alg.optimize_design(design=optimized_params, q_network=q_network, policy_network=policy_network)
        optimized_params = list(optimized_params)

        for i in range(design_cycles):
            self._design_counter += 1
            self._env.set_new_design(optimized_params)

            # Reinforcement Learning
            for _ in range(iterations):
                self.single_iteration()

            # Design Optimization
            if i % 2 == 1:
                self._data_design_type = 'Optimized'
                q_network = self._rl_alg_class.get_q_network(self._networks['population'])
                policy_network = self._rl_alg_class.get_policy_network(self._networks['population'])
                optimized_params = self._do_alg.optimize_design(design=optimized_params, q_network=q_network, policy_network=policy_network)
                optimized_params = list(optimized_params)
            else:
                self._data_design_type = 'Random'
                optimized_params = self._env.get_random_design()
                optimized_params = list(optimized_params)
            self.initialize_episode()

    def _intial_design_loop(self, iterations):
        """ The initial training loop for initial designs.

        The initial training loop in which no designs are optimized but only
        initial designs, provided by the environment, are used.

        Args:
            iterations: Integer stating how many training iterations/episodes
                to use per design.
        """
        self._data_design_type = 'Initial'
        for params in self._env.init_sim_params:
            self._design_counter += 1
            self._env.set_new_design(params)
            self.initialize_episode()

            # Reinforcement Learning
            for _ in range(iterations):
                self.single_iteration()
