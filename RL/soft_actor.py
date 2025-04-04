import numpy as np
import torch

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.sac import SACTrainer
import rlkit.torch.pytorch_util as ptu

from .rl_algorithm import RL_algorithm
import utils


# networks = {individual:, population:}
class SoftActorCritic(RL_algorithm):
    def __init__(self, config, env, replay, networks):
        """ Bascally a wrapper class for SAC from rlkit.

        Args:
            config: Configuration dictionary
            env: Environment
            replay: Replay buffer
            networks: dict containing two sub-dicts, 'individual' and 'population'
                which contain the networks.
        """
        super().__init__(config, env, replay, networks)

        self._variant_pop = config['rl_algorithm_config']['algo_params_pop']
        self._variant_spec = config['rl_algorithm_config']['algo_params']

        self._ind_qf1 = networks['individual']['qf1']
        self._ind_qf2 = networks['individual']['qf2']
        self._ind_qf1_target = networks['individual']['qf1_target']
        self._ind_qf2_target = networks['individual']['qf2_target']
        self._ind_policy = networks['individual']['policy']

        self._pop_qf1 = networks['population']['qf1']
        self._pop_qf2 = networks['population']['qf2']
        self._pop_qf1_target = networks['population']['qf1_target']
        self._pop_qf2_target = networks['population']['qf2_target']
        self._pop_policy = networks['population']['policy']

        self._batch_size = config['rl_algorithm_config']['batch_size']
        self._nmbr_indiv_updates = config['rl_algorithm_config']['indiv_updates']
        self._nmbr_pop_updates = config['rl_algorithm_config']['pop_updates']

        self._algorithm_ind = SACTrainer(
            env=self._env,
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            use_automatic_entropy_tuning = False,
            **self._variant_spec
        )

        self._algorithm_pop = SACTrainer(
            env=self._env,
            policy=self._pop_policy,
            qf1=self._pop_qf1,
            qf2=self._pop_qf2,
            target_qf1=self._pop_qf1_target,
            target_qf2=self._pop_qf2_target,
            use_automatic_entropy_tuning = False,
            **self._variant_pop
        )

        # self._algorithm_ind.to(ptu.device)
        # self._algorithm_pop.to(ptu.device)

    def episode_init(self):
        """ Initializations to be done before the first episode.

        In this case basically creates a fresh instance of SAC for the
        individual networks and copies the values of the target network.
        """
        self._algorithm_ind = SACTrainer(
            env=self._env,
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            use_automatic_entropy_tuning = False,
            # alt_alpha = self._alt_alpha,
            **self._variant_spec
        )
        if self._config['rl_algorithm_config']['copy_from_global']:
            utils.copy_pop_to_ind(networks_pop=self._networks['population'], networks_ind=self._networks['individual'])
        # We have only to do this because the version of rlkit which we use
        # creates internally a target network
        # vf_dict = self._algorithm_pop.target_vf.state_dict()
        # self._algorithm_ind.target_vf.load_state_dict(vf_dict)
        # self._algorithm_ind.target_vf.eval()
        # self._algorithm_ind.to(ptu.device)

    def single_train_step(self, train_ind=True, train_pop=False):
        """ A single trianing step.

        Args:
            train_ind: Boolean. If true the individual networks will be trained.
            train_pop: Boolean. If true the population networks will be trained.
        """
        if train_ind:
          # Get only samples from the species buffer
          self._replay.set_mode('species')
          # self._algorithm_ind.num_updates_per_train_call = self._variant_spec['num_updates_per_epoch']
          # self._algorithm_ind._try_to_train()
          for _ in range(self._nmbr_indiv_updates):
              batch = self._replay.random_batch(self._batch_size)
              self._algorithm_ind.train(batch)

        if train_pop:
          # Get only samples from the population buffer
          self._replay.set_mode('population')
          # self._algorithm_pop.num_updates_per_train_call = self._variant_pop['num_updates_per_epoch']
          # self._algorithm_pop._try_to_train()
          for _ in range(self._nmbr_pop_updates):
              batch = self._replay.random_batch(self._batch_size)
              self._algorithm_pop.train(batch)

    @staticmethod
    def create_networks(env, config):
        """ Creates all networks necessary for SAC.

        These networks have to be created before instantiating this class and
        used in the constructor.

        Args:
            config: A configuration dictonary containing population and
                individual networks

        Returns:
            A dictonary which contains the networks.
        """
        network_dict = {
            'individual' : SoftActorCritic._create_networks(env=env, config=config),
            'population' : SoftActorCritic._create_networks(env=env, config=config),
        }
        return network_dict

    @staticmethod
    def _create_networks(env, config):
        """ Creates all networks necessary for SAC.

        These networks have to be created before instantiating this class and
        used in the constructor.

        TODO: Maybe this should be reworked one day...

        Args:
            config: A configuration dictonary.

        Returns:
            A dictonary which contains the networks.
        """
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        net_size = config['rl_algorithm_config']['net_size']
        hidden_sizes = [net_size] * config['rl_algorithm_config']['network_depth']
        qf1 = ConcatMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf2 = ConcatMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf1_target = ConcatMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf2_target = ConcatMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        policy = TanhGaussianPolicy(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
        ).to(device=ptu.device)

        clip_value = 1.0
        for p in qf1.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in qf2.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return {'qf1' : qf1, 'qf2' : qf2, 'qf1_target' : qf1_target, 'qf2_target' : qf2_target, 'policy' : policy}

    @staticmethod
    def get_q_network(networks):
        """ Returns the q network from a dict of networks.

        This method extracts the q-network from the dictonary of networks
        created by the function create_networks.

        Args:
            networks: Dict containing the networks.

        Returns:
            The q-network as torch object.
        """
        return networks['qf1']

    @staticmethod
    def get_policy_network(networks):
        """ Returns the policy network from a dict of networks.

        This method extracts the policy network from the dictonary of networks
        created by the function create_networks.

        Args:
            networks: Dict containing the networks.

        Returns:
            The policy network as torch object.
        """
        return networks['policy']
