# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Agent wrapper for CyberBattle envrionments exposing additional
features extracted from the environment observations"""

from apt_cyber_sim._env.cyberbattle_env import EnvironmentBounds
import apt_cyber_sim._env.cyberbattle_env as cyberbattle_env
from typing import Optional, List
import enum
import numpy as np
from gym import spaces, Wrapper
from numpy import ndarray
import logging


class StateAugmentation:
    """Default agent state augmentation, consisting of the gym environment
    observation itself and nothing more."""

    def __init__(self, observation: cyberbattle_env.Observation):
        self.observation = observation

    def on_step(self, action: cyberbattle_env.Action, reward: float, done: bool,
                observation: cyberbattle_env.Observation):
        self.observation = observation

    def on_reset(self, observation: cyberbattle_env.Observation):
        self.observation = observation


class Feature(spaces.MultiDiscrete):
    """
    Feature consisting of multiple discrete dimensions.
    Parameters:
        nvec: is a vector defining the number of possible values
        for each discrete space.
    """

    def __init__(self, env_properties: EnvironmentBounds, nvec):
        self.env_properties = env_properties
        super().__init__(nvec)

    def flat_size(self):
        return np.prod(self.nvec)

    def name(self):
        """Return the name of the feature"""
        p = len(type(Feature(self.env_properties, [])).__name__) + 1
        return type(self).__name__[p:]

    def get(self, a: StateAugmentation, node: Optional[int]) -> np.ndarray:
        """Compute the current value of a feature value at
        the current observation and specific node"""
        raise NotImplementedError

    def pretty_print(self, v):
        return v


class Feature_active_node_properties(Feature):
    """Bitmask of all properties set for the active node"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.property_count)

    def get(self, a: StateAugmentation, node) -> ndarray:
        assert node is not None, 'feature only valid in the context of a node'

        node_prop = a.observation['discovered_nodes_properties']

        # list of all properties set/unset on the node
        # Remap to get rid of unknown value 0: 1 -> 1, and -1 -> 0 (and 0-> 0)
        assert node < len(node_prop), f'invalid node index {node} (not discovered yet)'
        remapped = np.array((1 + node_prop[node]) / 2, dtype=np.int)
        return remapped


class Feature_active_node_age(Feature):
    """How recently was this node discovered?
    (measured by reverse position in the list of discovered nodes)"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count])

    def get(self, a: StateAugmentation, node) -> ndarray:
        assert node is not None, 'feature only valid in the context of a node'

        discovered_node_count = len(a.observation['discovered_nodes_properties'])

        assert node < discovered_node_count, f'invalid node index {node} (not discovered yet)'

        return np.array([discovered_node_count - node - 1], dtype=np.int)


class Feature_active_node_id(Feature):
    """Return the node id itself"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count] * 1)

    def get(self, a: StateAugmentation, node) -> ndarray:
        return np.array([node], dtype=np.int)


class Feature_discovered_nodeproperties_sliding(Feature):
    """Bitmask indicating node properties seen in last few cache entries"""
    window_size = 3

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.property_count)

    def get(self, a: StateAugmentation, node) -> ndarray:
        node_prop = np.array(a.observation['discovered_nodes_properties'])

        # keep last window of entries
        node_prop_window = node_prop[-self.window_size:, :]

        # Remap to get rid of unknown value 0: 1 -> 1, and -1 -> 0 (and 0-> 0)
        node_prop_window_remapped = np.int32((1 + node_prop_window) / 2)

        countby = np.sum(node_prop_window_remapped, axis=0)

        bitmask = (countby > 0) * 1
        return bitmask


class Feature_discovered_ports(Feature):
    """Bitmask vector indicating each port seen so far in discovered credentials"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.port_count)

    def get(self, a: StateAugmentation, node):
        ccm = a.observation['credential_cache_matrix']
        known_credports = np.zeros(self.env_properties.port_count, dtype=np.int32)
        if a.observation['credential_cache_length'] > 0:
            known_credports[np.int32(ccm[:, 1])] = 1
        return known_credports


# class Feature_discovered_ports_sliding(Feature):
#     """Bitmask indicating port seen in last few cache entries"""
#     window_size = 3
#
#     def __init__(self, p: EnvironmentBounds):
#         super().__init__(p, [2] * p.port_count)
#
#     def get(self, a: StateAugmentation, node):
#         ccm = a.observation['credential_cache_matrix']
#         known_credports = np.zeros(self.env_properties.port_count, dtype=np.int32)
#         known_credports[np.int32(ccm[-self.window_size:, 1])] = 1
#         return known_credports


class Feature_discovered_ports_counts(Feature):
    """Count of each port seen so far in discovered credentials"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_total_credentials + 1] * p.port_count)

    def get(self, a: StateAugmentation, node):
        ccm = a.observation['credential_cache_matrix']
        if a.observation['credential_cache_length'] == 0:
            return np.zeros(self.env_properties.port_count)
        return np.bincount(np.int32(ccm[:, 1]), minlength=self.env_properties.port_count) / self.env_properties.maximum_node_count


class Feature_discovered_credential_count(Feature):
    """number of credentials discovered so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_total_credentials + 1])

    def get(self, a: StateAugmentation, node):
        return [a.observation['credential_cache_length'] / self.env_properties.maximum_total_credentials]


class Feature_discovered_node_count(Feature):
    """number of nodes discovered so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count + 1])

    def get(self, a: StateAugmentation, node):
        return [(len(a.observation['discovered_nodes_properties'])-len(a.observation['nodes_privilegelevel'])) / self.env_properties.maximum_node_count]
        #return [len(a.observation['discovered_nodes_properties']) / self.env_properties.maximum_node_count]


class Feature_owned_node_count(Feature):
    """number of owned nodes so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count + 1])

    def get(self, a: StateAugmentation, node):
        levels = a.observation['nodes_privilegelevel']
        owned_nodes_indices = np.where(levels > 0)[0]
        #return [len(owned_nodes_indices) / self.env_properties.maximum_node_count]
        return [(self.env_properties.maximum_total_credentials-len(owned_nodes_indices)+1) / self.env_properties.maximum_node_count]


class Feature_success_actions_all_nodes(Feature):
    """success actions count"""

    def __init__(self, p: EnvironmentBounds):
        self.max_action_count = max(p.local_attacks_count, p.remote_attacks_count)
        super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)

    def get(self, a: StateAugmentation, node) -> np.ndarray:
        vector = a.success_action_count.tolist()
        for i_action in range(len(vector)):
            if i_action < self.env_properties.maximum_node_count:  # local attack
                #vector[i_action] = vector[i_action] / self.env_properties.local_attacks_count
                vector[i_action] = 1 - vector[i_action]
            elif i_action < self.env_properties.maximum_node_count * 2:  # remote attack
                #vector[i_action] = vector[i_action] / self.env_properties.remote_attacks_count
                vector[i_action] = (2 - vector[i_action]) / 2
            else:
                #vector[i_action] = vector[i_action] / self.env_properties.maximum_node_count
                vector[i_action] = (a.observation['credential_cache_length']-vector[i_action]) / self.env_properties.maximum_total_credentials
            if vector[i_action] > 1:
                exit(-1)
        return vector


class Feature_failed_actions_all_nodes(Feature):
    """failed actions count"""

    #max_action_count = 100
    max_action_count = 100

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)

    def get(self, a: StateAugmentation, node) -> np.ndarray:
        #vector = np.minimum(a.failed_action_count, self.max_action_count) / self.max_action_count
        vector = np.minimum(a.failed_action_count, self.max_action_count) / self.max_action_count
        return vector


class Feature_all_credential_detail(Feature):
    """credential details"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_total_credentials + 1] * p.maximum_node_count)

    def get(self, a: StateAugmentation, node):
        vector = np.zeros(self.env_properties.maximum_node_count, dtype=np.int32)
        for i in range(a.observation['credential_cache_length']):
            t = a.observation['credential_cache_matrix'][i]
            vector[int(t[0])] += 1
        vector = vector / self.env_properties.maximum_total_credentials
        return vector


class Feature_all_port_detail(Feature):
    """port details"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.port_count + 1] * p.maximum_node_count)

    def get(self, a: StateAugmentation, node):
        port_vis_list = [set() for _ in range(self.env_properties.maximum_node_count)]
        vector = np.zeros(self.env_properties.maximum_node_count, dtype=np.int32)
        for i in range(a.observation['credential_cache_length']):
            t = a.observation['credential_cache_matrix'][i]
            node_id, port_id = int(t[0]), t[1]
            if port_id not in port_vis_list[node_id]:
                vector[node_id] += 1
                port_vis_list[node_id].add(port_id)
        vector = vector / self.env_properties.port_count
        return vector


class Feature_all_node_conquer_state(Feature):
    """whether nodes are owned"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.maximum_node_count)

    def get(self, a: StateAugmentation, node):
        return np.array(a.observation['all_nodes_conquer_state'])


class Feature_all_node_profile(Feature):
    """profile of all nodes so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.maximum_node_count * p.property_count)

    def get(self, a: StateAugmentation, node):
        return np.array(a.observation['all_nodes_properties'])


class ConcatFeatures(Feature):
    """ Concatenate a list of features into a single feature
    Parameters:
        feature_selection - a selection of features to combine
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature]):
        self.feature_selection = feature_selection
        self.dim_sizes = np.concatenate([f.nvec for f in feature_selection])
        super().__init__(p, [self.dim_sizes])

    def pretty_print(self, v):
        return v

    def get(self, a: StateAugmentation, node=None) -> np.ndarray:
        """Return the feature vector"""
        feature_vector = [f.get(a, node) for f in self.feature_selection]
        #print(feature_vector)
        return np.concatenate(feature_vector)


class FeatureEncoder(Feature):
    """ Encode a list of featues as a unique index
    """

    feature_selection: List[Feature]

    def vector_to_index(self, feature_vector: np.ndarray) -> int:
        raise NotImplementedError

    def feature_vector_of_observation_at(self, a: StateAugmentation, node: Optional[int]) -> np.ndarray:
        """Return the current feature vector"""
        feature_vector = [f.get(a, node) for f in self.feature_selection]
        # print(f'feature_vector={feature_vector}  self.feature_selection={self.feature_selection}')
        return np.concatenate(feature_vector)

    def feature_vector_of_observation(self, a: StateAugmentation):
        return self.feature_vector_of_observation_at(a, None)

    def encode(self, a: StateAugmentation, node=None) -> int:
        """Return the index encoding of the feature"""
        feature_vector_concat = self.feature_vector_of_observation_at(a, node)
        return self.vector_to_index(feature_vector_concat)

    def encode_at(self, a: StateAugmentation, node) -> int:
        """Return the current feature vector encoding with a node context"""
        feature_vector_concat = self.feature_vector_of_observation_at(a, node)
        return self.vector_to_index(feature_vector_concat)

    def get(self, a: StateAugmentation, node=None) -> np.ndarray:
        """Return the feature vector"""
        return np.array([self.encode(a, node)])

    def name(self):
        """Return a name for the feature encoding"""
        n = ', '.join([f.name() for f in self.feature_selection])
        return f'[{n}]'


class HashEncoding(FeatureEncoder):
    """ Feature defined as a hash of another feature
    Parameters:
       feature_selection: a selection of features to combine
       hash_dim: dimension after hashing with hash(str(feature_vector)) or -1 for no hashing
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature], hash_size: int):
        self.feature_selection = feature_selection
        self.hash_size = hash_size
        super().__init__(p, [hash_size])

    def flat_size(self):
        return self.hash_size

    def vector_to_index(self, feature_vector) -> int:
        """Hash the state vector"""
        return hash(str(feature_vector)) % self.hash_size

    def pretty_print(self, index):
        return f'#{index}'


class RavelEncoding(FeatureEncoder):
    """ Combine a set of features into a single feature with a unique index
     (calculated by raveling the original indices)
    Parameters:
        feature_selection - a selection of features to combine
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature]):
        self.feature_selection = feature_selection
        self.dim_sizes = np.concatenate([f.nvec for f in feature_selection])
        self.ravelled_size: int = np.prod(self.dim_sizes)
        assert np.shape(self.ravelled_size) == (), f'! {np.shape(self.ravelled_size)}'
        super().__init__(p, [self.ravelled_size])

    def vector_to_index(self, feature_vector):
        assert len(self.dim_sizes) == len(feature_vector), \
            f'feature vector of size {len(feature_vector)}, ' \
            f'expecting {len(self.dim_sizes)}: {feature_vector} -- {self.dim_sizes}'
        index: np.int32 = np.ravel_multi_index(feature_vector, self.dim_sizes)
        assert index < self.ravelled_size, \
            f'feature vector out of bound ({feature_vector}, dim={self.dim_sizes}) ' \
            f'-> index={index}, max_index={self.ravelled_size - 1})'
        return index

    def unravel_index(self, index) -> np.ndarray:
        return np.unravel_index(index, self.dim_sizes)

    def pretty_print(self, index):
        return self.unravel_index(index)


def owned_nodes(observation):
    """Return the list of owned nodes"""
    return np.nonzero(observation['nodes_privilegelevel'])[0]


def discovered_nodes_notowned(observation):
    """Return the list of discovered nodes that are not owned yet"""
    return np.nonzero(observation['nodes_privilegelevel'] == 0)[0]


class AbstractAction(Feature):
    """An abstraction of the gym state space that reduces
    the space dimension for learning use to just
        - local_attack(vulnid)    (source_node provided)
        - remote_attack(vulnid)   (source_node provided, target_node forgotten)
        - connect(port)           (source_node provided, target_node forgotten, credentials infered from cache)
    """

    def __init__(self, p: EnvironmentBounds):
        self.n_local_actions = p.local_attacks_count
        self.n_remote_actions = p.remote_attacks_count
        self.n_connect_actions = p.port_count
        # self.n_actions = self.n_local_actions + self.n_remote_actions + self.n_connect_actions
        self.n_actions = p.maximum_node_count * 2 + 1
        super().__init__(p, [self.n_actions])

    def specialize_to_gymaction(self, source_node: np.int32, observation, abstract_action_index: np.int32
                                ) -> Optional[cyberbattle_env.Action]:
        """Specialize an abstract "q"-action into a gym action.
        Return an adjustement weight (1.0 if the choice was deterministic, 1/n if a choice was made out of n)
        and the gym action"""

        # source_node 按env.__discovered_nodes的顺序索引
        abstract_action_index_int = int(abstract_action_index)

        node_prop = np.array(observation['discovered_nodes_properties'])

        if abstract_action_index_int < (self.n_actions - 1) / 2:  # local attack
            # vuln = abstract_action_index_int
            attack_target_index = abstract_action_index_int
            # local_attack_list = [i for (i, n) in enumerate(
            #     observation['action_mask']['local_vulnerability'][attack_target_index, :])
            #                      if n == 1]
            #local_attack_list = [local_vul for local_vul in range(self.n_local_actions) if
            #                     observation['action_mask']['local_vulnerability'][attack_target_index, local_vul] == 1]
            vuln = np.random.randint(self.n_local_actions)
            return {'local_vulnerability': np.array([attack_target_index, vuln], dtype=np.int32)}

        abstract_action_index_int -= (self.n_actions - 1) / 2
        if abstract_action_index_int < (self.n_actions - 1) / 2:  # remote attack
            vuln = np.random.randint(self.n_remote_actions)

            discovered_nodes_count = len(node_prop)
            # if discovered_nodes_count <= 1:
            if discovered_nodes_count <= 0:
                print('\n no discovered node but attacker attempt remote attacks \n')
                exit(-1)
                return None

            # NOTE: We can do better here than random pick: ultimately this
            # should be learnt from target node properties

            # pick any node from the discovered ones
            # excluding the source node itself
            # target = (source_node + 1 + np.random.choice(discovered_nodes_count - 1)) % discovered_nodes_count
            attack_target_index = int(abstract_action_index_int)
            return {'remote_vulnerability': np.array([source_node, attack_target_index, vuln], dtype=np.int32)}

        discovered_credentials = np.array(observation['credential_cache_matrix'])
        n_discovered_creds = observation['credential_cache_length']
        if n_discovered_creds <= 0:
            # no credential available in the cache: cannot poduce a valid connect action
            print('\n no available credential but attempt connect attack (prior) \n')
            exit(-1)
            return None

        # 选择合适的cred和port

        # 1. 随机选择port和cred
        # cred = np.int32(np.random.randint(n_discovered_creds))
        # port = np.random.randint(self.n_connect_actions)

        # 2. 查找匹配的cred-node-port
        potential_attack_target_indices = [i for i in range(len(observation['discovered_nodes'])) if observation['nodes_privilegelevel'][i] == 0]

        if not potential_attack_target_indices:
            cred = np.int32(np.random.randint(n_discovered_creds))
            port = np.random.randint(self.n_connect_actions)
            attack_target_index = np.random.randint(len(observation['discovered_nodes']))
        else:
            credential_indices_choices = [c for c in range(n_discovered_creds) if discovered_credentials[c, 0] in potential_attack_target_indices]

            if len(credential_indices_choices) > 0:
                cred = np.int32(np.random.choice(credential_indices_choices))
                port = np.int32(discovered_credentials[cred, 1])
                attack_target_index = np.int32(discovered_credentials[cred, 0])
            else:
                cred = np.int32(np.random.randint(n_discovered_creds))
                port = np.random.randint(self.n_connect_actions)
                attack_target_index = np.random.randint(len(observation['discovered_nodes']))
        return {'connect': np.array([source_node, attack_target_index, port, cred], dtype=np.int32)}

    def abstract_from_gymaction(self, gym_action: cyberbattle_env.Action) -> np.int32:
        """Abstract a gym action into an action to be index in the Q-matrix"""
        if 'local_vulnerability' in gym_action:
            return np.int32(gym_action['local_vulnerability'][0])
        elif 'remote_vulnerability' in gym_action:
            return np.int32((self.n_actions - 1) / 2 + gym_action['remote_vulnerability'][1])

        assert 'connect' in gym_action
        a = self.n_actions - 1
        return np.int32(a)


class ActionTrackingStateAugmentation(StateAugmentation):
    """An agent state augmentation consisting of
    the environment observation augmented with the following dynamic information:
       - success_action_count: count of action taken and succeeded at the current node
       - failed_action_count: count of action taken and failed at the current node
     """

    def __init__(self, p: EnvironmentBounds, observation: cyberbattle_env.Observation):
        self.aa = AbstractAction(p)
        # self.success_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        # self.failed_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        self.success_action_count = np.zeros(self.aa.n_actions, dtype=np.int32)
        self.failed_action_count = np.zeros(self.aa.n_actions, dtype=np.int32)
        self.env_properties = p
        super().__init__(observation)

    def on_step(self, action: cyberbattle_env.Action, reward: float, done: bool,
                observation: cyberbattle_env.Observation):
        node = cyberbattle_env.sourcenode_of_action(action)
        abstract_action = self.aa.abstract_from_gymaction(action)
        if reward > 0:
            self.success_action_count[abstract_action] += 1
        else:
            self.failed_action_count[abstract_action] += 1
        # if reward > 0:
        #     self.success_action_count[node, abstract_action] += 1
        # else:
        #     self.failed_action_count[node, abstract_action] += 1
        super().on_step(action, reward, done, observation)

    def on_reset(self, observation: cyberbattle_env.Observation):
        p = self.env_properties
        # self.success_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        # self.failed_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        self.success_action_count = np.zeros(self.aa.n_actions, dtype=np.int32)
        self.failed_action_count = np.zeros(self.aa.n_actions, dtype=np.int32)
        super().on_reset(observation)


class Feature_actions_tried_at_node(Feature):
    """A bit mask indicating which actions were already tried
    a the current node: 0 no tried, 1 tried"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * AbstractAction(p).n_actions)

    def get(self, a: ActionTrackingStateAugmentation, node: int):
        return ((a.failed_action_count[node, :] + a.success_action_count[node, :]) != 0) * 1


#
# class Feature_success_actions_at_node(Feature):
#     """number of time each action succeeded at a given node"""
#
#     max_action_count = 100
#
#     def __init__(self, p: EnvironmentBounds):
#         super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)
#
#     def get(self, a: ActionTrackingStateAugmentation, node: int):
#         return np.minimum(a.success_action_count[node, :], self.max_action_count - 1)
#
#
# class Feature_failed_actions_at_node(Feature):
#     """number of time each action failed at a given node"""
#
#     max_action_count = 100
#
#     def __init__(self, p: EnvironmentBounds):
#         super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)
#
#     def get(self, a: ActionTrackingStateAugmentation, node: int):
#         return np.minimum(a.failed_action_count[node, :], self.max_action_count - 1)


class Verbosity(enum.Enum):
    """Verbosity of the learning function"""
    Quiet = 0
    Normal = 1
    Verbose = 2


class AgentWrapper(Wrapper):
    """Gym wrapper to update the agent state on every step"""

    def __init__(self, env: cyberbattle_env.CyberBattleEnv, state: StateAugmentation):
        super().__init__(env)
        self.state = state

    def step(self, action: cyberbattle_env.Action):
        observation, reward, done, info = self.env.step(action)
        self.state.on_step(action, reward, done, observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.state.on_reset(observation)
        return observation