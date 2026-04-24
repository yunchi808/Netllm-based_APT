import torch
import numpy as np
from apt_eval.agent_wrapper import EnvironmentBounds
from apt_eval import agent_wrapper as w
import apt_cyber_sim._env.cyberbattle_env as cyberbattle_env
from typing import Optional, Tuple

attacker_rule = [
    "SAC_attacker"
]


class AbstractAttackerModel:
    def __init__(self, stateaction_model=None):
        if stateaction_model:
            self.state_dim = len(stateaction_model.state_space.dim_sizes)
            self.action_dim = stateaction_model.action_space.flat_size()

    def learn(self, *params):
        pass

    def save_param(self, *params):
        pass

    def save_outcome(self, *params):
        pass

    def choose_abstract_action(self, *params):
        pass

    def compute_action_mask(self, observation):
        mask_list = [0 for _ in range(self.action_dim)]
        for candidate_target_node in range(len(observation["discovered_nodes"])):
            if observation["nodes_privilegelevel"][candidate_target_node] > 0:
                mask_list[candidate_target_node] = 1
            mask_list[candidate_target_node + int((self.action_dim - 1) / 2)] = 1
        if observation["credential_cache_length"] > 0:
            mask_list[int(self.action_dim) - 1] = 1
        return mask_list


class CyberBattleStateActionModel:
    """Define an abstraction of state and action spaces for CyberBattle."""

    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        self.global_features = w.ConcatFeatures(
            ep,
            [
                w.Feature_discovered_node_count(ep),
                w.Feature_owned_node_count(ep),
                w.Feature_all_node_conquer_state(ep),
                w.Feature_all_credential_detail(ep),
                w.Feature_all_port_detail(ep),
                w.Feature_discovered_ports(ep),
                w.Feature_discovered_ports_counts(ep),
                w.Feature_discovered_credential_count(ep),
            ],
        )

        self.node_specific_features = w.ConcatFeatures(
            ep,
            [
                w.Feature_actions_tried_at_node(ep),
            ],
        )

        self.state_space = w.ConcatFeatures(ep, self.global_features.feature_selection)
        self.action_space = w.AbstractAction(ep)

    def get_state_astensor(self, state: w.StateAugmentation):
        state_vector = self.state_space.get(state, node=None)
        state_vector_float = np.array(state_vector, dtype=np.float32)
        state_tensor = torch.from_numpy(state_vector_float).unsqueeze(0)
        return state_tensor

    def implement_action(
        self, wrapped_env: w.AgentWrapper, abstract_action: np.int32
    ) -> Tuple[str, Optional[cyberbattle_env.Action], Optional[int]]:
        observation = wrapped_env.state.observation
        potential_source_nodes = [from_node for from_node in w.owned_nodes(observation)]

        if len(potential_source_nodes) > 0:
            source_node = np.random.choice(potential_source_nodes)
        else:
            print("\n len of potential source nodes is 0.\n")
            exit(-1)

        gym_action = self.action_space.specialize_to_gymaction(
            source_node, observation, np.int32(abstract_action)
        )

        if not gym_action:
            print(
                "\n action_space.specialize_to_gymaction() unable to output a gym action \n"
            )
            exit(-1)
            return "exploit[undefined]->explore", None, None
        else:
            return "exploit", gym_action, source_node
