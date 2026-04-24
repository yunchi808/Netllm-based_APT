import os
import csv
import argparse
import torch
import torch as th
import numpy as np
import gym

import apt_cyber_sim
import apt_cyber_sim._env.cyberbattle_env as cyberbattle_env
from apt_eval.agent_wrapper import AgentWrapper, ActionTrackingStateAugmentation, EnvironmentBounds
from dataset_pipeline.generator.DQN_attacker import DQN_attacker
from dataset_pipeline.generator.SAC_attacker import SAC_attacker
from dataset_pipeline.generator.state_action_tools import attacker_rule, CyberBattleStateActionModel


parser = argparse.ArgumentParser(description="CyberBattleSim experiment setting.")
parser.add_argument("--training_episode_count", default=5000, type=int, help="number of training epochs")
parser.add_argument("--iteration_count", default=600, type=int, help="number of simulation iterations for each epoch")
parser.add_argument("--reward_goal", default=2180, type=int, help="minimum target rewards to reach for attacker")
parser.add_argument("--ownership_goal", default=0.6, type=float, help="percentage of owned nodes goal")
parser.add_argument("--pytorch_seed", default=1, type=int, help="pytorch random seed")
parser.add_argument("--env-id", default="AptCyberBattleToyCtf-v0", type=str, help="Gym env id")
parser.add_argument("--step_cost", default=1.0, type=float, help="step cost of one action")
parser.add_argument("--winning_reward", default=300, type=int, help="winning termination reward")
parser.add_argument("--maximum_node_count", default=10, type=int, help="maximum node count")
parser.add_argument("--agent_alg", default="SAC", type=str, help="attacker algorithm")
parser.add_argument("--expert_mode", action="store_true", help="freeze loaded actor and only collect transitions")
parser.add_argument("--expert_checkpoint", default="", type=str, help="checkpoint in Model/path_str, like attacker-6500.pkl")
parser.add_argument("--expert_output", default="dataset_expert.csv", type=str, help="output csv filename in expert mode")
args = parser.parse_args()


def setup_seed():
    th.manual_seed(args.pytorch_seed)
    th.cuda.manual_seed_all(args.pytorch_seed)
    th.backends.cudnn.deterministic = True


setup_seed()
apt_cyber_sim.ensure_registered()
print(f"torch cuda available={th.cuda.is_available()}")

env_str = args.env_id.replace("-", "_")
step_cost_str = f"-step_cost{args.step_cost:.1f}"
winning_reward_str = f"-winning_reward{args.winning_reward}"
iteration_str = f"-episode_iteration{args.iteration_count}"
seed_str = f"seed{args.pytorch_seed}"

alg_index_list = [i for i, k in enumerate(attacker_rule) if args.agent_alg in k]
if not alg_index_list:
    print("Invalid algorithm announcement")
    exit(-1)
alg_index = alg_index_list[0]
alg_str = f"{attacker_rule[alg_index]}"
path_str = os.path.join(env_str + step_cost_str + winning_reward_str + iteration_str, alg_str)
path_str = os.path.join(path_str, seed_str)

if not os.path.exists(os.path.join("./Outcome", path_str)):
    os.makedirs(os.path.join("./Outcome", path_str))
if not os.path.exists(os.path.join("./Model", path_str)):
    os.makedirs(os.path.join("./Model", path_str))

cyberbattlectf = gym.make(
    args.env_id,
    attacker_goal=cyberbattle_env.AttackerGoal(own_atleast_percent=args.ownership_goal),
    step_cost=args.step_cost,
    winning_reward=args.winning_reward,
    maximum_node_count=args.maximum_node_count,
    disable_env_checker=True,
)

ep = EnvironmentBounds.of_identifiers(
    maximum_total_credentials=5,
    maximum_node_count=args.maximum_node_count,
    identifiers=cyberbattlectf.identifiers,
)
stateaction_model = CyberBattleStateActionModel(ep)

attacker_list = [SAC_attacker, DQN_attacker]
attacker = attacker_list[alg_index](stateaction_model)
print(f"Attacker idx={alg_index}, algorithm={attacker_rule[alg_index]}")

dataset_name = args.expert_output if args.expert_mode else "dataset.csv"
offlinerl_dataset = open(dataset_name, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(offlinerl_dataset)
csv_writer.writerow(["state", "action", "reward", "next_state", "done", "action_mask", "next_action_mask"])

if args.expert_mode:
    if not args.expert_checkpoint:
        raise ValueError("expert_mode requires --expert_checkpoint")
    ckpt_path = os.path.join("./Model", path_str, args.expert_checkpoint)
    attacker.attacker_net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    attacker.attacker_net.eval()
    attacker.memory_full = True

wrapped_env = AgentWrapper(cyberbattlectf, ActionTrackingStateAugmentation(ep, cyberbattlectf.reset()))
for i_episode in range(1, args.training_episode_count + 1):
    print(f"Episode {i_episode}")
    observation = wrapped_env.reset()
    current_state = np.array(stateaction_model.global_features.get(wrapped_env.state, node=None), dtype=np.float32)

    total_reward = 0.0
    for t in range(1, args.iteration_count + 1):
        action_mask = attacker.compute_action_mask(observation)
        if args.expert_mode:
            with torch.no_grad():
                model_device = next(attacker.attacker_net.parameters()).device
                state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(model_device)
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(model_device)
                action_prob_tensor = attacker.attacker_net(state_tensor, mask_tensor)
                abstract_action = int(torch.argmax(action_prob_tensor, dim=1).item())
            action_prob = None
        else:
            abstract_action, action_prob = attacker.choose_abstract_action(current_state, action_mask)

        _, gym_action, actor_node = stateaction_model.implement_action(wrapped_env, abstract_action)
        if not gym_action or actor_node is None:
            exit(-2)

        observation, reward, done, info = wrapped_env.step(gym_action)
        next_state = np.array(stateaction_model.global_features.get(wrapped_env.state, node=None), dtype=np.float32)
        next_action_mask = attacker.compute_action_mask(observation)

        csv_writer.writerow(
            [
                np.array2string(current_state, separator=","),
                str(abstract_action),
                str(reward),
                np.array2string(next_state, separator=","),
                str(int(done)),
                np.array2string(np.array(action_mask), separator=","),
                np.array2string(np.array(next_action_mask), separator=","),
            ]
        )

        if not args.expert_mode:
            attacker.learn(current_state, abstract_action, reward, next_state, path_str, done, action_prob, action_mask, next_action_mask)
        current_state = next_state
        total_reward += reward
        if done:
            break

    print(f"Episode finished at step {t}, total_reward={total_reward:.1f}")

    if not args.expert_mode:
        outcome_file = open(os.path.join(os.path.join("./Outcome", path_str), "outcome.txt"), "a+", encoding="utf-8")
        outcome_file.write(f"Total Reward:{total_reward:.1f}\n")
        outcome_file.write(f"Attack steps:{t}\n")
        outcome_file.close()

        attacker.save_outcome(total_reward, t)
        if i_episode % 10 == 0:
            attacker.save_param(path_str, i_episode)

wrapped_env.close()
offlinerl_dataset.close()
print("simulation ended")
