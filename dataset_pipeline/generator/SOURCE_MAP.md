# Source Mapping

This folder was assembled from:

- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/scripts/cyberattackerexp/run.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/scripts/cyberattackerexp/SAC_attacker.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/scripts/cyberattackerexp/DQN_attacker.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/scripts/cyberattackerexp/RAND_attacker.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/scripts/cyberattackerexp/Tools.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/scripts/cyberattackerexp/truncate_dataset_after_episodes.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/cyberbattle/samples/toyctf/toy_ctf.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/cyberbattle/samples/new_ctf/node10_v1.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/cyberbattle/samples/new_ctf/node10_v2.py`
- `E:/NetLLM/Dataset/Learn-offline_rl_dataset_generator/cyberbattle/samples/new_ctf/node10_v3.py`

## Adaptations made for APT

- `cyberbattle` imports were migrated to `apt_cyber_sim`.
- `agent_wrapper`/state-action utility imports were migrated to `apt_eval`.
- `mix_exp_pools.py` is newly added for pkl ratio mixing in APT workflows.
