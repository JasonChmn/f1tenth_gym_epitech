<!--
subtitle:   README_EN
author:     Crash&Learn Pedagogical Team
version:    1.0
-->

<!--
##################################################################
    INFORMATIONS BELOW MAY BE DIFFUSED TO STUDENTS/PUBLIC
##################################################################
-->

# Crash&Learn Grand Prix ![fr](https://img.shields.io/badge/Status-WIP-yellow.svg)
[![en](https://img.shields.io/badge/lang-en-red.svg)](README_EN.md)
<img src="logo.png" alt="logo" align="right" height="170"/>

[🚧 WIP: Crash&Learn Grand Prix is a reinforcement learning project where students train autonomous racing agents from scratch using a custom F1Tenth simulation environment. Students implement their own RL algorithms in PyTorch, experiment with contact-based racing dynamics, and compete in a live multi-agent tournament — learning about non-stationarity, domain randomization, and real-world RL challenges along the way.]
[🚧 WIP : When the project is done, change the status badge to OK / green]

  
|            |                        |
| ---------: | :--------------------- |
|   workload | ??? Weeks                |
|   Group    | 3 to 4                 |
[🚧 WIP : Change workload]

<br>

<p align="center">
  <a href="#-learning-objectives">Goal</a> •
  <a href="#activities">Activities</a> •
  <a href="#-see-also">See Also</a>
</p>

## 🎯 Learning Objectives

This project introduces students to **deep reinforcement learning through competitive autonomous racing**. Students will:

- Implement a deep RL algorithm (DQN or PPO) from scratch, or use an existing libraby such as Stable-Baselines or cleanRL...
- Build a complete RL training pipeline: environment wrapping, observation preprocessing, reward design, and policy evaluation
- Understand the challenges of **non-stationary environments** through contact-based multi-agent racing
- Develop practical experimentation skills using W&B or TensorBoard for tracking hyperparameter sweeps and training curves
- Learn to deploy trained policies as ONNX models and compete in a live tournament against peers

The project is designed to be **fun and engaging**: students race cars on real-world circuits (Bahrain, Monaco, Austin, Spa...), experiment with demolition-derby dynamics, and see their agents improve over training. 
The "Grand Crash" pedagogical moment — launching all trained agents together mid-project — creates a visceral understanding of multi-agent non-stationarity that no lecture could match.

**Prerequisites**: Basic knowledge of Neural Network design. A tiny bit of maths. Good visualization of the training process.

<br>

<!--
##################################################################
    ⚠️ INFORMATIONS BELOW ARE FOR PEDAGOGICAL STAFF ONLY ⚠️ 
##################################################################
-->


# 🧑‍🏫 Pedagogical Guidelines

## ⚠️ Watch Out

### Common Misconceptions

- **"I should use Stable-Baselines3 or other RL libraries."** — The project requires implementing the RL algorithm from scratch in PyTorch. Using SB3, RLlib, or similar libraries defeats the pedagogical purpose of understanding algorithms at the tensor level.
- **"Collisions should be avoided at all costs."** — This is a *crash-and-learn* project. Vehicle-vehicle contact is **not penalized** by the simulation. Students must decide whether to avoid or exploit contact in their reward design and policy — it's part of the challenge.
- **"The observation dict from env_simulation is my network input."** — `get_obs()` returns a raw dict (LiDAR scan, scalars, opponent positions). Each team selects/reduces features inside their own `agent.py` preprocessing, producing their own input tensor. Two agents with completely different network shapes can interoperate in the same tournament.
- **"I need to understand gymnasium.Env to use this sim."** — There is no Gymnasium dependency in `env_simulation`. Students build their own `gymnasium.Env` wrapper (~30 lines) on top of seven module-level functions. This is intentional: it teaches the abstraction layer between environments and algorithms.

### Intentional Ambiguities

- **Reward function design** is deliberately not prescribed. How students shape rewards (progress-based, collision-aware, speed-penalized...) directly impacts training success and is a key component of defense discussion.
- **Adversarial training structure** is open-ended: self-play, frozen opponent snapshots, curriculum of increasing difficulty, multiple policies fused or ensembled — teams must design and justify their approach.
- **Friction dynamics** are hidden from the observation space but accessible via `get_step_info()["friction_current"]`. This is a bonus for students who discover and exploit it, but not required for a working solution.

### Technical Difficulties

- **ONNX export compatibility**: Students must export their trained model with an opset **≤ the pinned maximum** in the tournament's `onnxruntime`. Newer opsets will be rejected at inference. Provide the max opset value to students before submission deadline.
- **No GPU required**: The simulation runs entirely on CPU (2D physics + numba LiDAR). Students who have GPUs may use them for training speedup, but CPU-only submissions are fully supported.
- **Seed reproducibility**: The simulation supports seeding via `reset(seed=...)`, but students should be warned that some stochasticity (friction changes, LiDAR noise) is intentional and non-deterministic across runs — this is part of domain randomization training.

## 📝 Resources

<div align="center">
  <a href="https://github.com/f1tenth/f1tenth_gym"><img src="https://img.shields.io/badge/backend-f1tenth_gym-green"/></a>
  <a href="#-bootstrap"><img src="https://img.shields.io/badge/bootstrap-yes-green"/></a>
  <a href="#defense"><img src="https://img.shields.io/badge/defense-live_tournament-orange"/></a>
</div>

### Simulation & Environment
- [F1Tenth Official Repository](https://github.com/f1tenth/f1tenth_gym) — upstream simulation engine
- [PettingZoo](https://pettingzoo.faramx.org/) — multi-agent API reference (for G-level multi-agent extension)
- [F1Tenth Championship](https://f1tenth.org/) — real-world student competition context

### RL Fundamentals
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/ddpg_tutorial.html) — DQN/PPO implementation reference
- [Stable-Baselines3 Tutorials](https://stable-baselines3.readthedocs.io/) — for algorithm understanding (implementation must be from scratch)
- [W&BRL Callbacks](https://docs.wandb.ai/guides/integrations/other/libraries/sb3) — experiment tracking setup

### Circuit Maps
- 20+ real-world circuits in `maps/` directory: Bahrain, Monaco, Austin, Catalunya, Spa, Monza, Silverstone...
- Centerline CSV files provide dense progress computation; occupancy grids provide LiDAR raycasting
- Maps can be switched mid-training via `set_map(name)`

## Activities

### Kickoff

The goal is to motivate students and provide them with the necessary context to approach the project with engagement. The kickoff should be **fun, visual, and competitive**.

**Suggested formats:**

- **Show a demo of autonomous racing in action** — play a short video or live demo of F1Tenth cars racing, or drone racing footage. Highlight how RL enables these systems to handle real-world uncertainty (sensor noise, friction changes, collisions).
- **Demo a crashed car recovering** — show the demolition-derby dynamics: cars colliding, bouncing, and continuing. Emphasize that in Crash&Learn, crashes are *part of the strategy*, not a failure state.
- **Debate on applications**: 
  - Autonomous driving in dense traffic (multi-agent interaction, collision handling)
  - Drone racing (fast decision-making under sensor noise)
  - Warehouse robotics (collision-aware navigation)
  - Any of these connects naturally to what students will build

**Key message**: "You're not training a car to avoid obstacles. You're training a driver who learns to race — including how to crash, recover, and use contact as a tool."

### Bootstrap

This bootstrap contains **two parallel paths**. Choose based on the group's background and time available:

#### Path A: Gymnasium mini-exercises (recommended for groups without RL experience)

**Exercise 1: CartPole with DQN from scratch**
- Goal: Implement a minimal DQN on Gymnasium's CartPole-v0 using PyTorch
- Why: Establishes the core RL loop (env.reset → step → reward → update → train) without any racing complexity
- Prerequisites: Basic PyTorch knowledge, numpy

**Exercise 2: Observation preprocessing pipeline**
- Goal: Build a feature extractor that converts raw observations into normalized input tensors
- Why: Prepares students for the LiDAR + opponent data preprocessing they'll need in the main project
- Focus: vectorization, normalization, handling missing/None values

#### Path B: Direct simulation introduction (recommended for groups with prior RL experience)

**Exercise 1: Run the simulation and visualize a random agent**
- Goal: Use `viz_topdown.py` to observe the F1Tenth environment in action
- Why: Gives students an intuitive understanding of LiDAR, track boundaries, car dynamics, and the 2×2 start grid before they write any training code
- Command: `python viz_topdown.py --label "random agent demo"`

**Exercise 2: Write a minimal gymnasium.Env wrapper**
- Goal: Wrap `env_simulation` seven functions inside a custom `gymnasium.Env` (~30 lines)
- Why: Teaches the environment abstraction layer; students see what's behind `gym.make()` and can debug their own wrapper if training fails
- Includes: defining observation/action spaces, reset/step with terminated/truncated, reward computation stub

#### Additional exercise (all groups): Self-evaluation environment

**Exercise 3: Build an evaluation harness**
- Goal: Create a script that runs the trained policy solo and generates an MP4 replay via `record_episode.py`
- Why: Provides auto-evaluation capability outside of automated testing; students can verify their agent drives competently on any circuit
- Connects to: `verify_submission.py` which validates ONNX export and tournament compatibility

### Follow-Ups

- **Weekly check-ins**: Review training curves (W&B/TensorBoard), observation the reward is actually improving, diagnose plateaus vs divergence
- **Mid-project "Grand Crash" moment** (G-level): Launch all agents from different teams together in simulation. Observe the chaos. Debrief on non-stationarity — why a policy trained against itself fails against strangers, and what domain randomization (friction changes, LiDAR noise) taught them to expect
- **Pre-defense review**: Verify each team can explain their reward design choices, preprocessing decisions, and training curve interpretation. Check ONNX export works with tournament-pinned onnxruntime

### Defense

The defense for this project is a **[🚧 WIP: live tournament demonstration | recorded demo video]**. Estimated duration per group is **[🚧 WIP: 10–15 minutes]**.

**Format suggestions:**
- Each team demonstrates their trained agent running in the simulation
- Teams present: reward design rationale, preprocessing approach, training curve analysis, and what they'd change with more time
- G-level groups additionally demonstrate multi-agent behavior (Grand Crash aftermath or adversarial policy performance)
- Open discussion on trade-offs: speed vs. safety, exploration strategy, generalization across circuits

**What pedagogical teams should look for:**
- Understanding of why collisions aren't penalized by default and how the team's reward function handles contact
- Ability to interpret training curves and explain hyperparameter choices
- ONNX model is correctly exported and loads in onnxruntime (not just torch)
- Clear separation between `env.py` (training environment), `agent.py` (inference policy), and `train.py` (training loop)

## 🔍 See Also
[X-TBD-TBD : Similar RL racing project with Stable-Baselines](https://github.com/Epitech/X-TBD-TBD)
[X-TBD-TBD : Multi-agent PettingZoo racing environment](https://github.com/Epilech/X-TBD-TBD)

### 📝 Help and Resources
- [F1Tenth Gym Documentation](https://f1tenth-gym.readthedocs.io/en/latest/) — upstream API reference
- [PyBullet collision docs](https://docs.pybullet.org/manuals/physicssimulation.html) — understanding contact physics
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/index.html) — inference deployment reference

### 🫶 Contributing

If you have a question, a suggestion, or feedback, the Discussion tab of this repo is the ideal place to share them. If you want to propose a change directly, you can submit a PR. We prefer these channels over direct contact in order to centralize exchanges and make them beneficial for the whole team. 😊

We would particularly appreciate suggestions or work on these points:
- Additional circuit maps with centerline CSVs
- Baseline training configurations (hyperparameters, curves) for comparison
- Anti-chirp measures for adversarial training phases


</write_to_file>