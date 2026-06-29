<!---
subtitle:   README
author:     jason.chemin@epitech.eu
version:    4.0
--->

<!--
##################################################################
    INFORMATIONS BELOW MAY BE DIFFUSED TO STUDENTS/PUBLIC
##################################################################
-->

# Crash&Learn Grand Prix ![fr](https://img.shields.io/badge/Status-WIP-yellow.svg)
[![en](https://img.shields.io/badge/lang-en-red.svg)](README_EN.md)
<img src="logo.png" alt="logo" align="right" height="170"/>

Crash&Learn Grand Prix is a reinforcement learning project where students train autonomous racing agents using a custom F1Tenth simulation environment. 
Students implement or use the RL algorithm of their choice, experiment with the simulation, and compete to have the best racing car.
In a final Grand Crash, the best agents face off in a live multi-agent showdown — where only the strongest policies survive.

  
|            |                        |
| ---------: | :--------------------- |
|   workload | ??? Weeks                |
|   Group    | 4 prefered                 |
[🚧 WIP : Change workload]

<br>

<p align="center">
  <a href="#-learning-objectives">Goal</a> •
  <a href="#activities">Activities</a> •
  <a href="#-see-also">See Also</a>
</p>

## 🎯 Learning Objectives

This project introduces students to **deep reinforcement learning through competitive autonomous racing**. Students will:

- Use or implement an RL algorithm to train your agent: single agent on the circuit, then adversarial fighting other groups...
- Build a complete RL training pipeline: environment wrapping, observation preprocessing, reward design, and policy evaluation.
- Develop practical experimentation skills, and use W&B or TensorBoard for tracking training curves.
- Learn to deploy trained policies as ONNX models.
- Compete in a live tournament against peers !

The "Grand Crash" pedagogical moment — launching all trained agents together mid-project — creates an understanding of why single-agent policies break down when thrown into chaos they never saw during training.

The project is designed to be **fun and engaging**: students race cars on real-world circuits (Bahrain, Monaco, Austin, Spa...), experiment with demolition-derby dynamics, and see their agents improve over training. 


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

- **"Collisions should be avoided at all costs."** — This is a *crash-and-learn* project. Vehicle-vehicle contact is **not penalized** by the simulation. 
- **"The WALLS can eat you entirely"** — Be careful O_O ! This is not a bug, it's a feature.
- **"The observation dict from env_simulation is my network input."** — Do really need everything ? Do you need more ?

### Intentional Ambiguities

- **Adversarial training structure** is open-ended. How will you fight your opponents ? Anything goes.
- What is that **Friction** parameter ? It is now sunny, but it may get worse... winter is coming.

### Technical Difficulties

- **ONNX export compatibility**: Students must export their trained model with an opset **≤ the pinned maximum** in the tournament's `onnxruntime`. Newer opsets will be rejected at inference. Provide the max opset value to students before submission deadline.
- **No GPU required**: The simulation runs entirely on CPU (2D physics + numba LiDAR). Students who have GPUs may use them for training speedup, but CPU-only submissions are fully supported.
- **Seed reproducibility**: The simulation supports seeding via `reset(seed=...)`, but students should be warned that some stochasticity (friction changes, LiDAR noise) is intentional and non-deterministic across runs — this is part of domain randomization training.
- **Observation, Action and Reward designs** are fun to visualize (and debug...) !

## 📝 Resources

<div align="center">
  <a href="https://github.com/f1tenth/f1tenth_gym"><img src="https://img.shields.io/badge/backend-f1tenth_gym-green"/></a>
  <a href="#-bootstrap"><img src="https://img.shields.io/badge/bootstrap-yes-green"/></a>
  <a href="#defense"><img src="https://img.shields.io/badge/defense-live_tournament-orange"/></a>
</div>

### Simulation & Environment
The simulation is a fork from F1Tenth simulation, with added collisions and other changes.
It is a competition of autonomous racing cars, similar to RoboCar but WorldWide and where races are 1v1.
- [F1Tenth Official Repository](https://github.com/f1tenth/f1tenth_gym) — upstream simulation engine
- [F1Tenth Championship](https://f1tenth.org/) — real-world student competition context
- [Video of the 2024 finale](https://www.youtube.com/watch?v=CQ9mFMQhltQ)

### RL Fundamentals
Many books and videos are available online about RL.
I'll just link one video that illustrate the RL concepts: [Learning walking on a Humanoid Robot](https://www.youtube.com/watch?v=QwJcF08hfs8)

More references:
- [Stable-Baselines3 Tutorials](https://stable-baselines3.readthedocs.io/)
- [CleanRL Tutorials](https://docs.cleanrl.dev/rl-algorithms/overview/)
- [W&BRL Callbacks](https://docs.wandb.ai/guides/integrations/other/libraries/sb3)

### Circuit Maps
- 20+ real-world circuits in `maps/` directory: Bahrain, Monaco, Austin, Catalunya, Spa, Monza, Silverstone...
- Centerline CSV files provide dense progress computation; occupancy grids provide LiDAR raycasting.
- Maps can be switched mid-training via `set_map(name)`

## Activities

### 🏁 Kickoff

The goal is to motivate students and provide them with the necessary context to approach the project with engagement. The kickoff should be **fun, visual, and competitive**.

**Suggested formats:**

- **History of Autonomous driving cars** (STRONGLY SUGGESTED) — Originally we use physics simulator like ours for this project that is simplified to more complex one such as [Carla](https://carla.org/). From that you can test your own classical algorihtm for autonomous driving (Pure Pursuit, Stanley, MPCs...) to more recently fully end-to-end learned driving. Why ? Because we can't do Reinforcement Learning in real-world -> Crash&Learn is the basis mechanism of RL and we can't do it in real life... without killing someone, that is why we need to do millions of tries and learning in simulation before.
However there is one major issue with that, the sim-to-real gap, where the simulation does not match perfectly with the reality and our method fails in the real-world.
But more recently on new cars such as Tesla there was that [video](https://youtu.be/IRu-cPkpiFk?si=XD8UlRtPCmiEPm6k&t=1023)starting exactly at 17:03 and without the sound and subtitles.... This is a all a simulation of the world, what we call *World Models*. That's the current and future big topic for AI & Robotics, to learn with RL in safety and without the sim-to-real issue.

- **F1Tenth & Robocar competition** — Presentation of the competition, strategies of students, equipments... And the place of RL in all that.

**Key message**: "You're not training a car to avoid obstacles. You're training a driver who learns to race — including how to crash, recover, and use contact as a tool."

### 🧪 Bootstrap

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
- Goal: Record an episode with `sim_recorder.py` then replay it with `viz_replay.py` to observe the F1Tenth environment in action
- Why: Gives students an intuitive understanding of LiDAR, track boundaries, car dynamics, and the 2×2 start grid before they write any training code
- Commands:
  - Record: `python sim_recorder.py --steps 500 --num-cars 1`
  - Replay: `python viz_replay.py --npz episode.npz`

**Exercise 2: Write a minimal gymnasium.Env wrapper**
- Goal: Wrap `env_simulation` seven functions inside a custom `gymnasium.Env` (~30 lines)
- Why: Teaches the environment abstraction layer; students see what's behind `gym.make()` and can debug their own wrapper if training fails
- Includes: defining observation/action spaces, reset/step with terminated/truncated, reward computation stub

#### Additional exercise (all groups): Self-evaluation environment

**Exercise 3: Build an evaluation harness**
- Goal: Create a script that runs the trained policy solo and records/replays it using `sim_recorder.py` + `viz_replay.py`
- Why: Provides auto-evaluation capability outside of automated testing; students can verify their agent drives competently on any circuit
- Connects to: `verify_submission.py` which validates ONNX export and tournament compatibility

### 🔄 Follow-Ups

- **Weekly check-ins**: Review training curves (W&B/TensorBoard), observation the reward is actually improving, diagnose plateaus vs divergence
- **Mid-project "Grand Crash" moment** (G-level): Launch all agents from different teams together in simulation. Observe the chaos. Debrief on non-stationarity — why a policy trained against itself fails against strangers, and what domain randomization (friction changes, LiDAR noise) taught them to expect
- **Pre-defense review**: Verify each team can explain their reward design choices, preprocessing decisions, and training curve interpretation. Check ONNX export works with tournament-pinned onnxruntime

### 🎯🏆 Defense

The defense for this project is a **presentation** of their implementation for each group and the **GRAND PRIX tournament**. Estimated duration per group is **[🚧 WIP: 10–15 minutes]**.

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