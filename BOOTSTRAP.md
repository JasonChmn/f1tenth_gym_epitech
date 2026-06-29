---
subtitle:       Bootstrap
author:         Claudia MIGLIACCIO, Pierre ROBERT, Gaspard VARENNES
version:        0.3
---

### Bootstrap: Understanding Reinforcement Learning

The objective of this bootstrap is **not** to write a complete RL algorithm from scratch. Instead, the goal is to understand the different building blocks of reinforcement learning before tackling autonomous racing.


#### Exercise 1 – Observe a random policy

Read CartPole doc from gymnasium.

Run CartPole with random actions (See USE_RANDOM in eval.py).

Print all actions, observation, reward, terminate state to understand what happens.

Questions:

* When does an episode terminate?
* What is an observation? -> clamp or perturb one feature and observe stability.
* What is an action? -> Try a random policy instead... try to add noise to the policy ?
* What is the reward? -> modify reward function and observe learning behavior. E.g. Add -1 to the reward (0 if success, -1 if fail). Does it change something ?


---

#### Exercise 2 – Run a working DQN

A complete DQN implementation is provided.

Students should:

* train the agent,
* visualize the learning curve (tensorboard),
* see the trained agent.

Questions:

* Why does performance improve ?
* Why is exploration necessary ? How is exploration done here ?

---

#### Exercise 3 – Read the algorithm

Open the DQN implementation and identify:

* where observations enter the network,
* where the system “decides” actions,
* where rewards are stored,
* where the neural network is updated,
* where the target network is used,
* where exploration (ε-greedy) happens.

The objective is not to understand every PyTorch instruction but to understand the RL loop.

Here we use DQN, it's quite old.
Make them see what other RL algorithm there are out there, the most used etc...

---

#### Exercise 4 – Small modifications

Play with hyperparameters in train_cartpole and observe its effect. Params like :

* learning rate,
* discount factor,
* replay buffer size,
* exploration schedule,
* network size.
* batch size...

Discuss why training succeeds or fails. what changes first? What breaks training?

Discuss how they could improve the training speed ? (multiprocessing, subproc env...)