---
subtitle:       Bootstrap
author:         jason CHEMIN
version:        0.1
---

## History of Autonomous Driving Cars (strongly recommended)

Start from classical robotics: early autonomous driving relied on hand-crafted perception + control pipelines. The core idea was to decompose the problem:

- perception (lanes, obstacles)
- localization
- planning
- control

This led to classical methods such as Pure Pursuit, Stanley controller, and MPC, which are still widely used today because they are stable, interpretable, and data-efficient.

Then came large-scale learning approaches, driven by the availability of datasets and *simulation* environments like CARLA. These enable:

- hybrid systems combining learned perception + classical planning
- large-scale evaluation without risking real-world damage
- imitation from expert demonstrations and *reinforcement learning* 

Reinforcement learning exists in robotics and driving research, but its use in real cars is limited by:

- safety constraints (exploration is risky in the real world)
- sample inefficiency (requires large amounts of interaction data)
- distribution shift between training and deployment

Because of this, most learning-based driving systems rely heavily on simulation or logged data, not direct real-world trial-and-error.

A central open problem is the *sim-to-real gap*: simulation never perfectly matches reality (dynamics, friction, sensors, edge cases), which can cause brittle real-world behavior.

---

Just a glimpse into the future: I will now show you a short Tesla [video](https://youtu.be/IRu-cPkpiFk?si=XD8UlRtPCmiEPm6k&t=1023) (starting at 17:03, no sound and no subtitles). At first glance, it looks like a simple visualization of the car's surroundings.

Plot twist: this is actually the car's learned internal representation of the world—a world model. Instead of relying solely on hand-crafted simulators, future AI systems may learn their own predictive models of the environment, allowing them to plan, imagine future trajectories, and potentially train reinforcement learning agents more safely and more efficiently. Similar ideas are explored in projects such as GAIA-1 (Wayve) and DriveDreamer.

## F1Tenth & RoboCar competition

Autonomous racing is one of the most challenging robotics benchmarks: cars must drive at the limit while making decisions in real time using only onboard sensors.

Introduce the F1Tenth and RoboCar competitions with pictures or short videos. 
Show different student strategies and highlight that there is no single "best" solution: some teams rely on classical control (Pure Pursuit, Stanley, MPC), while others use learning-based approaches such as imitation learning or reinforcement learning.

Explain that these competitions capture the same challenges as full-scale autonomous driving:

- driving as fast as possible while remaining safe,
- making robust decisions under uncertainty,
- balancing model-based methods and learned policies.

Video of F1Tenth competition at IV (international vehicule) symposium 2024 - [video](https://epitechfr-my.sharepoint.com/:v:/g/personal/jason_chemin_epitech_eu/IQBjToPMKKRlSbnWxDsH56Q4ATovNHeL7pBZ7W-HvNu4g64?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=9taf6q)

Finally, connect this to the course project: students will face the same engineering trade-offs on a simplified racing platform, where they will compete to build the fastest and most reliable autonomous driver.