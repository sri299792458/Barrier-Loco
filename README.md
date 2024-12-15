# Robust Blind Locomotion Policy

This repository extends the capabilities of [Legged Gym](https://github.com/leggedrobotics/legged_gym) by implementing a robust blind locomotion policy. Our work combines the implicit model from [HIMLoco](https://github.com/author/himloco) with a multi-critic approach, where barrier functions are leveraged to enforce flexible gait adaptation. This flexibility allows locomotion on uneven terrains such as slopes and stairs, building upon the principles introduced in the paper [A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards](https://arxiv.org/abs/2409.15780).

---

## General Framework
The architecture of this framework is illustrated below:

![barrier asymmetric actor critic](https://github.com/user-attachments/assets/e85f9686-e7fa-40b3-80b0-8dac1986cebf)

### Components:
1. **Temporal and Current Observations**:
   - Inputs to the **Estimator** include temporal partial observations (o<sub>t</sub><sup>H</sup>) and current observations (o<sub>t</sub>).
   
2. **Estimator**:
   - Encodes information to generate representations such as v<sub>t</sub> and z<sub>t</sub>, which are fed into the **Actor**.
   - The estimator can leverage:
     - Variational autoencoders (VAEs) like in [DreamWaQ](https://arxiv.org/abs/2301.10602) paper and here is my [implementation](https://github.com/srinivask08/DreamWaQ) of it  .
     - Contrastive learning as in [HIMLoco](https://github.com/OpenRobotLab/HIMLoco), currently being integrated.

3. **Actor and Policy Gradient**:
   - The **Actor** outputs actions (a<sub>t</sub>) that control the robot's movements.
   - The policy gradient is informed by both:
     - A **Reward Critic** trained using privileged observations (s<sub>t</sub>) and the standard rewards like tracking velocity are passed it it.
     - A **Barrier Critic** also trained using privileged observations (s<sub>t</sub>) and the barrier rewards calculated using the relaxed log barrier functions as described in the [A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards](https://arxiv.org/abs/2409.15780) paper.

4. **Barrier Critic**:
   - This allows us to enforce gaits flexibly, unlike [Walk-These-Ways](https://github.com/Improbable-AI/walk-these-ways) which is a robust blind locomotion policy on flat terrains, where gaits are kind of rigidly "enforced" through standard rewards, limiting adaptation to uneven terrains such as stairs.

5. **Robot Control**:
   - The resulting actions enable the robot to navigate uneven terrains.

---

## Installation
This repository builds upon [Legged Gym](https://github.com/leggedrobotics/legged_gym). Follow their installation instructions to set up the required environment. Additional dependencies specific to this implementation will be detailed in the `requirements.txt` file.

---

## Current Work
We are currently integrating a planning module on top of this framework. This enhancement will enable more sophisticated navigation capabilities. The code is presently private but will be released along with our forthcoming paper.

Stay tuned!



