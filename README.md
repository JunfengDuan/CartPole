# Cartpole(小车立杆)

## pytorch官网的DQN示例
>https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
>https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py

As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state, 
and also returns a reward that indicates the consequences of the action. In this task, rewards are +1 for every incremental 
timestep and the environment terminates if the pole falls over too far or the cart moves more then 2.4 units away from center. 
This means better performing scenarios will run for longer duration, accumulating larger return.

The CartPole task is designed so that the inputs to the agent are 4 real values representing the environment state (position, velocity, etc.). 
However, neural networks can solve the task purely by looking at the scene, so we’ll use a patch of the screen centered on the cart as an input. 
Because of this, our results aren’t directly comparable to the ones from the official leaderboard - our task is much harder. Unfortunately this 
does slow down the training, because we have to render all the frames.

Strictly speaking, we will present the state as the difference between the current screen patch and the previous one. 
This will allow the agent to take the velocity of the pole into account from one image.

# DQN algorithm
Our environment is deterministic, so all equations presented here are also formulated deterministically for the sake of simplicity. In the reinforcement learning literature, they would also contain expectations over stochastic transitions in the environment.


