# Reinforcement Learning for LLMs

This repo serves as a workspace for my learning of RL in the context of modern LLMs.

Resources:
- https://spinningup.openai.com/

## Basic concepts

Term | Explanation
---|---
Agent | Us
Environment | The world
Action | Environment changes upon an agent's action, but it can also change on its own. Current action is acted **after**/**on** current state.
State | Complete view of the world.
Observation | What agent can see. Partial or full state of the world.
Policy | Agent's rule to decide which action to take. It can be probabilistic / stochastic. Usually it's parameterized so that we can optimize it.
Trajectory (aka episodes, rollouts) | Sequence of states and actions
State transition | What happens to the world in the next time step. Depends on the laws of the environment, and only the current action. It can be stochastic.
Reward | Response from the environment to the agent. Only depends on **current state, current action, and next state**. It can be simplified to only depend on current state, or current state and current action.
Return | Cumulative reward over a trajectory. The agent's goal is to maximize this. For infinite-horizon, we can include a discount factor.
Expected return | Since policy and transition can be stochastic, trajectory is stochastic as well. Hence, expected return is computed over the distribution of trajectories. The **optimal policy** maximizes expected return.
Value function | Expected return at a particular state, following a particular policy. **Action-value function** (or Q-function) additionally conditioned on a particular (current) action. It follows that Value function is expected value of Action-value function over policy (action distribution). **Optimal action** corresponds to optimal Q-function.
Advantage function | The difference between Q-function and Value function i.e. Q - V -> how much a particular action is better than average, given a policy.

## Model-free vs Model-based

Model-based: Agent learns to model the environment. This allows the agent to plan ahead, but the learned model might not generalise well.

### Model-free RL

Learning | Description
---|---
Policy optimization | Models the policy. Optimize via stochastic gradient ascent on (approximate) Expected return. Typically **on-policy** - use data according to latest policy. Usually more stable. e.g. A2C/A3C, PPO.
Q-Learning | Models the Q-function. Optimize based on Bellman equation. Typically **off-policy** - use any data, not necessarily with the latest policy. The corresponding policy is choosing the action with the largest Q-function value. e.g. DQN.

## Policy gradient

**Policy gradient** is the gradient of Expected return (our objective) with respect to Policy's parameters. We can derive a nice form for it.

