import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_algorithms.DDQN import DDQNModel, DDQN

X = torch.tensor([
    [[0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],],
    [[0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 2 ,0, 0, 0],
     [0, 0, 0, 1 ,0, 0, 0],
     [0, 0, 0, 1 ,0, 0, 0],
     [0, 0, 0, 2 ,1, 2, 0],
     [0, 1, 2, 1 ,2, 2, 1],],
    [[0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 1 ,1, 1, 0],
     [0, 0, 0, 1 ,2, 2, 0],
     [0, 0, 0, 2 ,2, 1, 0],
     [1, 1, 1, 2 ,2, 2, 0],],
    [[0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],]
])

X1 = torch.tensor([
    [[0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [1, 0, 0, 0 ,0, 0, 0],],
    [[0, 0, 0, 1 ,0, 0, 0],
     [0, 0, 0, 2 ,0, 0, 0],
     [0, 0, 0, 1 ,0, 0, 0],
     [0, 0, 0, 1 ,0, 0, 0],
     [0, 0, 0, 2 ,1, 2, 0],
     [0, 1, 2, 1 ,2, 2, 1],],
    [[0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 2 ,0, 0, 0],
     [0, 0, 0, 1 ,1, 1, 0],
     [0, 0, 0, 1 ,2, 2, 0],
     [0, 0, 0, 2 ,2, 1, 0],
     [1, 1, 1, 2 ,2, 2, 0],],
    [[0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 1 ,0, 0, 0],]
])

model = DDQNModel()
ddqn = DDQN("aaa")
state = torch.tensor(
    [[0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0],
     [0, 0, 0, 0 ,0, 0, 0]])
ddqn.model.train()

a = torch.tensor([0, 0, 1, 0])
print(a.any())

# Compute Q-values and targets
# states, actions, rewards, next_states, runs = experiences
# states = X
# next_states = X1
# ddqn.batch_size = 4
# rewards = torch.tensor([0.0, 10.0, -10.0, 5.0])
# runs = torch.tensor([1.0, 0.0, 1.0, 1.0])
# actions = torch.tensor([0, 3, 3, 3])
# weights = torch.tensor([0.0, 1.0 , 0.5, 0.3])
# next_actions = torch.argmax(ddqn.model(next_states), axis=1)
# next_Q_values = ddqn.target(next_states)  # using target network
# max_next_Q_values = next_Q_values[torch.arange(ddqn.batch_size), next_actions]
# target_Q_values = rewards + runs * ddqn.gamma * max_next_Q_values
# target_Q_values = target_Q_values.reshape(-1,1)
# mask = F.one_hot(actions, 7)

# all_Q_values = ddqn.model(states)
# Q_values = torch.sum(all_Q_values * mask, axis=1, keepdims=True)
# td_errors = target_Q_values - Q_values
# if ddqn.use_prioritized:
#     print((weights * ddqn.loss_fn(target_Q_values, Q_values)).shape)
#     weighted_loss = torch.sum(weights * ddqn.loss_fn(target_Q_values, Q_values))
#     weighted_loss.backward()
#     # loss = torch.mean(weighted_loss)
# else:
#     loss = ddqn.loss_fn(target_Q_values, Q_values)
#     loss.backward()
# ddqn.optimizer.step()
# ddqn.optimizer.zero_grad()