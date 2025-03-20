import torch.optim as optim
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import random

class DeepQNetwork(nn.Module):
  def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, lr):
    super(DeepQNetwork, self).__init__()
    self.input_dims = input_dims
    self.n_actions = n_actions
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.lr = lr
    self.gamma = 0.99
    self.exploration_proba = 1.0
    self.exploration_proba_decay = 0.005
    self.batch_size = 64

    self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
    self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
    self.loss = nn.MSELoss()
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)


  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    actions = self.fc3(x)
    return actions


  def update_exploration_probability(self):
    self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
    # print(self.exploration_proba)


class DoubleDQNAgent():
  def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
               max_mem_size = 10000, eps_end = 0.01, eps_dec = 5e-4):
    self.gamma = gamma
    self.epsilon = epsilon
    self.eps_min = eps_end
    self.eps_dec = eps_dec
    self.lr = lr
    self.actions_space = [i for i in range(n_actions)]
    self.mem_size = max_mem_size
    self.batch_size = batch_size
    self.mem_cntr = 0

    self.Q_eval = DeepQNetwork(input_dims, n_actions, fc1_dims=256, fc2_dims=256, lr=self.lr)
    self.target_Q = copy.deepcopy(self.Q_eval)

    self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
    self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
    self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
    self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
    self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    self.priorities =  np.zeros(self.mem_size, dtype=np.float32)

  ########## Buffer ##########
  def store_transition(self, state, action, reward, new_state, done):
    index= self.mem_cntr % self.mem_size
    self.state_memory[index] = state
    self.new_state_memory[index] = new_state
    self.reward_memory[index] = reward
    self.action_memory[index] = action
    self.terminal_memory[index] = done
    self.mem_cntr += 1
    self.priorities[index] = max(self.priorities) if max(self.priorities)!=0 else 1

  ########## Priority Experience Buffer ##########
  def get_priorities(self, priority_scale):
    scaled_priorities = np.array(self.priorities) ** priority_scale
    sample_probabilities = scaled_priorities / sum(scaled_priorities)
    return sample_probabilities
  
  def get_importance(self, probabilities):
    importance = 1/self.mem_cntr * 1/probabilities
    importance_normalized = importance/max(importance)
    return importance_normalized

  def set_priorities(self, indices, errors, offset=0.1):
    for i, e in zip(indices, errors):
      self.priorities[i] = abs(e) + offset

  ########## Model ##########
  def save_models(self):
    print("..... Saving Models .....")
    # T.save(self.Q_eval, "./Q_network_DQN.h5")
    # T.save(self.target_Q, "./Target_network_DQN.h5")

  def choose_action(self, observation):
    observation = observation.astype(np.float32)
    state = T.tensor(observation).to(self.Q_eval.device)
    actions = self.Q_eval.forward(state)
    # action = T.argmax(actions).item()
    return actions

  def learn(self, priority_scale=1.0):
    if self.mem_cntr < self.batch_size:
      return

    self.Q_eval.optimizer.zero_grad()
    max_mem = min(self.mem_cntr, self.mem_size)

    sample_size = min(self.mem_cntr, self.batch_size)
    sample_probs = self.get_priorities(priority_scale)

    batch = random.choices(range(max_mem), k=sample_size, weights=sample_probs[0:max_mem])
    batch_index = np.arange(self.batch_size, dtype=np.int32)
    importance = self.get_importance(sample_probs[batch_index])

    # batch = np.random.choice(max_mem, self.batch_size, replace=False, )
    # batch_index = np.arange(self.batch_size, dtype=np.int32)

    state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
    new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
    reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
    terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

    action_batch = self.action_memory[batch]
    
    q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

    q_next = self.target_Q.forward(new_state_batch)
    q_next[terminal_batch] = 0.0

    q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

    error = abs(q_eval - q_target)
    self.set_priorities(batch_index, error)

    loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

    ## second format; considering importance
    loss = T.mean(T.multiply(T.square(loss), T.tensor(importance)))

    loss.backward()
    self.Q_eval.optimizer.step()

    self.epsilon = self.epsilon-self.eps_dec if self.epsilon>self.eps_min else self.eps_min

    return loss