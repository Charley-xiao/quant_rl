import torch
import torch.optim as optim
import numpy as np
from nets import PolicyNetwork, ValueNetwork

class RLAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        
        # Initialize policy and value networks
        self.policy_net = PolicyNetwork(state_dim, action_dim, max_action)
        self.value_net1 = ValueNetwork(state_dim, action_dim)
        self.value_net2 = ValueNetwork(state_dim, action_dim)  # TD3 uses two value networks
        
        # Target networks
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, max_action)
        self.target_value_net1 = ValueNetwork(state_dim, action_dim)
        self.target_value_net2 = ValueNetwork(state_dim, action_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer1 = optim.Adam(self.value_net1.parameters(), lr=3e-4)
        self.value_optimizer2 = optim.Adam(self.value_net2.parameters(), lr=3e-4)

        # Copy parameters to target networks
        self.update_target_networks(1.0)

    def choose_action(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.policy_net(state).detach().numpy()[0]
        # Add some noise for exploration
        action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def update_policy(self, replay_buffer, batch_size):
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Convert data to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # Compute the target value using target policy and value networks
        with torch.no_grad():
            next_actions = self.target_policy_net(next_states)
            target_value1 = self.target_value_net1(next_states, next_actions)
            target_value2 = self.target_value_net2(next_states, next_actions)
            target_value = torch.min(target_value1, target_value2)
            target_value = rewards + self.gamma * (1 - dones) * target_value
        
        # Update value networks
        current_value1 = self.value_net1(states, actions)
        current_value2 = self.value_net2(states, actions)
        value_loss1 = torch.nn.MSELoss()(current_value1, target_value)
        value_loss2 = torch.nn.MSELoss()(current_value2, target_value)
        
        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        self.value_optimizer1.step()

        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        self.value_optimizer2.step()

        # Delayed policy update
        if replay_buffer.steps % 2 == 0:
            # Update policy network
            policy_loss = -self.value_net1(states, self.policy_net(states)).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update target networks with soft update
            self.update_target_networks(self.tau)

    def update_target_networks(self, tau):
        # Soft update for the target policy and value networks
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_value_net1.parameters(), self.value_net1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_value_net2.parameters(), self.value_net2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
