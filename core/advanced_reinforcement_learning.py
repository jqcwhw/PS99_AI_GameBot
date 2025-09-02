"""
Advanced Reinforcement Learning System
Integrates multiple RL algorithms from SerpentAI and Reinforcement Learning Book
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np
from collections import deque
import json
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ExperienceBuffer:
    """
    Experience replay buffer for reinforcement learning
    Based on implementations from multiple RL algorithms
    """
    
    def __init__(self, buffer_size: int = 100000, prioritized: bool = False):
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        
        # Standard buffers
        self.observations = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_observations = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        
        # For prioritized replay
        if prioritized:
            self.priorities = deque(maxlen=buffer_size)
            self.max_priority = 1.0
        
        self.logger = logging.getLogger(__name__)
        
    def add(self, observation: np.ndarray, action: Union[int, np.ndarray], reward: float, 
            next_observation: np.ndarray, done: bool, priority: float = None):
        """Add experience to buffer"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.dones.append(done)
        
        if self.prioritized:
            if priority is None:
                priority = self.max_priority
            self.priorities.append(priority)
            self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch from experience buffer"""
        if len(self.observations) < batch_size:
            return None
        
        if self.prioritized:
            # Prioritized sampling
            priorities = np.array(self.priorities, dtype=np.float32)
            probabilities = priorities / np.sum(priorities)
            indices = np.random.choice(len(self.observations), batch_size, p=probabilities)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.observations), batch_size, replace=False)
        
        batch = {
            'observations': np.array([self.observations[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'next_observations': np.array([self.next_observations[i] for i in indices]),
            'dones': np.array([self.dones[i] for i in indices]),
            'indices': indices
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for prioritized replay"""
        if not self.prioritized:
            return
        
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.observations)

class DQNNetwork(nn.Module):
    """
    Deep Q-Network implementation
    Based on SerpentAI and RL book implementations
    """
    
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 512):
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Convolutional layers for image input
        if len(input_shape) == 3:  # Image input (C, H, W)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            # Calculate conv output size
            conv_out_size = self._get_conv_out_size(input_shape)
            
            self.fc_layers = nn.Sequential(
                nn.Linear(conv_out_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions)
            )
        else:
            # Fully connected layers for vector input
            input_size = np.prod(input_shape)
            self.fc_layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions)
            )
            self.conv_layers = None
    
    def _get_conv_out_size(self, shape):
        """Calculate convolutional output size"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            conv_out = self.conv_layers(dummy_input)
            return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        if self.conv_layers is not None:
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
        
        return self.fc_layers(x)

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for policy gradient methods
    Supports both discrete and continuous action spaces
    """
    
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int, 
                 continuous: bool = False, hidden_size: int = 512):
        super(ActorCriticNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.continuous = continuous
        
        # Shared feature extraction
        if len(input_shape) == 3:  # Image input
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            conv_out_size = self._get_conv_out_size(input_shape)
            feature_size = conv_out_size
        else:
            self.feature_extractor = None
            feature_size = np.prod(input_shape)
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        if continuous:
            # Continuous actions: output mean and std
            self.actor_mean = nn.Linear(hidden_size, num_actions)
            self.actor_std = nn.Linear(hidden_size, num_actions)
        else:
            # Discrete actions: output action probabilities
            self.actor = nn.Linear(hidden_size, num_actions)
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
    
    def _get_conv_out_size(self, shape):
        """Calculate convolutional output size"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            conv_out = self.feature_extractor(dummy_input)
            return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        # Feature extraction
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
        
        # Shared processing
        features = self.shared_layers(x)
        
        # Actor output
        if self.continuous:
            action_mean = self.actor_mean(features)
            action_std = F.softplus(self.actor_std(features))
            actor_output = (action_mean, action_std)
        else:
            action_logits = self.actor(features)
            actor_output = F.softmax(action_logits, dim=-1)
        
        # Critic output
        value = self.critic(features)
        
        return actor_output, value

class AdvancedRLAgent:
    """
    Advanced Reinforcement Learning Agent
    Supports multiple algorithms: DQN, PPO, DDPG, TD3
    """
    
    def __init__(self, algorithm: str = "DQN", input_shape: Tuple[int, ...] = (84, 84, 4), 
                 num_actions: int = 4, continuous: bool = False, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for reinforcement learning")
        
        self.algorithm = algorithm.upper()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.continuous = continuous
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Algorithm-specific initialization
        self._initialize_algorithm()
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        self.logger.info(f"Advanced RL Agent initialized: {algorithm} on {self.device}")
    
    def _initialize_algorithm(self):
        """Initialize algorithm-specific components"""
        if self.algorithm == "DQN":
            self._initialize_dqn()
        elif self.algorithm == "PPO":
            self._initialize_ppo()
        elif self.algorithm == "DDPG":
            self._initialize_ddpg()
        elif self.algorithm == "TD3":
            self._initialize_td3()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _initialize_dqn(self):
        """Initialize DQN components"""
        self.q_network = DQNNetwork(self.input_shape, self.num_actions).to(self.device)
        self.target_q_network = DQNNetwork(self.input_shape, self.num_actions).to(self.device)
        
        # Copy parameters to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.experience_buffer = ExperienceBuffer(buffer_size=100000, prioritized=True)
        
        # DQN hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.target_update_freq = 1000
    
    def _initialize_ppo(self):
        """Initialize PPO components"""
        self.actor_critic = ActorCriticNetwork(
            self.input_shape, self.num_actions, self.continuous
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.0003)
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.batch_size = 64
        self.ppo_epochs = 4
        
        # PPO rollout storage
        self.rollout_buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def _initialize_ddpg(self):
        """Initialize DDPG components"""
        if not self.continuous:
            raise ValueError("DDPG requires continuous action space")
        
        # Actor and Critic networks
        self.actor = ActorCriticNetwork(
            self.input_shape, self.num_actions, continuous=True
        ).to(self.device)
        
        self.critic = DQNNetwork(
            (self.input_shape[0] + self.num_actions,) + self.input_shape[1:], 1
        ).to(self.device)
        
        # Target networks
        self.target_actor = ActorCriticNetwork(
            self.input_shape, self.num_actions, continuous=True
        ).to(self.device)
        
        self.target_critic = DQNNetwork(
            (self.input_shape[0] + self.num_actions,) + self.input_shape[1:], 1
        ).to(self.device)
        
        # Copy parameters
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        self.experience_buffer = ExperienceBuffer(buffer_size=100000)
        
        # DDPG hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # Soft update rate
        self.batch_size = 32
        self.noise_std = 0.1
    
    def _initialize_td3(self):
        """Initialize TD3 components (similar to DDPG but with twin critics)"""
        if not self.continuous:
            raise ValueError("TD3 requires continuous action space")
        
        # Similar to DDPG but with twin critics
        self._initialize_ddpg()
        
        # Add second critic
        self.critic2 = DQNNetwork(
            (self.input_shape[0] + self.num_actions,) + self.input_shape[1:], 1
        ).to(self.device)
        
        self.target_critic2 = DQNNetwork(
            (self.input_shape[0] + self.num_actions,) + self.input_shape[1:], 1
        ).to(self.device)
        
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)
        
        # TD3 specific hyperparameters
        self.policy_delay = 2
        self.target_noise = 0.2
        self.noise_clip = 0.5
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """Select action based on current policy"""
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        if self.algorithm == "DQN":
            return self._select_action_dqn(observation, training)
        elif self.algorithm == "PPO":
            return self._select_action_ppo(observation, training)
        elif self.algorithm in ["DDPG", "TD3"]:
            return self._select_action_ddpg(observation, training)
    
    def _select_action_dqn(self, observation: torch.Tensor, training: bool) -> int:
        """DQN action selection with epsilon-greedy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        with torch.no_grad():
            q_values = self.q_network(observation)
            return q_values.argmax().item()
    
    def _select_action_ppo(self, observation: torch.Tensor, training: bool) -> Union[int, np.ndarray]:
        """PPO action selection"""
        with torch.no_grad():
            actor_output, value = self.actor_critic(observation)
            
            if self.continuous:
                mean, std = actor_output
                if training:
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    return action.cpu().numpy().flatten()
                else:
                    return mean.cpu().numpy().flatten()
            else:
                action_probs = actor_output
                if training:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    return action.item()
                else:
                    return action_probs.argmax().item()
    
    def _select_action_ddpg(self, observation: torch.Tensor, training: bool) -> np.ndarray:
        """DDPG/TD3 action selection"""
        with torch.no_grad():
            actor_output, _ = self.actor(observation)
            mean, _ = actor_output
            
            action = mean.cpu().numpy().flatten()
            
            if training:
                # Add exploration noise
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            
            return action
    
    def train_step(self, observation: np.ndarray, action: Union[int, np.ndarray], 
                   reward: float, next_observation: np.ndarray, done: bool):
        """Perform one training step"""
        # Add experience to buffer
        if hasattr(self, 'experience_buffer'):
            self.experience_buffer.add(observation, action, reward, next_observation, done)
        
        # Algorithm-specific training
        if self.algorithm == "DQN":
            return self._train_dqn()
        elif self.algorithm == "PPO":
            return self._train_ppo(observation, action, reward, next_observation, done)
        elif self.algorithm == "DDPG":
            return self._train_ddpg()
        elif self.algorithm == "TD3":
            return self._train_td3()
    
    def _train_dqn(self) -> Dict[str, float]:
        """DQN training step"""
        if len(self.experience_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = self.experience_buffer.sample(self.batch_size)
        
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_observations = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_observations).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.training_step += 1
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def _train_ppo(self, observation: np.ndarray, action: Union[int, np.ndarray], 
                   reward: float, next_observation: np.ndarray, done: bool) -> Dict[str, float]:
        """PPO training step (simplified rollout collection)"""
        # Add to rollout buffer
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actor_output, value = self.actor_critic(obs_tensor)
            
            if self.continuous:
                mean, std = actor_output
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(torch.FloatTensor(action).to(self.device)).sum()
            else:
                action_probs = actor_output
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(torch.LongTensor([action]).to(self.device))
        
        self.rollout_buffer['observations'].append(observation)
        self.rollout_buffer['actions'].append(action)
        self.rollout_buffer['rewards'].append(reward)
        self.rollout_buffer['values'].append(value.item())
        self.rollout_buffer['log_probs'].append(log_prob.item())
        self.rollout_buffer['dones'].append(done)
        
        # Train when rollout is complete or buffer is full
        if done or len(self.rollout_buffer['observations']) >= self.batch_size:
            return self._ppo_update()
        
        return {}
    
    def _ppo_update(self) -> Dict[str, float]:
        """Perform PPO update"""
        if len(self.rollout_buffer['observations']) == 0:
            return {}
        
        # Convert to tensors
        observations = torch.FloatTensor(self.rollout_buffer['observations']).to(self.device)
        actions = torch.LongTensor(self.rollout_buffer['actions']).to(self.device) if not self.continuous else torch.FloatTensor(self.rollout_buffer['actions']).to(self.device)
        rewards = torch.FloatTensor(self.rollout_buffer['rewards']).to(self.device)
        old_values = torch.FloatTensor(self.rollout_buffer['values']).to(self.device)
        old_log_probs = torch.FloatTensor(self.rollout_buffer['log_probs']).to(self.device)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, old_values)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        
        # PPO update epochs
        for _ in range(self.ppo_epochs):
            # Forward pass
            actor_output, values = self.actor_critic(observations)
            
            if self.continuous:
                mean, std = actor_output
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
            else:
                action_probs = actor_output
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            
            # PPO loss components
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy_loss = -self.entropy_coeff * entropy
            
            loss = policy_loss + self.value_coeff * value_loss + entropy_loss
            total_loss += loss.item()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear rollout buffer
        for key in self.rollout_buffer:
            self.rollout_buffer[key].clear()
        
        self.training_step += 1
        
        return {'loss': total_loss / self.ppo_epochs}
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage
            advantages[t] = advantage
        
        return advantages
    
    def _train_ddpg(self) -> Dict[str, float]:
        """DDPG training step"""
        if len(self.experience_buffer) < self.batch_size:
            return {}
        
        batch = self.experience_buffer.sample(self.batch_size)
        
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_observations = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions, _ = self.target_actor(next_observations)
            next_actions = next_actions[0]  # Get mean
            target_q = self.target_critic(torch.cat([next_observations, next_actions], dim=1))
            target_q = rewards.unsqueeze(1) + (self.gamma * target_q * ~dones.unsqueeze(1))
        
        current_q = self.critic(torch.cat([observations, actions], dim=1))
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_actions, _ = self.actor(observations)
        actor_actions = actor_actions[0]  # Get mean
        actor_loss = -self.critic(torch.cat([observations, actor_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        self.training_step += 1
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
    
    def _train_td3(self) -> Dict[str, float]:
        """TD3 training step"""
        # Similar to DDPG but with delayed policy updates and twin critics
        return self._train_ddpg()  # Simplified for now
    
    def _soft_update(self, target_network, source_network):
        """Soft update of target network"""
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save model state"""
        try:
            state = {
                'algorithm': self.algorithm,
                'training_step': self.training_step,
                'episode_count': self.episode_count
            }
            
            if self.algorithm == "DQN":
                state['q_network'] = self.q_network.state_dict()
                state['optimizer'] = self.optimizer.state_dict()
                state['epsilon'] = self.epsilon
            elif self.algorithm == "PPO":
                state['actor_critic'] = self.actor_critic.state_dict()
                state['optimizer'] = self.optimizer.state_dict()
            elif self.algorithm in ["DDPG", "TD3"]:
                state['actor'] = self.actor.state_dict()
                state['critic'] = self.critic.state_dict()
                state['actor_optimizer'] = self.actor_optimizer.state_dict()
                state['critic_optimizer'] = self.critic_optimizer.state_dict()
            
            torch.save(state, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        try:
            state = torch.load(filepath, map_location=self.device)
            
            self.training_step = state.get('training_step', 0)
            self.episode_count = state.get('episode_count', 0)
            
            if self.algorithm == "DQN":
                self.q_network.load_state_dict(state['q_network'])
                self.target_q_network.load_state_dict(state['q_network'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.epsilon = state.get('epsilon', self.epsilon)
            elif self.algorithm == "PPO":
                self.actor_critic.load_state_dict(state['actor_critic'])
                self.optimizer.load_state_dict(state['optimizer'])
            elif self.algorithm in ["DDPG", "TD3"]:
                self.actor.load_state_dict(state['actor'])
                self.critic.load_state_dict(state['critic'])
                self.target_actor.load_state_dict(state['actor'])
                self.target_critic.load_state_dict(state['critic'])
                self.actor_optimizer.load_state_dict(state['actor_optimizer'])
                self.critic_optimizer.load_state_dict(state['critic_optimizer'])
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            'algorithm': self.algorithm,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        }
        
        if self.algorithm == "DQN":
            stats['epsilon'] = self.epsilon
        
        return stats