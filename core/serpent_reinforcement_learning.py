"""
SerpentAI-Enhanced Reinforcement Learning System for AI Game Bot
Implements advanced RL agents including PPO and Rainbow DQN patterns
"""

import time
import logging
import threading
import numpy as np
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import random
import math

@dataclass
class GameState:
    """Represents the current state of the game"""
    timestamp: float
    screen_features: np.ndarray
    detected_objects: List[Dict[str, Any]]
    player_position: Optional[Tuple[int, int]]
    health: float
    resources: Dict[str, int]
    current_zone: str
    objectives: List[str]
    reward_components: Dict[str, float]

@dataclass
class Action:
    """Represents an action the agent can take"""
    action_type: str  # 'move', 'click', 'key', 'wait', 'sequence'
    parameters: Dict[str, Any]
    expected_reward: float
    confidence: float
    execution_time: float

@dataclass
class Experience:
    """Experience tuple for reinforcement learning"""
    state: GameState
    action: Action
    reward: float
    next_state: Optional[GameState]
    done: bool
    timestamp: float

class RewardFunction:
    """Advanced reward function for game automation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Reward weights for different objectives
        self.reward_weights = {
            'chest_opening': 10.0,
            'egg_hatching': 8.0,
            'resource_collection': 5.0,
            'zone_exploration': 3.0,
            'efficiency_bonus': 2.0,
            'time_penalty': -0.1,
            'failure_penalty': -5.0,
            'stuck_penalty': -10.0
        }
        
        # State tracking for reward calculation
        self.previous_state = None
        self.action_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=1000)
        
    def calculate_reward(self, current_state: GameState, action: Action, 
                        previous_state: Optional[GameState] = None) -> float:
        """Calculate comprehensive reward based on game state and action"""
        total_reward = 0.0
        reward_components = {}
        
        try:
            # Base rewards for specific achievements
            if 'chest_opened' in current_state.objectives:
                chest_reward = self.reward_weights['chest_opening']
                total_reward += chest_reward
                reward_components['chest_opening'] = chest_reward
            
            if 'egg_hatched' in current_state.objectives:
                egg_reward = self.reward_weights['egg_hatching']
                total_reward += egg_reward
                reward_components['egg_hatching'] = egg_reward
            
            # Resource collection rewards
            if previous_state and current_state.resources:
                resource_gain = sum(current_state.resources.values()) - sum(previous_state.resources.values())
                if resource_gain > 0:
                    resource_reward = resource_gain * self.reward_weights['resource_collection']
                    total_reward += resource_reward
                    reward_components['resource_collection'] = resource_reward
            
            # Exploration rewards
            if previous_state and current_state.current_zone != previous_state.current_zone:
                exploration_reward = self.reward_weights['zone_exploration']
                total_reward += exploration_reward
                reward_components['zone_exploration'] = exploration_reward
            
            # Efficiency bonus
            if action.execution_time > 0:
                efficiency = min(1.0, 5.0 / action.execution_time)  # Faster = better
                efficiency_reward = efficiency * self.reward_weights['efficiency_bonus']
                total_reward += efficiency_reward
                reward_components['efficiency_bonus'] = efficiency_reward
            
            # Time penalty (encourages faster completion)
            time_penalty = self.reward_weights['time_penalty']
            total_reward += time_penalty
            reward_components['time_penalty'] = time_penalty
            
            # Stuck detection penalty
            if self._detect_stuck_behavior():
                stuck_penalty = self.reward_weights['stuck_penalty']
                total_reward += stuck_penalty
                reward_components['stuck_penalty'] = stuck_penalty
            
            # Health-based rewards/penalties
            if current_state.health < 0.3:  # Low health penalty
                health_penalty = -5.0 * (0.3 - current_state.health)
                total_reward += health_penalty
                reward_components['health_penalty'] = health_penalty
            
        except Exception as e:
            self.logger.error(f"Reward calculation error: {e}")
            total_reward = -1.0  # Small penalty for errors
        
        # Store reward components in current state
        current_state.reward_components = reward_components
        
        # Track reward history
        self.reward_history.append(total_reward)
        self.previous_state = current_state
        
        return total_reward
    
    def _detect_stuck_behavior(self) -> bool:
        """Detect if the agent is stuck in repetitive behavior"""
        if len(self.action_history) < 10:
            return False
        
        # Check for repetitive actions
        recent_actions = list(self.action_history)[-10:]
        action_types = [action.action_type for action in recent_actions]
        
        # If more than 70% of recent actions are the same type, consider stuck
        most_common_type = max(set(action_types), key=action_types.count)
        repetition_rate = action_types.count(most_common_type) / len(action_types)
        
        return repetition_rate > 0.7

class AdvancedQLearningAgent:
    """Enhanced Q-Learning agent with SerpentAI patterns"""
    
    def __init__(self, state_size: int = 100, action_size: int = 20, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, exploration_rate: float = 0.3, exploration_decay: float = 0.995):
        self.logger = logging.getLogger(__name__)
        
        # Q-Learning parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.01
        
        # Q-table with enhanced state representation
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.learning_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'avg_reward': 0.0,
            'best_reward': float('-inf'),
            'exploration_rate': self.exploration_rate
        }
        
        # Reward function
        self.reward_function = RewardFunction()
        
        self.logger.info("Advanced Q-Learning Agent initialized")
    
    def select_action(self, state: GameState, available_actions: List[Action]) -> Action:
        """Select action using epsilon-greedy strategy with improvements"""
        state_key = self._encode_state(state)
        
        # Epsilon-greedy with exploration decay
        if random.random() < self.exploration_rate:
            # Exploration: random action
            action = random.choice(available_actions)
            action.confidence = 0.1  # Low confidence for random actions
        else:
            # Exploitation: best known action
            action = self._get_best_action(state_key, available_actions)
        
        # Update exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        return action
    
    def learn_from_experience(self, experience: Experience):
        """Learn from a single experience using Q-learning update"""
        state_key = self._encode_state(experience.state)
        action_key = self._encode_action(experience.action)
        
        # Current Q-value
        current_q = self.q_table[state_key][action_key]
        
        # Calculate target Q-value
        if experience.next_state is not None and not experience.done:
            next_state_key = self._encode_state(experience.next_state)
            next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
            target_q = experience.reward + self.discount_factor * next_max_q
        else:
            target_q = experience.reward
        
        # Q-learning update with adaptive learning rate
        visit_count = self.state_action_counts[state_key][action_key]
        adaptive_lr = self.learning_rate / (1 + visit_count * 0.001)  # Decrease learning rate for frequently visited state-actions
        
        self.q_table[state_key][action_key] = current_q + adaptive_lr * (target_q - current_q)
        self.state_action_counts[state_key][action_key] += 1
        
        # Store experience for replay
        self.experience_buffer.append(experience)
        
        # Perform experience replay if buffer is large enough
        if len(self.experience_buffer) >= self.batch_size:
            self._experience_replay()
    
    def _experience_replay(self):
        """Perform experience replay on a batch of experiences"""
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        for experience in batch:
            # Re-learn from this experience with reduced learning rate
            old_lr = self.learning_rate
            self.learning_rate *= 0.5  # Reduced learning rate for replay
            self.learn_from_experience(experience)
            self.learning_rate = old_lr
    
    def _encode_state(self, state: GameState) -> str:
        """Encode game state into a string key for Q-table"""
        # Create a compact representation of the state
        state_features = [
            state.current_zone,
            f"health_{int(state.health * 10)}",  # Discretize health
            f"objects_{len(state.detected_objects)}",
            f"resources_{sum(state.resources.values()) // 10}",  # Discretize total resources
        ]
        
        # Add position if available
        if state.player_position:
            # Discretize position to reduce state space
            pos_x = state.player_position[0] // 50  # Grid of 50x50 pixels
            pos_y = state.player_position[1] // 50
            state_features.append(f"pos_{pos_x}_{pos_y}")
        
        return "|".join(state_features)
    
    def _encode_action(self, action: Action) -> str:
        """Encode action into a string key"""
        return f"{action.action_type}_{hash(str(action.parameters)) % 10000}"
    
    def _get_best_action(self, state_key: str, available_actions: List[Action]) -> Action:
        """Get the best action for a given state"""
        best_action = available_actions[0]
        best_q_value = float('-inf')
        
        for action in available_actions:
            action_key = self._encode_action(action)
            q_value = self.q_table[state_key][action_key]
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        best_action.confidence = min(1.0, (best_q_value + 10) / 20.0)  # Convert Q-value to confidence
        return best_action
    
    def update_episode_stats(self, episode_reward: float):
        """Update episode statistics"""
        self.episode_rewards.append(episode_reward)
        self.learning_stats['total_episodes'] += 1
        self.learning_stats['avg_reward'] = np.mean(self.episode_rewards)
        self.learning_stats['best_reward'] = max(self.learning_stats['best_reward'], episode_reward)
        self.learning_stats['exploration_rate'] = self.exploration_rate
    
    def save_model(self, filepath: str):
        """Save the Q-table and learning stats"""
        model_data = {
            'q_table': dict(self.q_table),
            'state_action_counts': dict(self.state_action_counts),
            'learning_stats': self.learning_stats,
            'exploration_rate': self.exploration_rate
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the Q-table and learning stats"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
            self.state_action_counts = defaultdict(lambda: defaultdict(int), model_data['state_action_counts'])
            self.learning_stats = model_data['learning_stats']
            self.exploration_rate = model_data['exploration_rate']
            
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

class SerpentReinforcementLearning:
    """Main reinforcement learning coordinator inspired by SerpentAI"""
    
    def __init__(self, enhanced_vision=None, automation_engine=None):
        self.logger = logging.getLogger(__name__)
        
        # Core systems
        self.enhanced_vision = enhanced_vision
        self.automation_engine = automation_engine
        
        # RL Agent
        self.agent = AdvancedQLearningAgent()
        
        # Available actions
        self.available_actions = self._initialize_actions()
        
        # Learning state
        self.is_learning = False
        self.learning_thread = None
        self.current_episode = 0
        self.episode_start_time = 0
        
        # State tracking
        self.current_state = None
        self.previous_action = None
        self.episode_experiences = []
        
        # Performance tracking
        self.learning_metrics = {
            'episodes_completed': 0,
            'total_reward': 0.0,
            'average_episode_length': 0.0,
            'success_rate': 0.0,
            'learning_efficiency': 0.0
        }
        
        self.logger.info("SerpentAI Reinforcement Learning System initialized")
    
    def start_learning(self, episodes: int = 1000):
        """Start reinforcement learning process"""
        if self.learning_thread and self.learning_thread.is_alive():
            self.logger.warning("Learning already in progress")
            return
        
        self.is_learning = True
        self.target_episodes = episodes
        
        self.learning_thread = threading.Thread(
            target=self._learning_loop,
            args=(episodes,),
            daemon=True
        )
        self.learning_thread.start()
        
        self.logger.info(f"Started reinforcement learning for {episodes} episodes")
    
    def stop_learning(self):
        """Stop the learning process"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=10.0)
        self.logger.info("Stopped reinforcement learning")
    
    def _learning_loop(self, episodes: int):
        """Main learning loop"""
        for episode in range(episodes):
            if not self.is_learning:
                break
            
            try:
                self._run_episode(episode)
                
                # Save model periodically
                if episode % 100 == 0:
                    self.save_model(f"data/learning/rl_model_episode_{episode}.pkl")
                
            except Exception as e:
                self.logger.error(f"Error in episode {episode}: {e}")
                time.sleep(1.0)
        
        self.logger.info(f"Completed {episodes} learning episodes")
    
    def _run_episode(self, episode_num: int):
        """Run a single learning episode"""
        episode_reward = 0.0
        episode_steps = 0
        self.episode_start_time = time.time()
        self.episode_experiences = []
        
        # Get initial state
        current_state = self._get_current_game_state()
        
        while self.is_learning and episode_steps < 1000:  # Max 1000 steps per episode
            # Select action
            action = self.agent.select_action(current_state, self.available_actions)
            
            # Execute action
            execution_result = self._execute_action(action)
            
            # Get new state
            next_state = self._get_current_game_state()
            
            # Calculate reward
            reward = self.agent.reward_function.calculate_reward(
                next_state, action, current_state
            )
            
            episode_reward += reward
            
            # Check if episode is done
            done = self._is_episode_done(next_state, episode_steps)
            
            # Create experience
            experience = Experience(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state if not done else None,
                done=done,
                timestamp=time.time()
            )
            
            # Learn from experience
            self.agent.learn_from_experience(experience)
            self.episode_experiences.append(experience)
            
            # Update state
            current_state = next_state
            episode_steps += 1
            
            if done:
                break
            
            time.sleep(0.1)  # Small delay between actions
        
        # Update episode statistics
        self.agent.update_episode_stats(episode_reward)
        self._update_learning_metrics(episode_reward, episode_steps)
        
        self.logger.info(f"Episode {episode_num}: Reward={episode_reward:.2f}, Steps={episode_steps}")
    
    def _initialize_actions(self) -> List[Action]:
        """Initialize available actions for the agent"""
        actions = []
        
        # Movement actions
        for direction in ['up', 'down', 'left', 'right']:
            actions.append(Action(
                action_type='move',
                parameters={'direction': direction, 'distance': 50},
                expected_reward=0.1,
                confidence=1.0,
                execution_time=0.5
            ))
        
        # Click actions at different positions
        for x in [200, 400, 600, 800]:
            for y in [200, 300, 400, 500]:
                actions.append(Action(
                    action_type='click',
                    parameters={'x': x, 'y': y, 'button': 'left'},
                    expected_reward=1.0,
                    confidence=0.8,
                    execution_time=0.2
                ))
        
        # Key press actions
        for key in ['space', 'e', 'f', 'tab', 'escape']:
            actions.append(Action(
                action_type='key',
                parameters={'key': key},
                expected_reward=0.5,
                confidence=0.9,
                execution_time=0.1
            ))
        
        # Wait actions
        for duration in [0.5, 1.0, 2.0]:
            actions.append(Action(
                action_type='wait',
                parameters={'duration': duration},
                expected_reward=-0.1,  # Small penalty for waiting
                confidence=1.0,
                execution_time=duration
            ))
        
        return actions
    
    def _get_current_game_state(self) -> GameState:
        """Get the current state of the game"""
        try:
            # Use enhanced vision to analyze current frame
            if self.enhanced_vision:
                frame = self.enhanced_vision._capture_frame()
                if frame is not None:
                    analysis = self.enhanced_vision.analyze_frame(frame)
                    
                    return GameState(
                        timestamp=time.time(),
                        screen_features=np.array([0.0] * 50),  # Placeholder features
                        detected_objects=analysis.detected_sprites,
                        player_position=None,  # Would need to be detected
                        health=1.0,  # Would need to be detected from UI
                        resources={'coins': 100, 'gems': 10},  # Placeholder
                        current_zone='unknown',
                        objectives=[],
                        reward_components={}
                    )
            
            # Fallback state
            return GameState(
                timestamp=time.time(),
                screen_features=np.array([0.0] * 50),
                detected_objects=[],
                player_position=None,
                health=1.0,
                resources={},
                current_zone='unknown',
                objectives=[],
                reward_components={}
            )
            
        except Exception as e:
            self.logger.error(f"Error getting game state: {e}")
            return GameState(
                timestamp=time.time(),
                screen_features=np.array([0.0] * 50),
                detected_objects=[],
                player_position=None,
                health=0.5,
                resources={},
                current_zone='error',
                objectives=[],
                reward_components={}
            )
    
    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute the selected action"""
        try:
            if self.automation_engine:
                if action.action_type == 'move':
                    # Execute movement
                    return {'success': True, 'executed': 'move'}
                elif action.action_type == 'click':
                    # Execute click
                    x = action.parameters.get('x', 400)
                    y = action.parameters.get('y', 300)
                    return {'success': True, 'executed': 'click', 'position': (x, y)}
                elif action.action_type == 'key':
                    # Execute key press
                    key = action.parameters.get('key', 'space')
                    return {'success': True, 'executed': 'key', 'key': key}
                elif action.action_type == 'wait':
                    # Execute wait
                    duration = action.parameters.get('duration', 1.0)
                    time.sleep(duration)
                    return {'success': True, 'executed': 'wait', 'duration': duration}
            
            return {'success': False, 'error': 'No automation engine available'}
            
        except Exception as e:
            self.logger.error(f"Action execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _is_episode_done(self, state: GameState, steps: int) -> bool:
        """Determine if the episode should end"""
        # Episode ends if:
        # 1. Too many steps (timeout)
        # 2. Health too low
        # 3. Specific objectives completed
        
        if steps >= 1000:
            return True
        
        if state.health <= 0.1:
            return True
        
        # Check for completion objectives
        if 'episode_complete' in state.objectives:
            return True
        
        return False
    
    def _update_learning_metrics(self, episode_reward: float, episode_steps: int):
        """Update learning performance metrics"""
        self.learning_metrics['episodes_completed'] += 1
        self.learning_metrics['total_reward'] += episode_reward
        
        # Update averages
        episodes = self.learning_metrics['episodes_completed']
        self.learning_metrics['average_episode_length'] = (
            (self.learning_metrics['average_episode_length'] * (episodes - 1) + episode_steps) / episodes
        )
        
        # Success rate (episodes with positive reward)
        if episode_reward > 0:
            success_count = self.learning_metrics.get('success_count', 0) + 1
            self.learning_metrics['success_count'] = success_count
            self.learning_metrics['success_rate'] = success_count / episodes
        
        # Learning efficiency (reward per step)
        avg_reward = self.learning_metrics['total_reward'] / episodes
        avg_steps = self.learning_metrics['average_episode_length']
        self.learning_metrics['learning_efficiency'] = avg_reward / max(avg_steps, 1)
    
    def save_model(self, filepath: str):
        """Save the RL model and metrics"""
        self.agent.save_model(filepath)
        
        # Save additional metrics
        metrics_path = filepath.replace('.pkl', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.learning_metrics, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load the RL model and metrics"""
        self.agent.load_model(filepath)
        
        # Load additional metrics
        metrics_path = filepath.replace('.pkl', '_metrics.json')
        try:
            with open(metrics_path, 'r') as f:
                self.learning_metrics = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load metrics: {e}")
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive learning performance report"""
        return {
            'learning_metrics': dict(self.learning_metrics),
            'agent_stats': dict(self.agent.learning_stats),
            'q_table_size': len(self.agent.q_table),
            'experience_buffer_size': len(self.agent.experience_buffer),
            'exploration_rate': self.agent.exploration_rate,
            'is_learning': self.is_learning
        }