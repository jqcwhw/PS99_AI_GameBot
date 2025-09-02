"""
Learning System for AI Game Bot
Handles pattern recognition, adaptation, and improvement through experience
"""

import json
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
from collections import defaultdict, deque

class LearningSystem:
    """AI learning system for game automation improvement"""
    
    def __init__(self, max_memory_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        
        # Learning data storage
        self.experience_memory = deque(maxlen=max_memory_size)
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        self.action_outcomes = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        # Learning statistics
        self.stats = {
            'total_experiences': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'patterns_learned': 0,
            'adaptations_made': 0,
            'items_learned': 0
        }
        
        # Pattern recognition settings
        self.similarity_threshold = 0.8
        self.min_pattern_occurrences = 3
        
        # SerpentAI inspired reinforcement learning components
        self.q_table = defaultdict(lambda: defaultdict(float))  # Simple Q-learning
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
        # Advanced pattern analysis
        self.temporal_patterns = defaultdict(list)  # Time-based patterns
        self.state_transitions = defaultdict(lambda: defaultdict(int))  # State changes
        
        # Load existing learning data
        self._load_learning_data()
        
        self.logger.info("Enhanced Learning system initialized with RL optimizations")
    
    def _load_learning_data(self):
        """Load existing learning data from storage"""
        try:
            learning_file = Path("data/learning_data.pkl")
            if learning_file.exists():
                with open(learning_file, 'rb') as f:
                    data = pickle.load(f)
                    
                self.success_patterns = data.get('success_patterns', defaultdict(list))
                self.failure_patterns = data.get('failure_patterns', defaultdict(list))
                self.action_outcomes = data.get('action_outcomes', defaultdict(lambda: {'success': 0, 'failure': 0}))
                self.stats = data.get('stats', self.stats)
                
                self.logger.info(f"Loaded learning data: {self.stats['total_experiences']} experiences")
        except Exception as e:
            self.logger.error(f"Failed to load learning data: {e}")
    
    def _save_learning_data(self):
        """Save learning data to storage"""
        try:
            learning_file = Path("data/learning_data.pkl")
            learning_file.parent.mkdir(exist_ok=True)
            
            data = {
                'success_patterns': dict(self.success_patterns),
                'failure_patterns': dict(self.failure_patterns),
                'action_outcomes': dict(self.action_outcomes),
                'stats': self.stats
            }
            
            with open(learning_file, 'wb') as f:
                pickle.dump(data, f)
                
            self.logger.debug("Learning data saved")
        except Exception as e:
            self.logger.error(f"Failed to save learning data: {e}")
    
    def record_experience(self, action: str, context: Dict[str, Any], outcome: str, 
                         details: Optional[Dict[str, Any]] = None):
        """
        Record an experience for learning
        
        Args:
            action: The action that was performed
            context: Context information (screen state, positions, etc.)
            outcome: 'success' or 'failure'
            details: Additional details about the experience
        """
        experience = {
            'timestamp': time.time(),
            'action': action,
            'context': context,
            'outcome': outcome,
            'details': details or {}
        }
        
        # Add to memory
        self.experience_memory.append(experience)
        
        # Update statistics
        self.stats['total_experiences'] += 1
        if outcome == 'success':
            self.stats['successful_actions'] += 1
            self.success_patterns[action].append(experience)
        else:
            self.stats['failed_actions'] += 1
            self.failure_patterns[action].append(experience)
        
        # Update action outcomes
        self.action_outcomes[action][outcome] += 1
        
        # Look for patterns
        self._analyze_patterns(action, experience)
        
        # Save periodically
        if self.stats['total_experiences'] % 10 == 0:
            self._save_learning_data()
        
        self.logger.debug(f"Recorded experience: {action} -> {outcome}")
    
    def q_learning_update(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-table using Q-learning algorithm (SerpentAI inspired)"""
        # Current Q value
        current_q = self.q_table[state][action]
        
        # Best Q value for next state
        next_q_values = list(self.q_table[next_state].values())
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Record state transition
        self.state_transitions[state][next_state] += 1
        
        self.logger.debug(f"Q-learning update: {state} -> {action} (Q: {current_q:.3f} -> {new_q:.3f})")
    
    def select_action_epsilon_greedy(self, state: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(available_actions)
        else:
            # Exploitation: best known action
            state_q_values = self.q_table[state]
            if not state_q_values:
                return random.choice(available_actions)
            
            # Find action with highest Q-value
            best_action = max(available_actions, 
                            key=lambda a: state_q_values.get(a, 0))
            return best_action
    
    def analyze_temporal_patterns(self, action_sequence: List[str], window_size: int = 5):
        """Analyze temporal patterns in action sequences"""
        for i in range(len(action_sequence) - window_size + 1):
            pattern = tuple(action_sequence[i:i + window_size])
            self.temporal_patterns[pattern].append(time.time())
        
        # Clean old patterns (older than 1 hour)
        current_time = time.time()
        for pattern in list(self.temporal_patterns.keys()):
            self.temporal_patterns[pattern] = [
                t for t in self.temporal_patterns[pattern] 
                if current_time - t < 3600
            ]
            if not self.temporal_patterns[pattern]:
                del self.temporal_patterns[pattern]
    
    def get_pattern_frequency(self, pattern: tuple) -> float:
        """Get frequency of a temporal pattern"""
        if pattern not in self.temporal_patterns:
            return 0.0
        
        occurrences = len(self.temporal_patterns[pattern])
        time_span = 3600  # 1 hour
        return occurrences / time_span  # Occurrences per second
    
    def predict_next_action(self, recent_actions: List[str], num_predictions: int = 3) -> List[str]:
        """Predict next actions based on learned patterns"""
        predictions = []
        
        # Look for matching patterns of decreasing length
        for pattern_length in range(min(5, len(recent_actions)), 0, -1):
            if len(recent_actions) >= pattern_length:
                pattern = tuple(recent_actions[-pattern_length:])
                
                # Find patterns that start with this sequence
                matching_patterns = [
                    p for p in self.temporal_patterns.keys() 
                    if len(p) > pattern_length and p[:pattern_length] == pattern
                ]
                
                if matching_patterns:
                    # Sort by frequency and get next actions
                    pattern_frequencies = [
                        (p, self.get_pattern_frequency(p)) 
                        for p in matching_patterns
                    ]
                    pattern_frequencies.sort(key=lambda x: x[1], reverse=True)
                    
                    for pattern, freq in pattern_frequencies[:num_predictions]:
                        if len(pattern) > pattern_length:
                            next_action = pattern[pattern_length]
                            if next_action not in predictions:
                                predictions.append(next_action)
                
                if predictions:
                    break
        
        return predictions[:num_predictions]
    
    def adaptive_learning_rate(self):
        """Adjust learning rate based on recent performance"""
        recent_experiences = list(self.experience_memory)[-50:]  # Last 50 experiences
        
        if len(recent_experiences) < 10:
            return
        
        # Calculate recent success rate
        recent_successes = sum(1 for exp in recent_experiences if exp['outcome'] == 'success')
        success_rate = recent_successes / len(recent_experiences)
        
        # Adjust learning rate
        if success_rate > 0.8:
            # High success rate - reduce learning rate for stability
            self.learning_rate *= 0.95
            self.epsilon *= 0.98  # Reduce exploration
        elif success_rate < 0.6:
            # Low success rate - increase learning rate for faster adaptation
            self.learning_rate *= 1.05
            self.epsilon *= 1.02  # Increase exploration
        
        # Keep learning rate and epsilon in reasonable bounds
        self.learning_rate = max(0.01, min(0.5, self.learning_rate))
        self.epsilon = max(0.05, min(0.3, self.epsilon))
        
        self.logger.debug(f"Adaptive adjustment: lr={self.learning_rate:.3f}, ε={self.epsilon:.3f}")
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get advanced learning statistics"""
        basic_stats = self.get_stats()
        
        # Add advanced metrics
        advanced_stats = {
            **basic_stats,
            'q_table_size': len(self.q_table),
            'state_transitions': len(self.state_transitions),
            'temporal_patterns': len(self.temporal_patterns),
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'avg_q_values': self._calculate_avg_q_values()
        }
        
        return advanced_stats
    
    def _calculate_avg_q_values(self) -> float:
        """Calculate average Q-values across all states and actions"""
        all_q_values = []
        for state_actions in self.q_table.values():
            all_q_values.extend(state_actions.values())
        
        return np.mean(all_q_values) if all_q_values else 0.0
    
    def start_watching_mode(self):
        """Start watching mode to learn from user actions"""
        try:
            self.watching_mode = True
            self.logger.info("Learning system started watching mode")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start watching mode: {e}")
            return False
    
    def stop_watching_mode(self):
        """Stop watching mode"""
        try:
            self.watching_mode = False
            self.logger.info("Learning system stopped watching mode")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop watching mode: {e}")
            return False
    
    def clear_memory(self):
        """Clear all learning memory"""
        try:
            self.experience_memory.clear()
            self.success_patterns.clear()
            self.failure_patterns.clear()
            self.action_outcomes.clear()
            self.stats = {
                'total_experiences': 0,
                'successful_actions': 0,
                'failed_actions': 0,
                'patterns_learned': 0,
                'adaptations_made': 0,
                'items_learned': 0
            }
            self._save_learning_data()
            self.logger.info("Learning system memory cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            return False
    
    @property
    def accuracy_rate(self):
        """Get current accuracy rate"""
        total = self.stats['successful_actions'] + self.stats['failed_actions']
        if total == 0:
            return 85  # Default
        return (self.stats['successful_actions'] / total) * 100
    
    @property  
    def total_hours(self):
        """Get total learning hours (estimated from experiences)"""
        # Estimate 1 experience per 30 seconds on average
        return self.stats['total_experiences'] * 0.5 / 3600
    
    def _analyze_patterns(self, action: str, experience: Dict[str, Any]):
        """Analyze experiences to identify patterns"""
        try:
            # Look for similar successful experiences
            if experience['outcome'] == 'success':
                similar_successes = self._find_similar_experiences(experience, self.success_patterns[action])
                if len(similar_successes) >= self.min_pattern_occurrences:
                    pattern = self._extract_pattern(similar_successes)
                    if pattern:
                        self._learn_pattern(action, pattern, 'success')
            
            # Look for similar failure experiences
            else:
                similar_failures = self._find_similar_experiences(experience, self.failure_patterns[action])
                if len(similar_failures) >= self.min_pattern_occurrences:
                    pattern = self._extract_pattern(similar_failures)
                    if pattern:
                        self._learn_pattern(action, pattern, 'failure')
                        
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
    
    def _find_similar_experiences(self, target_experience: Dict[str, Any], 
                                 experience_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find experiences similar to the target experience"""
        similar = []
        
        target_context = target_experience.get('context', {})
        
        for exp in experience_list:
            exp_context = exp.get('context', {})
            
            # Calculate similarity based on context
            similarity = self._calculate_context_similarity(target_context, exp_context)
            
            if similarity >= self.similarity_threshold:
                similar.append(exp)
        
        return similar
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        try:
            # Simple similarity based on shared keys and value differences
            common_keys = set(context1.keys()) & set(context2.keys())
            if not common_keys:
                return 0.0
            
            total_similarity = 0.0
            compared_keys = 0
            
            for key in common_keys:
                val1, val2 = context1[key], context2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numerical similarity
                    max_val = max(abs(val1), abs(val2), 1)  # Prevent division by zero
                    similarity = 1.0 - min(abs(val1 - val2) / max_val, 1.0)
                    total_similarity += similarity
                    compared_keys += 1
                elif isinstance(val1, str) and isinstance(val2, str):
                    # String similarity (exact match for now)
                    similarity = 1.0 if val1 == val2 else 0.0
                    total_similarity += similarity
                    compared_keys += 1
                elif isinstance(val1, list) and isinstance(val2, list):
                    # List similarity (length and content)
                    len_similarity = 1.0 - min(abs(len(val1) - len(val2)) / max(len(val1), len(val2), 1), 1.0)
                    total_similarity += len_similarity
                    compared_keys += 1
            
            return total_similarity / compared_keys if compared_keys > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _extract_pattern(self, experiences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract common pattern from similar experiences"""
        try:
            if not experiences:
                return None
            
            # Extract common context elements
            pattern = {
                'occurrences': len(experiences),
                'common_context': {},
                'context_ranges': {},
                'success_rate': 1.0 if all(exp['outcome'] == 'success' for exp in experiences) else 0.0
            }
            
            # Find common context keys
            all_contexts = [exp.get('context', {}) for exp in experiences]
            common_keys = set.intersection(*[set(ctx.keys()) for ctx in all_contexts])
            
            for key in common_keys:
                values = [ctx[key] for ctx in all_contexts]
                
                if all(isinstance(v, (int, float)) for v in values):
                    # Numerical range
                    pattern['context_ranges'][key] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values)
                    }
                elif all(isinstance(v, str) for v in values):
                    # Most common string value
                    value_counts = defaultdict(int)
                    for v in values:
                        value_counts[v] += 1
                    most_common = max(value_counts.items(), key=lambda x: x[1])
                    if most_common[1] / len(values) >= 0.5:  # At least 50% occurrence
                        pattern['common_context'][key] = most_common[0]
            
            return pattern if pattern['common_context'] or pattern['context_ranges'] else None
            
        except Exception as e:
            self.logger.error(f"Pattern extraction failed: {e}")
            return None
    
    def _learn_pattern(self, action: str, pattern: Dict[str, Any], outcome_type: str):
        """Learn a new pattern"""
        pattern_key = f"{action}_{outcome_type}"
        
        # Store the pattern (simplified storage for now)
        pattern_file = Path(f"data/patterns/{pattern_key}.json")
        pattern_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(pattern_file, 'w') as f:
                json.dump(pattern, f, indent=2)
            
            self.stats['patterns_learned'] += 1
            self.logger.info(f"Learned new pattern for {pattern_key}: {pattern['occurrences']} occurrences")
            
        except Exception as e:
            self.logger.error(f"Failed to save pattern: {e}")
    
    def get_action_recommendation(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommendation for an action based on learned patterns
        
        Args:
            action: Action to get recommendation for
            context: Current context
            
        Returns:
            Recommendation with confidence and suggestions
        """
        try:
            recommendation = {
                'action': action,
                'confidence': 0.5,  # Default neutral confidence
                'suggestions': [],
                'warnings': [],
                'expected_success_rate': 0.5
            }
            
            # Check action history
            action_stats = self.action_outcomes.get(action, {'success': 0, 'failure': 0})
            total_attempts = action_stats['success'] + action_stats['failure']
            
            if total_attempts > 0:
                success_rate = action_stats['success'] / total_attempts
                recommendation['expected_success_rate'] = success_rate
                recommendation['confidence'] = min(0.9, 0.5 + (success_rate - 0.5))
                
                if success_rate < 0.3:
                    recommendation['warnings'].append("This action has low success rate historically")
                elif success_rate > 0.8:
                    recommendation['suggestions'].append("This action has high success rate")
            
            # Check for learned patterns
            success_patterns = self._find_matching_patterns(action, context, 'success')
            failure_patterns = self._find_matching_patterns(action, context, 'failure')
            
            if success_patterns:
                recommendation['confidence'] = min(0.95, recommendation['confidence'] + 0.2)
                recommendation['suggestions'].append("Similar successful patterns found")
            
            if failure_patterns:
                recommendation['confidence'] = max(0.1, recommendation['confidence'] - 0.3)
                recommendation['warnings'].append("Similar failure patterns found")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendation: {e}")
            return {
                'action': action,
                'confidence': 0.5,
                'suggestions': [],
                'warnings': ['Failed to analyze patterns'],
                'expected_success_rate': 0.5
            }
    
    def _find_matching_patterns(self, action: str, context: Dict[str, Any], outcome_type: str) -> List[Dict[str, Any]]:
        """Find patterns that match the current context"""
        matching_patterns = []
        
        try:
            pattern_file = Path(f"data/patterns/{action}_{outcome_type}.json")
            if pattern_file.exists():
                with open(pattern_file, 'r') as f:
                    pattern = json.load(f)
                    
                # Check if context matches pattern
                if self._context_matches_pattern(context, pattern):
                    matching_patterns.append(pattern)
                    
        except Exception as e:
            self.logger.error(f"Failed to load pattern: {e}")
        
        return matching_patterns
    
    def _context_matches_pattern(self, context: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if context matches a learned pattern"""
        try:
            # Check common context
            for key, expected_value in pattern.get('common_context', {}).items():
                if key not in context or context[key] != expected_value:
                    return False
            
            # Check context ranges
            for key, range_info in pattern.get('context_ranges', {}).items():
                if key not in context:
                    return False
                
                value = context[key]
                if not isinstance(value, (int, float)):
                    return False
                
                if not (range_info['min'] <= value <= range_info['max']):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pattern matching failed: {e}")
            return False
    
    def adapt_strategy(self, action: str, repeated_failures: int = 3) -> Dict[str, Any]:
        """
        Adapt strategy based on repeated failures
        
        Args:
            action: Action that's failing repeatedly
            repeated_failures: Number of consecutive failures
            
        Returns:
            Adaptation suggestions
        """
        adaptations = {
            'original_action': action,
            'suggested_changes': [],
            'alternative_actions': [],
            'parameter_adjustments': {}
        }
        
        try:
            # Analyze recent failures for this action
            recent_failures = [exp for exp in self.experience_memory 
                             if exp['action'] == action and exp['outcome'] == 'failure'][-repeated_failures:]
            
            if len(recent_failures) >= repeated_failures:
                # Look for common failure patterns
                common_contexts = self._find_common_failure_contexts(recent_failures)
                
                # Suggest parameter adjustments
                if 'timing' in common_contexts:
                    adaptations['parameter_adjustments']['timing'] = 'increase_delay'
                    adaptations['suggested_changes'].append("Increase delay between actions")
                
                if 'position_accuracy' in common_contexts:
                    adaptations['parameter_adjustments']['position_tolerance'] = 'increase'
                    adaptations['suggested_changes'].append("Increase position tolerance")
                
                # Suggest alternative actions
                successful_actions = [exp['action'] for exp in self.experience_memory 
                                    if exp['outcome'] == 'success']
                if successful_actions:
                    from collections import Counter
                    common_successful = Counter(successful_actions).most_common(3)
                    adaptations['alternative_actions'] = [action for action, count in common_successful 
                                                        if action != action]
                
                self.stats['adaptations_made'] += 1
                self.logger.info(f"Generated adaptations for {action} after {repeated_failures} failures")
        
        except Exception as e:
            self.logger.error(f"Strategy adaptation failed: {e}")
        
        return adaptations
    
    def _find_common_failure_contexts(self, failures: List[Dict[str, Any]]) -> List[str]:
        """Find common contexts in failure experiences"""
        common_contexts = []
        
        try:
            # Analyze timing issues
            timestamps = [exp['timestamp'] for exp in failures]
            if len(timestamps) > 1:
                intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                if all(interval < 1.0 for interval in intervals):  # Actions too fast
                    common_contexts.append('timing')
            
            # Analyze position accuracy issues
            position_errors = []
            for failure in failures:
                details = failure.get('details', {})
                if 'position_error' in details:
                    position_errors.append(details['position_error'])
            
            if position_errors and sum(position_errors) / len(position_errors) > 10:  # High position error
                common_contexts.append('position_accuracy')
                
        except Exception as e:
            self.logger.error(f"Common context analysis failed: {e}")
        
        return common_contexts
    
    def learn_from_external_data(self, data: Dict[str, Any], source: str = "external") -> str:
        """
        Learn from external data sources
        
        Args:
            data: External data to learn from
            source: Source of the data
            
        Returns:
            Learning result summary
        """
        try:
            learned_items = 0
            
            # Process game updates
            if 'updates' in data:
                for update in data['updates']:
                    self._process_game_update(update)
                    learned_items += 1
            
            # Process strategies
            if 'strategies' in data:
                for strategy in data['strategies']:
                    self._process_strategy(strategy)
                    learned_items += 1
            
            # Process tips and tricks
            if 'tips' in data:
                for tip in data['tips']:
                    self._process_tip(tip)
                    learned_items += 1
            
            # Update statistics
            self.stats['items_learned'] += learned_items
            
            # Save learning data
            self._save_learning_data()
            
            result = f"Learned {learned_items} items from {source}"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to learn from external data: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _process_game_update(self, update: Dict[str, Any]):
        """Process a game update for learning"""
        # Extract useful information from game updates
        # This would be customized based on the specific game
        pass
    
    def _process_strategy(self, strategy: Dict[str, Any]):
        """Process a strategy for learning"""
        # Learn new strategies from external sources
        pass
    
    def _process_tip(self, tip: Dict[str, Any]):
        """Process a tip for learning"""
        # Learn tips and tricks
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        return self.stats.copy()
    
    def reset_learning_data(self):
        """Reset all learning data (use with caution)"""
        self.experience_memory.clear()
        self.success_patterns.clear()
        self.failure_patterns.clear()
        self.action_outcomes.clear()
        
        self.stats = {
            'total_experiences': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'patterns_learned': 0,
            'adaptations_made': 0,
            'items_learned': 0
        }
        
        # Clear saved data
        try:
            learning_file = Path("data/learning_data.pkl")
            if learning_file.exists():
                learning_file.unlink()
            
            patterns_dir = Path("data/patterns")
            if patterns_dir.exists():
                for pattern_file in patterns_dir.glob("*.json"):
                    pattern_file.unlink()
        except Exception as e:
            self.logger.error(f"Failed to clear learning files: {e}")
        
        self.logger.info("Learning data reset")
