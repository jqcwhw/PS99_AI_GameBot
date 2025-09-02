"""
Autonomous Learning System for AI Game Bot
Provides continuous self-learning while maintaining core game automation purpose
"""

import numpy as np
import logging
import time
import threading
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import deque, defaultdict
import random
import psycopg2
import os

class AutonomousLearningSystem:
    """Self-learning system that continuously improves while maintaining purpose"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core learning state
        self.is_learning = False
        self.learning_thread = None
        
        # Memory systems
        self.experience_buffer = deque(maxlen=5000)
        self.pattern_library = {}
        self.strategy_book = {}
        
        # Goal-oriented learning - CORE PURPOSE PROTECTION
        self.core_purposes = {
            "open_chests": {"priority": 1.0, "performance": 0.7, "target": 0.95},
            "hatch_eggs": {"priority": 0.9, "performance": 0.6, "target": 0.9}, 
            "stay_in_breakables": {"priority": 0.8, "performance": 0.7, "target": 0.9},
            "macro_adaptation": {"priority": 0.7, "performance": 0.5, "target": 0.85}
        }
        
        # Q-Learning for action optimization
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.exploration_rate = 0.2
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        
        # Unsupervised learning tracking
        self.patterns_discovered = 0
        self.autonomous_improvements = 0
        self.idle_learning_cycles = 0
        
        # Database connection for repository knowledge
        self.db_connection = None
        self._connect_to_database()
        
        # Learning intervals
        self.idle_learning_interval = 3.0  # Learn every 3 seconds when idle
        self.pattern_analysis_interval = 60.0  # Analyze patterns every minute
        self.knowledge_integration_interval = 300.0  # Integrate new knowledge every 5 minutes
        
        # Load previous learning
        self._load_learning_state()
        
        self.logger.info("Autonomous Learning System initialized - AI will self-improve continuously")
    
    def start_autonomous_learning(self):
        """Start continuous autonomous learning"""
        if self.learning_thread and self.learning_thread.is_alive():
            return
        
        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._autonomous_learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        self.logger.info("🧠 Autonomous learning started - AI will now continuously self-improve")
    
    def stop_autonomous_learning(self):
        """Stop autonomous learning"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=3.0)
        self._save_learning_state()
        self.logger.info("Autonomous learning stopped")
    
    def _autonomous_learning_loop(self):
        """Main autonomous learning loop"""
        last_pattern_analysis = time.time()
        last_knowledge_integration = time.time()
        
        while self.is_learning:
            try:
                current_time = time.time()
                self.idle_learning_cycles += 1
                
                # Continuous learning activities
                self._unsupervised_pattern_discovery()
                self._optimize_existing_strategies()
                self._maintain_core_purpose_alignment()
                
                # Periodic deep learning
                if current_time - last_pattern_analysis > self.pattern_analysis_interval:
                    self._deep_pattern_analysis()
                    last_pattern_analysis = current_time
                
                # Repository knowledge integration
                if current_time - last_knowledge_integration > self.knowledge_integration_interval:
                    self._integrate_repository_knowledge()
                    last_knowledge_integration = current_time
                
                # Self-reflection to ensure purpose alignment
                self._perform_purpose_alignment_check()
                
                time.sleep(self.idle_learning_interval)
                
            except Exception as e:
                self.logger.error(f"Autonomous learning error: {e}")
                time.sleep(10.0)
    
    def record_game_experience(self, action: str, context: Dict[str, Any], outcome: str, reward: float = None):
        """Record game experience for learning"""
        # Calculate reward if not provided
        if reward is None:
            reward = self._calculate_smart_reward(action, context, outcome)
        
        experience = {
            'action': action,
            'context': context,
            'outcome': outcome,
            'reward': reward,
            'timestamp': time.time(),
            'game_state_hash': self._hash_game_state(context)
        }
        
        self.experience_buffer.append(experience)
        
        # Update Q-learning
        self._update_q_learning(experience)
        
        # Immediate pattern detection
        self._detect_real_time_patterns(experience)
        
        self.logger.debug(f"🎮 Recorded experience: {action} -> {outcome} (reward: {reward:.2f})")
    
    def choose_intelligent_action(self, available_actions: List[str], context: Dict[str, Any]) -> str:
        """Choose action using learned intelligence"""
        state_key = self._context_to_state_key(context)
        
        # Exploration vs exploitation with smart decay
        if random.random() < self.exploration_rate:
            action = random.choice(available_actions)
            self.logger.debug(f"🔍 Exploring: chose {action}")
        else:
            # Choose action with highest Q-value
            q_scores = {action: self.q_values[state_key][action] for action in available_actions}
            action = max(q_scores, key=q_scores.get) if any(q_scores.values()) else random.choice(available_actions)
            self.logger.debug(f"🎯 Exploiting: chose {action} (Q={q_scores.get(action, 0):.2f})")
        
        # Decay exploration over time for more focused learning
        self.exploration_rate = max(0.05, self.exploration_rate * 0.9995)
        
        return action
    
    def _unsupervised_pattern_discovery(self):
        """Discover patterns without supervision"""
        if len(self.experience_buffer) < 50:
            return
        
        try:
            recent_experiences = list(self.experience_buffer)[-100:]
            
            # Find action sequences that lead to success
            successful_sequences = self._find_successful_action_sequences(recent_experiences)
            
            # Discover environmental patterns
            environmental_patterns = self._discover_environmental_patterns(recent_experiences)
            
            # Learn temporal patterns
            timing_patterns = self._learn_timing_patterns(recent_experiences)
            
            # Store discovered patterns
            new_patterns = len(successful_sequences) + len(environmental_patterns) + len(timing_patterns)
            if new_patterns > 0:
                self._store_patterns({
                    'action_sequences': successful_sequences,
                    'environmental': environmental_patterns,
                    'timing': timing_patterns
                })
                
                self.patterns_discovered += new_patterns
                self.logger.debug(f"🔬 Discovered {new_patterns} new patterns autonomously")
        
        except Exception as e:
            self.logger.error(f"Pattern discovery error: {e}")
    
    def _optimize_existing_strategies(self):
        """Optimize existing strategies based on performance"""
        try:
            for purpose, data in self.core_purposes.items():
                current_performance = self._measure_performance(purpose)
                
                if current_performance > data["performance"]:
                    # Performance improved - reinforce successful strategies
                    self._reinforce_successful_strategies(purpose)
                    data["performance"] = current_performance
                    self.autonomous_improvements += 1
                    self.logger.debug(f"📈 Improved {purpose}: {current_performance:.2f}")
                
                elif current_performance < data["performance"] * 0.9:
                    # Performance degraded - try new approaches
                    self._generate_alternative_strategies(purpose)
                    self.logger.debug(f"🔄 Adapting strategy for {purpose}")
        
        except Exception as e:
            self.logger.error(f"Strategy optimization error: {e}")
    
    def _maintain_core_purpose_alignment(self):
        """Ensure AI maintains its core game automation purpose"""
        # Core purpose protection - these must always be prioritized
        core_actions = ["open_chests", "hatch_eggs", "stay_in_breakables", "farm", "macro"]
        
        # Check if recent actions align with core purposes
        if len(self.experience_buffer) > 20:
            recent_actions = [exp['action'] for exp in list(self.experience_buffer)[-20:]]
            core_action_ratio = sum(1 for action in recent_actions if any(core in action.lower() for core in core_actions)) / len(recent_actions)
            
            if core_action_ratio < 0.6:  # Less than 60% core actions
                self.logger.warning("⚠️ AI deviating from core purpose - realigning priorities")
                self._realign_to_core_purpose()
    
    def _integrate_repository_knowledge(self):
        """Integrate knowledge from analyzed repositories"""
        if not self.db_connection:
            return
        
        try:
            cursor = self.db_connection.cursor()
            
            # Get relevant automation patterns
            cursor.execute("""
                SELECT pattern_type, pattern_name, code_snippet, relevance_score
                FROM code_patterns 
                WHERE pattern_type IN ('computer_vision', 'input_automation', 'macro_system')
                AND relevance_score > 0.5
                ORDER BY relevance_score DESC
                LIMIT 5
            """)
            
            patterns = cursor.fetchall()
            
            for pattern_type, pattern_name, code_snippet, relevance in patterns:
                # Adapt repository knowledge to current goals
                adapted_technique = self._adapt_repository_pattern(pattern_type, pattern_name, code_snippet)
                
                if adapted_technique:
                    self._integrate_technique(adapted_technique)
                    self.autonomous_improvements += 1
                    self.logger.debug(f"🧠 Integrated {pattern_name} from repository knowledge")
            
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Repository integration error: {e}")
    
    def _perform_purpose_alignment_check(self):
        """Regular check to ensure AI stays aligned with its purpose"""
        # Verify core purposes maintain high priority
        for purpose, data in self.core_purposes.items():
            if data["priority"] < 0.5:
                data["priority"] = max(0.7, data["priority"] * 1.2)  # Boost priority
                self.logger.debug(f"🎯 Boosted priority for core purpose: {purpose}")
        
        # Check if learning is producing beneficial results
        if self.autonomous_improvements > 0 and self.idle_learning_cycles > 100:
            improvement_rate = self.autonomous_improvements / self.idle_learning_cycles
            if improvement_rate < 0.01:  # Less than 1% improvement rate
                self.logger.info("🔧 Adjusting learning parameters for better improvement rate")
                self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
    
    def get_autonomous_learning_status(self) -> Dict[str, Any]:
        """Get status of autonomous learning"""
        return {
            "autonomous_learning_active": self.is_learning,
            "idle_learning_cycles": self.idle_learning_cycles,
            "patterns_discovered": self.patterns_discovered,
            "autonomous_improvements": self.autonomous_improvements,
            "experience_buffer_size": len(self.experience_buffer),
            "core_purposes": {
                name: {
                    "priority": data["priority"],
                    "performance": data["performance"],
                    "target": data["target"],
                    "progress": data["performance"] / data["target"]
                }
                for name, data in self.core_purposes.items()
            },
            "exploration_rate": self.exploration_rate,
            "patterns_in_library": len(self.pattern_library),
            "strategies_learned": len(self.strategy_book),
            "learning_effectiveness": self._calculate_learning_effectiveness()
        }
    
    # Helper methods for learning algorithms
    def _calculate_smart_reward(self, action: str, context: Dict[str, Any], outcome: str) -> float:
        """Calculate intelligent reward based on action, context and outcome"""
        base_reward = 0.0
        
        # Outcome-based rewards
        if "success" in outcome.lower() or "completed" in outcome.lower() or "found" in outcome.lower():
            base_reward = 1.0
        elif "partial" in outcome.lower() or "progress" in outcome.lower():
            base_reward = 0.5
        elif "failed" in outcome.lower() or "error" in outcome.lower():
            base_reward = -0.3
        
        # Core purpose alignment bonus
        core_actions = ["chest", "egg", "breakable", "farm", "macro"]
        if any(core in action.lower() for core in core_actions):
            base_reward += 0.2  # Bonus for core actions
        
        # Efficiency bonus
        if context.get("time_taken", float('inf')) < context.get("expected_time", float('inf')):
            base_reward += 0.1
        
        # Innovation bonus for trying new approaches
        if context.get("novel_approach", False):
            base_reward += 0.05
        
        return base_reward
    
    def _hash_game_state(self, context: Dict[str, Any]) -> str:
        """Create hash of game state for pattern recognition"""
        key_elements = [
            context.get("ui_elements_count", 0),
            context.get("screen_region", "unknown"),
            context.get("player_stats", {}).get("level", 0),
            len(context.get("available_actions", []))
        ]
        return str(hash(tuple(key_elements)))
    
    def _context_to_state_key(self, context: Dict[str, Any]) -> str:
        """Convert context to state key for Q-learning"""
        return f"ui_{len(context.get('ui_elements', {}))}_region_{context.get('screen_region', 'unknown')}"
    
    def _update_q_learning(self, experience: Dict[str, Any]):
        """Update Q-learning values"""
        state_key = experience['game_state_hash']
        action = experience['action']
        reward = experience['reward']
        
        # Simple Q-learning update
        current_q = self.q_values[state_key][action]
        self.q_values[state_key][action] = current_q + self.learning_rate * (reward - current_q)
    
    def _detect_real_time_patterns(self, experience: Dict[str, Any]):
        """Detect patterns in real-time"""
        if len(self.experience_buffer) < 5:
            return
        
        # Look for immediate action-outcome patterns
        recent = list(self.experience_buffer)[-5:]
        if all(exp['reward'] > 0.5 for exp in recent):
            pattern_key = f"successful_sequence_{experience['action']}"
            if pattern_key not in self.pattern_library:
                self.pattern_library[pattern_key] = {
                    'type': 'action_sequence',
                    'effectiveness': 0.8,
                    'discovered_at': time.time()
                }
                self.patterns_discovered += 1
    
    def _connect_to_database(self):
        """Connect to repository knowledge database"""
        try:
            self.db_connection = psycopg2.connect(
                host=os.getenv('PGHOST'),
                port=os.getenv('PGPORT'),
                database=os.getenv('PGDATABASE'),
                user=os.getenv('PGUSER'),
                password=os.getenv('PGPASSWORD')
            )
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
    
    def _save_learning_state(self):
        """Save learning state"""
        try:
            state = {
                'core_purposes': self.core_purposes,
                'patterns_discovered': self.patterns_discovered,
                'autonomous_improvements': self.autonomous_improvements,
                'pattern_library': self.pattern_library,
                'strategy_book': self.strategy_book,
                'exploration_rate': self.exploration_rate
            }
            
            learning_dir = Path("data/autonomous_learning")
            learning_dir.mkdir(exist_ok=True)
            
            with open(learning_dir / "learning_state.json", 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save Q-values separately
            with open(learning_dir / "q_values.json", 'w') as f:
                json.dump(dict(self.q_values), f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save learning state: {e}")
    
    def _load_learning_state(self):
        """Load previous learning state"""
        try:
            learning_dir = Path("data/autonomous_learning")
            
            state_file = learning_dir / "learning_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.core_purposes = state.get('core_purposes', self.core_purposes)
                self.patterns_discovered = state.get('patterns_discovered', 0)
                self.autonomous_improvements = state.get('autonomous_improvements', 0)
                self.pattern_library = state.get('pattern_library', {})
                self.strategy_book = state.get('strategy_book', {})
                self.exploration_rate = state.get('exploration_rate', 0.2)
            
            # Load Q-values
            q_file = learning_dir / "q_values.json"
            if q_file.exists():
                with open(q_file, 'r') as f:
                    q_data = json.load(f)
                    for state_key, actions in q_data.items():
                        for action, value in actions.items():
                            self.q_values[state_key][action] = value
            
            self.logger.info("Loaded previous autonomous learning state")
            
        except Exception as e:
            self.logger.error(f"Failed to load learning state: {e}")
    
    # Simplified implementations of helper methods
    def _find_successful_action_sequences(self, experiences: List[Dict]) -> List[Dict]:
        """Find action sequences that lead to success"""
        sequences = []
        for i in range(len(experiences) - 2):
            if all(exp['reward'] > 0.5 for exp in experiences[i:i+3]):
                sequences.append({
                    'actions': [exp['action'] for exp in experiences[i:i+3]],
                    'effectiveness': sum(exp['reward'] for exp in experiences[i:i+3]) / 3
                })
        return sequences
    
    def _discover_environmental_patterns(self, experiences: List[Dict]) -> List[Dict]:
        """Discover environmental patterns"""
        patterns = []
        context_groups = defaultdict(list)
        
        for exp in experiences:
            context_key = exp['context'].get('screen_region', 'unknown')
            context_groups[context_key].append(exp)
        
        for context, exps in context_groups.items():
            if len(exps) > 3:
                avg_reward = sum(exp['reward'] for exp in exps) / len(exps)
                if avg_reward > 0.3:
                    patterns.append({
                        'context': context,
                        'avg_effectiveness': avg_reward,
                        'frequency': len(exps)
                    })
        
        return patterns
    
    def _learn_timing_patterns(self, experiences: List[Dict]) -> List[Dict]:
        """Learn timing patterns"""
        patterns = []
        
        # Simple timing analysis
        for i in range(len(experiences) - 1):
            time_diff = experiences[i+1]['timestamp'] - experiences[i]['timestamp']
            if 1.0 < time_diff < 5.0 and experiences[i+1]['reward'] > 0.7:
                patterns.append({
                    'optimal_delay': time_diff,
                    'action_pair': (experiences[i]['action'], experiences[i+1]['action']),
                    'effectiveness': experiences[i+1]['reward']
                })
        
        return patterns
    
    def _store_patterns(self, pattern_groups: Dict[str, List]):
        """Store discovered patterns"""
        timestamp = time.time()
        for pattern_type, patterns in pattern_groups.items():
            for i, pattern in enumerate(patterns):
                key = f"{pattern_type}_{timestamp}_{i}"
                self.pattern_library[key] = {
                    'type': pattern_type,
                    'data': pattern,
                    'discovered_at': timestamp
                }
    
    def _measure_performance(self, purpose: str) -> float:
        """Measure current performance for a purpose"""
        if len(self.experience_buffer) < 10:
            return 0.5
        
        # Simple performance measurement based on recent experiences
        recent_experiences = [exp for exp in list(self.experience_buffer)[-20:] 
                            if purpose.replace('_', ' ') in exp['action'].lower()]
        
        if not recent_experiences:
            return 0.5
        
        avg_reward = sum(exp['reward'] for exp in recent_experiences) / len(recent_experiences)
        return max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))  # Normalize to 0-1
    
    def _calculate_learning_effectiveness(self) -> float:
        """Calculate overall learning effectiveness"""
        if self.idle_learning_cycles == 0:
            return 0.0
        
        effectiveness = (self.autonomous_improvements + self.patterns_discovered) / max(self.idle_learning_cycles, 1)
        return min(1.0, effectiveness * 10)  # Scale and cap at 1.0
    
    # Placeholder implementations for remaining methods
    def _reinforce_successful_strategies(self, purpose: str):
        """Reinforce strategies that are working well"""
        pass
    
    def _generate_alternative_strategies(self, purpose: str):
        """Generate new strategies when current ones aren't working"""
        pass
    
    def _realign_to_core_purpose(self):
        """Realign AI behavior to core purposes"""
        for purpose in self.core_purposes:
            self.core_purposes[purpose]["priority"] = min(1.0, self.core_purposes[purpose]["priority"] * 1.2)
    
    def _adapt_repository_pattern(self, pattern_type: str, pattern_name: str, code_snippet: str) -> Optional[Dict]:
        """Adapt repository patterns to current needs"""
        return {
            'name': f"adapted_{pattern_name}",
            'type': pattern_type,
            'source': 'repository',
            'effectiveness': 0.6
        }
    
    def _integrate_technique(self, technique: Dict):
        """Integrate a new technique into strategy book"""
        self.strategy_book[technique['name']] = technique
    
    def _deep_pattern_analysis(self):
        """Perform deep analysis of collected patterns"""
        if self.patterns_discovered > 0:
            self.logger.debug(f"🔍 Deep analysis: {self.patterns_discovered} patterns analyzed")
    
    def __del__(self):
        """Cleanup when destroyed"""
        try:
            self.stop_autonomous_learning()
            if self.db_connection:
                self.db_connection.close()
        except:
            pass