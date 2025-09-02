"""
Enhanced Integration System
Combines SerpentAI, D3DShot, and Reinforcement Learning optimizations
for maximum AI game automation performance
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np
import json
from collections import deque
import random

# Import our enhanced core systems
from .serpent_enhanced_vision import SerpentEnhancedVision
from .advanced_reinforcement_learning import AdvancedRLAgent
from .vision_system import VisionSystem
from .automation_engine import AutomationEngine
from .learning_system import LearningSystem

class EnhancedIntegrationSystem:
    """
    Master integration system combining all AI enhancements
    Orchestrates vision, automation, and learning with SerpentAI optimizations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize enhanced systems
        self.serpent_vision = SerpentEnhancedVision(
            width=self.config['vision']['width'],
            height=self.config['vision']['height'],
            fps=self.config['vision']['fps'],
            pipeline_string=self.config['vision']['pipeline']
        )
        
        self.vision_system = VisionSystem()
        self.automation_engine = AutomationEngine()
        self.learning_system = LearningSystem()
        
        # Initialize reinforcement learning agent if enabled
        self.rl_agent = None
        if self.config['rl']['enabled']:
            try:
                self.rl_agent = AdvancedRLAgent(
                    algorithm=self.config['rl']['algorithm'],
                    input_shape=tuple(self.config['rl']['input_shape']),
                    num_actions=self.config['rl']['num_actions'],
                    continuous=self.config['rl']['continuous']
                )
                self.logger.info(f"RL Agent initialized: {self.config['rl']['algorithm']}")
            except Exception as e:
                self.logger.warning(f"RL Agent initialization failed: {e}")
        
        # System state
        self.is_active = False
        self.performance_mode = "balanced"  # balanced, precision, speed
        self.current_game_state = {}
        
        # Enhanced coordination
        self.action_coordinator = ActionCoordinator(self)
        self.pattern_analyzer = PatternAnalyzer(self)
        self.performance_optimizer = PerformanceOptimizer(self)
        
        # Integration metrics
        self.integration_stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'ai_decisions': 0,
            'pattern_matches': 0,
            'optimization_cycles': 0
        }
        
        self.logger.info("Enhanced Integration System initialized with SerpentAI optimizations")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the integration system"""
        return {
            'vision': {
                'width': 1024,
                'height': 768,
                'fps': 30,
                'pipeline': "RESIZE:640x480|GRAYSCALE|NORMALIZE"
            },
            'rl': {
                'enabled': True,
                'algorithm': 'PPO',  # PPO, DQN, DDPG, TD3
                'input_shape': [4, 84, 84],  # 4-frame stack, 84x84 resolution
                'num_actions': 6,  # move_up, move_down, move_left, move_right, click, wait
                'continuous': False
            },
            'automation': {
                'serpent_mode': True,
                'adaptive_timing': True,
                'human_like_movement': True
            },
            'learning': {
                'q_learning': True,
                'pattern_analysis': True,
                'adaptive_rate': True
            }
        }
    
    def start_enhanced_mode(self):
        """Start the enhanced integration mode"""
        if self.is_active:
            self.logger.warning("Enhanced mode already active")
            return
        
        self.is_active = True
        
        # Enable SerpentAI mode in automation
        if self.config['automation']['serpent_mode']:
            self.automation_engine.enable_serpent_mode()
        
        # Start high-performance vision capture
        self.serpent_vision.start_capture()
        
        # Start automation engine
        self.automation_engine.start()
        
        # Start AI coordination thread
        self.coordination_thread = threading.Thread(target=self._ai_coordination_loop, daemon=True)
        self.coordination_thread.start()
        
        self.logger.info("Enhanced integration mode started")
    
    def stop_enhanced_mode(self):
        """Stop the enhanced integration mode"""
        if not self.is_active:
            return
        
        self.is_active = False
        
        # Stop all systems
        self.serpent_vision.stop_capture()
        self.automation_engine.stop()
        
        # Wait for coordination thread to finish
        if hasattr(self, 'coordination_thread') and self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=5.0)
        
        self.logger.info("Enhanced integration mode stopped")
    
    def _ai_coordination_loop(self):
        """Main AI coordination loop combining all systems"""
        last_optimization = time.time()
        action_sequence = deque(maxlen=20)
        
        while self.is_active:
            try:
                loop_start = time.time()
                
                # Get current frame from enhanced vision
                current_frame = self.serpent_vision.get_latest_frame()
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                
                # Analyze game state
                game_state = self._analyze_game_state(current_frame)
                
                # AI decision making
                if self.rl_agent and self.config['rl']['enabled']:
                    action = self._make_rl_decision(game_state, current_frame)
                else:
                    action = self._make_heuristic_decision(game_state)
                
                # Execute action if determined
                if action:
                    success = self._execute_enhanced_action(action, game_state)
                    
                    # Record for learning
                    self._record_action_outcome(action, game_state, success)
                    action_sequence.append(action['type'])
                    
                    # Update integration stats
                    self.integration_stats['total_actions'] += 1
                    if success:
                        self.integration_stats['successful_actions'] += 1
                
                # Analyze patterns periodically
                if len(action_sequence) >= 5:
                    self.pattern_analyzer.analyze_sequence(list(action_sequence))
                    self.learning_system.analyze_temporal_patterns(list(action_sequence))
                
                # Performance optimization every 30 seconds
                current_time = time.time()
                if current_time - last_optimization > 30:
                    self.performance_optimizer.optimize_systems()
                    last_optimization = current_time
                    self.integration_stats['optimization_cycles'] += 1
                
                # Adaptive learning rate adjustment
                if self.config['learning']['adaptive_rate']:
                    self.learning_system.adaptive_learning_rate()
                
                # Control loop timing
                loop_duration = time.time() - loop_start
                sleep_time = max(0.1, 1.0/30 - loop_duration)  # 30 FPS target
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in AI coordination loop: {e}")
                time.sleep(1.0)
    
    def _analyze_game_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze current game state using enhanced vision"""
        game_state = {
            'timestamp': time.time(),
            'frame_shape': frame.shape,
            'detected_elements': {},
            'regions': {}
        }
        
        try:
            # Use both vision systems for comprehensive analysis
            
            # SerpentAI enhanced detection
            sprite_detections = self.serpent_vision.detect_sprites(
                frame, self.serpent_vision.sprite_templates
            )
            
            # Color-based detection for game elements
            color_ranges = {
                'chests': ((20, 100, 100), (30, 255, 255)),  # Golden chests
                'eggs': ((100, 150, 50), (120, 255, 255)),   # Blue eggs
                'breakables': ((0, 100, 100), (10, 255, 255))  # Red breakables
            }
            
            color_detections = self.serpent_vision.detect_color_regions(
                frame, color_ranges, min_area=50
            )
            
            # Traditional template matching
            template_matches = {}
            for element_type in ['chests', 'eggs', 'breakables']:
                matches = self.vision_system.find_template(element_type, frame)
                if matches:
                    template_matches[element_type] = matches
            
            # Combine detections
            game_state['detected_elements'] = {
                'sprites': sprite_detections,
                'colors': color_detections,
                'templates': template_matches
            }
            
            # Analyze specific regions
            for region_name in self.serpent_vision.detection_regions:
                region_frame = self.serpent_vision.analyze_region(frame, region_name)
                if region_frame is not None:
                    game_state['regions'][region_name] = {
                        'analyzed': True,
                        'shape': region_frame.shape
                    }
            
        except Exception as e:
            self.logger.error(f"Game state analysis failed: {e}")
        
        self.current_game_state = game_state
        return game_state
    
    def _make_rl_decision(self, game_state: Dict[str, Any], frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Make decision using reinforcement learning agent"""
        if not self.rl_agent:
            return None
        
        try:
            # Prepare observation for RL agent
            observation = self._prepare_rl_observation(frame, game_state)
            
            # Select action
            action_index = self.rl_agent.select_action(observation, training=True)
            
            # Convert action index to game action
            action = self._convert_rl_action(action_index, game_state)
            
            self.integration_stats['ai_decisions'] += 1
            return action
            
        except Exception as e:
            self.logger.error(f"RL decision making failed: {e}")
            return None
    
    def _make_heuristic_decision(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make decision using heuristic rules and learning patterns"""
        detected = game_state.get('detected_elements', {})
        
        # Priority: chests > eggs > breakables
        
        # Check for chests
        chest_positions = []
        for detection_type in ['sprites', 'colors', 'templates']:
            if 'chests' in detected.get(detection_type, {}):
                chest_data = detected[detection_type]['chests']
                if detection_type == 'colors' and chest_data:
                    chest_positions.extend([(item['center'][0], item['center'][1]) for item in chest_data])
                elif detection_type in ['sprites', 'templates'] and chest_data:
                    chest_positions.extend(chest_data)
        
        if chest_positions:
            # Use learning system to select best chest
            best_chest = self._select_best_target('chest', chest_positions)
            return {
                'type': 'open_chest',
                'target': best_chest,
                'priority': 'high'
            }
        
        # Check for eggs
        egg_positions = []
        for detection_type in ['sprites', 'colors', 'templates']:
            if 'eggs' in detected.get(detection_type, {}):
                egg_data = detected[detection_type]['eggs']
                if detection_type == 'colors' and egg_data:
                    egg_positions.extend([(item['center'][0], item['center'][1]) for item in egg_data])
                elif detection_type in ['sprites', 'templates'] and egg_data:
                    egg_positions.extend(egg_data)
        
        if egg_positions:
            best_egg = self._select_best_target('egg', egg_positions)
            return {
                'type': 'hatch_egg',
                'target': best_egg,
                'priority': 'medium'
            }
        
        # Check for breakables
        breakable_positions = []
        for detection_type in ['sprites', 'colors', 'templates']:
            if 'breakables' in detected.get(detection_type, {}):
                breakable_data = detected[detection_type]['breakables']
                if detection_type == 'colors' and breakable_data:
                    breakable_positions.extend([(item['center'][0], item['center'][1]) for item in breakable_data])
                elif detection_type in ['sprites', 'templates'] and breakable_data:
                    breakable_positions.extend(breakable_data)
        
        if breakable_positions:
            best_breakable = self._select_best_target('breakable', breakable_positions)
            return {
                'type': 'break_object',
                'target': best_breakable,
                'priority': 'low'
            }
        
        # No targets found - explore or wait
        return {
            'type': 'explore',
            'target': None,
            'priority': 'low'
        }
    
    def _select_best_target(self, target_type: str, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Select best target using learning system insights"""
        if not positions:
            return (0, 0)
        
        # Use Q-learning if available
        if self.config['learning']['q_learning']:
            state = f"{target_type}_selection"
            available_actions = [f"target_{i}" for i in range(len(positions))]
            
            if len(available_actions) > 0:
                best_action = self.learning_system.select_action_epsilon_greedy(state, available_actions)
                try:
                    action_index = int(best_action.split('_')[1])
                    if 0 <= action_index < len(positions):
                        return positions[action_index]
                except:
                    pass
        
        # Fallback: select closest target
        screen_center = (self.config['vision']['width'] // 2, self.config['vision']['height'] // 2)
        closest_position = min(positions, key=lambda pos: 
            ((pos[0] - screen_center[0]) ** 2 + (pos[1] - screen_center[1]) ** 2) ** 0.5
        )
        
        return closest_position
    
    def _execute_enhanced_action(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Execute action using enhanced automation engine"""
        try:
            action_type = action['type']
            target = action.get('target')
            
            if action_type == 'open_chest' and target:
                result = self.automation_engine.open_chests([target])
                success = "Queued" in result
                
            elif action_type == 'hatch_egg' and target:
                result = self.automation_engine.hatch_eggs([target])
                success = "Queued" in result
                
            elif action_type == 'break_object' and target:
                # Click on breakable object
                self.automation_engine.queue_click(target[0], target[1])
                success = True
                
            elif action_type == 'explore':
                # Move to random position for exploration
                x = random.randint(100, self.config['vision']['width'] - 100)
                y = random.randint(100, self.config['vision']['height'] - 100)
                self.automation_engine.queue_move(x, y)
                success = True
                
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return False
    
    def _record_action_outcome(self, action: Dict[str, Any], game_state: Dict[str, Any], success: bool):
        """Record action outcome for learning systems"""
        # Record in learning system
        context = {
            'game_state': game_state,
            'action_details': action
        }
        
        outcome = 'success' if success else 'failure'
        self.learning_system.record_experience(
            action['type'], context, outcome
        )
        
        # Q-learning update if enabled
        if self.config['learning']['q_learning'] and action.get('target'):
            state = f"{action['type']}_execution"
            action_key = f"execute_{action['type']}"
            reward = 1.0 if success else -0.5
            next_state = "completed" if success else "failed"
            
            self.learning_system.q_learning_update(state, action_key, reward, next_state)
        
        # Train RL agent if available
        if self.rl_agent and hasattr(self, '_last_observation'):
            try:
                reward = 1.0 if success else -0.1
                current_observation = self._prepare_rl_observation(
                    self.serpent_vision.get_latest_frame(), game_state
                )
                
                if current_observation is not None:
                    self.rl_agent.train_step(
                        self._last_observation, self._last_action,
                        reward, current_observation, not success
                    )
            except Exception as e:
                self.logger.error(f"RL training failed: {e}")
    
    def _prepare_rl_observation(self, frame: np.ndarray, game_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare observation for RL agent"""
        try:
            # Resize frame to RL input size
            target_size = (self.config['rl']['input_shape'][1], self.config['rl']['input_shape'][2])
            
            import cv2
            resized_frame = cv2.resize(frame, target_size)
            
            # Convert to grayscale if needed
            if len(resized_frame.shape) == 3:
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Normalize
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            
            # Get frame stack
            frame_stack = self.serpent_vision.get_frame_stack(
                list(range(-self.config['rl']['input_shape'][0], 0)),
                stack_dimension="first"
            )
            
            if frame_stack is not None:
                # Resize and normalize frame stack
                processed_stack = []
                for i in range(frame_stack.shape[0]):
                    frame_slice = cv2.resize(frame_stack[i], target_size)
                    if len(frame_slice.shape) == 3:
                        frame_slice = cv2.cvtColor(frame_slice, cv2.COLOR_BGR2GRAY)
                    processed_stack.append(frame_slice.astype(np.float32) / 255.0)
                
                return np.stack(processed_stack, axis=0)
            else:
                # Use current frame repeated
                return np.stack([normalized_frame] * self.config['rl']['input_shape'][0], axis=0)
                
        except Exception as e:
            self.logger.error(f"RL observation preparation failed: {e}")
            return None
    
    def _convert_rl_action(self, action_index: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert RL action index to game action"""
        action_map = {
            0: {'type': 'move_up'},
            1: {'type': 'move_down'},
            2: {'type': 'move_left'},
            3: {'type': 'move_right'},
            4: {'type': 'click'},
            5: {'type': 'wait'}
        }
        
        base_action = action_map.get(action_index, {'type': 'wait'})
        
        # Add target information based on current game state
        if base_action['type'] in ['move_up', 'move_down', 'move_left', 'move_right']:
            # Convert to movement coordinates
            center_x = self.config['vision']['width'] // 2
            center_y = self.config['vision']['height'] // 2
            move_distance = 50
            
            if base_action['type'] == 'move_up':
                base_action['target'] = (center_x, center_y - move_distance)
            elif base_action['type'] == 'move_down':
                base_action['target'] = (center_x, center_y + move_distance)
            elif base_action['type'] == 'move_left':
                base_action['target'] = (center_x - move_distance, center_y)
            elif base_action['type'] == 'move_right':
                base_action['target'] = (center_x + move_distance, center_y)
            
            base_action['type'] = 'move'
        
        elif base_action['type'] == 'click':
            # Find best click target from detected elements
            detected = game_state.get('detected_elements', {})
            click_targets = []
            
            for detection_type in ['sprites', 'colors', 'templates']:
                for element_type in ['chests', 'eggs', 'breakables']:
                    if element_type in detected.get(detection_type, {}):
                        element_data = detected[detection_type][element_type]
                        if detection_type == 'colors' and element_data:
                            click_targets.extend([(item['center'][0], item['center'][1]) for item in element_data])
                        elif detection_type in ['sprites', 'templates'] and element_data:
                            click_targets.extend(element_data)
            
            if click_targets:
                base_action['target'] = random.choice(click_targets)
                base_action['type'] = 'click_target'
            else:
                base_action['type'] = 'wait'
        
        return base_action
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        stats = self.integration_stats.copy()
        
        # Add system-specific stats
        stats['vision'] = self.serpent_vision.get_capture_stats()
        stats['automation'] = self.automation_engine.get_performance_stats()
        stats['learning'] = self.learning_system.get_advanced_stats()
        
        if self.rl_agent:
            stats['rl'] = self.rl_agent.get_training_stats()
        
        # Calculate success rate
        if stats['total_actions'] > 0:
            stats['overall_success_rate'] = (stats['successful_actions'] / stats['total_actions']) * 100
        else:
            stats['overall_success_rate'] = 0.0
        
        stats['performance_mode'] = self.performance_mode
        stats['is_active'] = self.is_active
        
        return stats
    
    def set_performance_mode(self, mode: str):
        """Set performance mode: balanced, precision, speed"""
        if mode not in ['balanced', 'precision', 'speed']:
            self.logger.warning(f"Invalid performance mode: {mode}")
            return
        
        self.performance_mode = mode
        
        if mode == 'precision':
            # High precision, slower speed
            self.automation_engine.disable_serpent_mode()
            self.serpent_vision.set_fps(15)
            
        elif mode == 'speed':
            # High speed, lower precision
            self.automation_engine.enable_serpent_mode()
            self.serpent_vision.set_fps(60)
            
        else:  # balanced
            # Balanced settings
            self.automation_engine.enable_serpent_mode()
            self.serpent_vision.set_fps(30)
        
        self.logger.info(f"Performance mode set to: {mode}")


class ActionCoordinator:
    """Coordinates actions between different AI systems"""
    
    def __init__(self, integration_system):
        self.integration_system = integration_system
        self.logger = logging.getLogger(__name__)
        self.pending_actions = deque(maxlen=50)
    
    def coordinate_action(self, action: Dict[str, Any]) -> bool:
        """Coordinate action execution across systems"""
        # Add coordination logic here
        self.pending_actions.append(action)
        return True


class PatternAnalyzer:
    """Analyzes patterns across all AI systems"""
    
    def __init__(self, integration_system):
        self.integration_system = integration_system
        self.logger = logging.getLogger(__name__)
        self.detected_patterns = {}
    
    def analyze_sequence(self, action_sequence: List[str]):
        """Analyze action sequence for patterns"""
        # Pattern analysis implementation
        if len(action_sequence) >= 3:
            pattern = tuple(action_sequence[-3:])
            if pattern in self.detected_patterns:
                self.detected_patterns[pattern] += 1
            else:
                self.detected_patterns[pattern] = 1
            
            self.integration_system.integration_stats['pattern_matches'] += 1


class PerformanceOptimizer:
    """Optimizes performance across all systems"""
    
    def __init__(self, integration_system):
        self.integration_system = integration_system
        self.logger = logging.getLogger(__name__)
    
    def optimize_systems(self):
        """Perform system-wide optimization"""
        # Get current performance stats
        stats = self.integration_system.get_integration_stats()
        
        # Adjust automation based on success rate
        self.integration_system.automation_engine.adaptive_difficulty_adjustment()
        
        # Optimize vision capture rate based on workload
        current_fps = stats['vision']['fps_target']
        if stats['vision']['avg_fps'] < current_fps * 0.8:
            # Reduce target FPS if falling behind
            new_fps = max(15, int(current_fps * 0.9))
            self.integration_system.serpent_vision.set_fps(new_fps)
            self.logger.info(f"Reduced vision FPS to {new_fps} for optimization")
        
        self.logger.debug("Performance optimization completed")