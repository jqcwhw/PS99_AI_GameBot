"""
Autonomous Play System for AI Game Bot
Enables fully autonomous gameplay with intelligent decision making and goal management
"""

import time
import logging
import threading
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import numpy as np

class PlayMode(Enum):
    IDLE = "idle"
    FARMING = "farming"
    COLLECTION = "collection"
    EXPLORATION = "exploration"
    OPTIMIZATION = "optimization"
    CUSTOM_OBJECTIVE = "custom_objective"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AutonomousGoal:
    """Represents an autonomous gameplay goal"""
    goal_id: str
    objective: str
    priority: Priority
    estimated_duration: float
    success_criteria: Dict[str, Any]
    current_progress: float
    max_attempts: int
    current_attempts: int
    strategies: List[str]
    performance_history: List[float]

@dataclass
class AutonomousAction:
    """Represents an action in autonomous play"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    expected_outcome: str
    confidence: float
    estimated_time: float
    dependencies: List[str]
    retry_count: int = 0
    max_retries: int = 3

class AutonomousPlaySystem:
    """System for fully autonomous gameplay"""
    
    def __init__(self, enhanced_vision=None, automation_engine=None, 
                 learning_system=None, gameplay_recorder=None):
        self.logger = logging.getLogger(__name__)
        
        # Core systems
        self.enhanced_vision = enhanced_vision
        self.automation_engine = automation_engine
        self.learning_system = learning_system
        self.gameplay_recorder = gameplay_recorder
        
        # Autonomous state
        self.is_autonomous = False
        self.current_mode = PlayMode.IDLE
        self.autonomous_thread = None
        
        # Goal management
        self.active_goals = {}
        self.completed_goals = {}
        self.failed_goals = {}
        self.goal_queue = deque()
        
        # Decision making
        self.decision_history = deque(maxlen=1000)
        self.performance_metrics = {}
        self.strategy_effectiveness = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        # Safety and monitoring
        self.safety_checks = []
        self.max_runtime = 3600  # 1 hour default
        self.start_time = 0
        self.error_count = 0
        self.max_errors = 10
        
        # Adaptive behavior
        self.behavior_parameters = {
            'aggression': 0.5,  # How aggressively to pursue goals
            'exploration_rate': 0.3,  # How often to try new strategies
            'patience': 0.7,  # How long to wait for results
            'risk_tolerance': 0.4  # Willingness to try risky strategies
        }
        
        # Knowledge base
        self.known_strategies = {}
        self.environmental_memory = {}
        self.player_preferences = {}
        
        self._load_autonomous_knowledge()
        self.logger.info("Autonomous Play System initialized")
    
    def start_autonomous_play(self, mode: PlayMode = PlayMode.FARMING, 
                            objectives: Optional[List[str]] = None,
                            max_duration: int = 3600) -> str:
        """Start autonomous gameplay"""
        if self.is_autonomous:
            return "Autonomous play already active"
        
        try:
            self.is_autonomous = True
            self.current_mode = mode
            self.start_time = time.time()
            self.max_runtime = max_duration
            self.error_count = 0
            
            # Initialize goals based on mode and objectives
            self._initialize_goals(mode, objectives)
            
            # Start autonomous thread
            self.autonomous_thread = threading.Thread(target=self._autonomous_play_loop)
            self.autonomous_thread.daemon = True
            self.autonomous_thread.start()
            
            # Start recording if available
            if self.gameplay_recorder:
                self.gameplay_recorder.start_recording(
                    objective=f"autonomous_{mode.value}",
                    session_type="autonomous"
                )
            
            result = f"Autonomous play started in {mode.value} mode"
            self.logger.info(result)
            return result
            
        except Exception as e:
            self.is_autonomous = False
            error_msg = f"Failed to start autonomous play: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def stop_autonomous_play(self) -> str:
        """Stop autonomous gameplay"""
        if not self.is_autonomous:
            return "No autonomous play active"
        
        try:
            self.is_autonomous = False
            
            if self.autonomous_thread:
                self.autonomous_thread.join(timeout=10.0)
            
            # Stop recording
            if self.gameplay_recorder:
                recording_result = self.gameplay_recorder.stop_recording()
                self.logger.info(f"Recording stopped: {recording_result}")
            
            # Save session results
            self._save_autonomous_session()
            
            duration = time.time() - self.start_time
            completed_goals = len(self.completed_goals)
            
            result = f"Autonomous play stopped. Duration: {duration:.1f}s, Goals completed: {completed_goals}"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to stop autonomous play: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _autonomous_play_loop(self):
        """Main autonomous play loop"""
        while self.is_autonomous:
            try:
                # Safety checks
                if not self._perform_safety_checks():
                    self.logger.warning("Safety check failed - stopping autonomous play")
                    break
                
                # Runtime limit check
                if time.time() - self.start_time > self.max_runtime:
                    self.logger.info("Runtime limit reached - stopping autonomous play")
                    break
                
                # Error limit check
                if self.error_count > self.max_errors:
                    self.logger.error("Error limit exceeded - stopping autonomous play")
                    break
                
                # Main decision cycle
                self._autonomous_decision_cycle()
                
                # Adaptive pause between cycles
                self._adaptive_pause()
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Autonomous play error: {e}")
                time.sleep(5.0)  # Error recovery pause
        
        self.is_autonomous = False
    
    def _autonomous_decision_cycle(self):
        """Core decision making cycle"""
        try:
            # 1. Assess current situation
            game_state = self._assess_game_state()
            
            # 2. Update goal progress
            self._update_goal_progress(game_state)
            
            # 3. Select next action
            next_action = self._select_next_action(game_state)
            
            # 4. Execute action
            if next_action:
                result = self._execute_autonomous_action(next_action, game_state)
                
                # 5. Learn from result
                self._learn_from_action_result(next_action, result, game_state)
                
                # 6. Adapt behavior
                self._adapt_behavior(result)
            else:
                # No action available - reassess goals
                self._reassess_goals(game_state)
                
        except Exception as e:
            self.logger.error(f"Decision cycle error: {e}")
            self.error_count += 1
    
    def _assess_game_state(self) -> Dict[str, Any]:
        """Comprehensive assessment of current game state"""
        game_state = {
            'timestamp': time.time(),
            'autonomous_session_duration': time.time() - self.start_time,
            'detected_elements': {},
            'opportunities': [],
            'threats': [],
            'navigation_context': {},
            'resources': {},
            'ui_state': {},
            'minigame_active': False
        }
        
        try:
            if self.enhanced_vision:
                vision_analysis = self.enhanced_vision.analyze_game_state()
                game_state.update(vision_analysis)
                
                # Identify opportunities
                opportunities = self._identify_opportunities(vision_analysis)
                game_state['opportunities'] = opportunities
                
                # Identify threats or blockers
                threats = self._identify_threats(vision_analysis)
                game_state['threats'] = threats
        
        except Exception as e:
            self.logger.error(f"Game state assessment failed: {e}")
        
        return game_state
    
    def _identify_opportunities(self, vision_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities in current game state"""
        opportunities = []
        
        try:
            detected_elements = vision_analysis.get('detected_elements', {})
            
            # Treasure opportunities
            if 'chests' in detected_elements and detected_elements['chests']:
                opportunities.append({
                    'type': 'treasure_collection',
                    'priority': Priority.HIGH,
                    'count': len(detected_elements['chests']),
                    'estimated_value': len(detected_elements['chests']) * 10,
                    'locations': [elem.position for elem in detected_elements['chests']]
                })
            
            # Egg hatching opportunities
            if 'eggs' in detected_elements and detected_elements['eggs']:
                opportunities.append({
                    'type': 'egg_hatching',
                    'priority': Priority.MEDIUM,
                    'count': len(detected_elements['eggs']),
                    'estimated_value': len(detected_elements['eggs']) * 5,
                    'locations': [elem.position for elem in detected_elements['eggs']]
                })
            
            # Resource gathering opportunities
            if 'resources' in detected_elements and detected_elements['resources']:
                opportunities.append({
                    'type': 'resource_gathering',
                    'priority': Priority.MEDIUM,
                    'count': len(detected_elements['resources']),
                    'estimated_value': len(detected_elements['resources']) * 3,
                    'locations': [elem.position for elem in detected_elements['resources']]
                })
            
            # Farming opportunities (breakables areas)
            active_zones = vision_analysis.get('active_zones', {})
            if 'breakables' in active_zones:
                opportunities.append({
                    'type': 'farming',
                    'priority': Priority.MEDIUM,
                    'zone_info': active_zones['breakables'],
                    'estimated_value': 15,
                    'sustainable': True  # Can be repeated
                })
            
            # Minigame opportunities
            minigame_status = vision_analysis.get('minigame_status', {})
            if minigame_status.get('detected', False):
                priority = Priority.CRITICAL if 'timed' in minigame_status.get('type', '') else Priority.HIGH
                opportunities.append({
                    'type': 'minigame',
                    'priority': priority,
                    'minigame_type': minigame_status.get('type'),
                    'time_limit': minigame_status.get('time_limit'),
                    'estimated_value': 20
                })
        
        except Exception as e:
            self.logger.error(f"Opportunity identification failed: {e}")
        
        return opportunities
    
    def _identify_threats(self, vision_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify threats or blockers in current game state"""
        threats = []
        
        try:
            # UI errors or problems
            ui_state = vision_analysis.get('ui_state', {})
            if ui_state.get('error_dialog', False):
                threats.append({
                    'type': 'ui_error',
                    'priority': Priority.HIGH,
                    'description': 'Error dialog detected'
                })
            
            # Stuck in place detection
            navigation_context = vision_analysis.get('navigation_context', {})
            if navigation_context.get('blocked_paths'):
                threats.append({
                    'type': 'navigation_blocked',
                    'priority': Priority.MEDIUM,
                    'description': 'Movement paths blocked'
                })
            
            # Resource depletion
            if not vision_analysis.get('detected_elements'):
                threats.append({
                    'type': 'resource_depletion',
                    'priority': Priority.LOW,
                    'description': 'No immediate opportunities detected'
                })
        
        except Exception as e:
            self.logger.error(f"Threat identification failed: {e}")
        
        return threats
    
    def _select_next_action(self, game_state: Dict[str, Any]) -> Optional[AutonomousAction]:
        """Select the next action to take"""
        try:
            # Priority order: Handle threats, pursue opportunities, continue goals
            
            # 1. Handle critical threats first
            for threat in game_state.get('threats', []):
                if threat['priority'] == Priority.CRITICAL:
                    return self._create_threat_response_action(threat)
            
            # 2. Pursue high-value opportunities
            opportunities = sorted(game_state.get('opportunities', []),
                                 key=lambda x: (x['priority'].value, x.get('estimated_value', 0)),
                                 reverse=True)
            
            for opportunity in opportunities:
                if opportunity['priority'].value >= Priority.HIGH.value:
                    action = self._create_opportunity_action(opportunity, game_state)
                    if action:
                        return action
            
            # 3. Continue with active goals
            for goal in self.active_goals.values():
                if goal.priority.value >= Priority.MEDIUM.value:
                    action = self._create_goal_action(goal, game_state)
                    if action:
                        return action
            
            # 4. Handle medium threats
            for threat in game_state.get('threats', []):
                if threat['priority'] == Priority.MEDIUM:
                    return self._create_threat_response_action(threat)
            
            # 5. Pursue remaining opportunities
            for opportunity in opportunities:
                action = self._create_opportunity_action(opportunity, game_state)
                if action:
                    return action
            
            # 6. Exploration action if nothing else
            return self._create_exploration_action(game_state)
            
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            return None
    
    def _create_opportunity_action(self, opportunity: Dict[str, Any], 
                                 game_state: Dict[str, Any]) -> Optional[AutonomousAction]:
        """Create action to pursue an opportunity"""
        try:
            action_id = f"opp_{opportunity['type']}_{int(time.time())}"
            
            if opportunity['type'] == 'treasure_collection':
                return AutonomousAction(
                    action_id=action_id,
                    action_type='open_chests',
                    parameters={
                        'chest_positions': opportunity['locations'],
                        'approach_strategy': 'efficient_pathing'
                    },
                    expected_outcome='chests_opened',
                    confidence=0.8,
                    estimated_time=len(opportunity['locations']) * 2.0
                )
            
            elif opportunity['type'] == 'egg_hatching':
                return AutonomousAction(
                    action_id=action_id,
                    action_type='hatch_eggs',
                    parameters={
                        'egg_positions': opportunity['locations'],
                        'approach_strategy': 'batch_processing'
                    },
                    expected_outcome='eggs_hatched',
                    confidence=0.7,
                    estimated_time=len(opportunity['locations']) * 1.5
                )
            
            elif opportunity['type'] == 'farming':
                return AutonomousAction(
                    action_id=action_id,
                    action_type='farm_area',
                    parameters={
                        'zone_info': opportunity['zone_info'],
                        'duration': 30.0,  # Farm for 30 seconds
                        'strategy': 'stay_and_collect'
                    },
                    expected_outcome='resources_farmed',
                    confidence=0.9,
                    estimated_time=30.0
                )
            
            elif opportunity['type'] == 'minigame':
                return AutonomousAction(
                    action_id=action_id,
                    action_type='engage_minigame',
                    parameters={
                        'minigame_type': opportunity['minigame_type'],
                        'time_limit': opportunity.get('time_limit'),
                        'strategy': 'adaptive'
                    },
                    expected_outcome='minigame_completed',
                    confidence=0.6,
                    estimated_time=opportunity.get('time_limit', 60.0)
                )
        
        except Exception as e:
            self.logger.error(f"Opportunity action creation failed: {e}")
        
        return None
    
    def _execute_autonomous_action(self, action: AutonomousAction, 
                                 game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an autonomous action"""
        try:
            self.logger.info(f"Executing autonomous action: {action.action_type}")
            
            result = {
                'action_id': action.action_id,
                'success': False,
                'outcome': '',
                'execution_time': 0,
                'error': None
            }
            
            start_time = time.time()
            
            # Execute based on action type
            if action.action_type == 'open_chests':
                outcome = self._execute_chest_opening(action.parameters)
                result['success'] = 'opened' in outcome.lower()
                result['outcome'] = outcome
            
            elif action.action_type == 'hatch_eggs':
                outcome = self._execute_egg_hatching(action.parameters)
                result['success'] = 'hatched' in outcome.lower()
                result['outcome'] = outcome
            
            elif action.action_type == 'farm_area':
                outcome = self._execute_farming(action.parameters)
                result['success'] = 'farming' in outcome.lower()
                result['outcome'] = outcome
            
            elif action.action_type == 'engage_minigame':
                outcome = self._execute_minigame(action.parameters)
                result['success'] = 'completed' in outcome.lower()
                result['outcome'] = outcome
            
            elif action.action_type == 'explore_area':
                outcome = self._execute_exploration(action.parameters)
                result['success'] = True  # Exploration is always "successful"
                result['outcome'] = outcome
            
            else:
                result['error'] = f"Unknown action type: {action.action_type}"
            
            result['execution_time'] = time.time() - start_time
            
            # Record action for learning
            if self.gameplay_recorder:
                self.gameplay_recorder.record_manual_action(
                    action_type=action.action_type,
                    position=action.parameters.get('position'),
                    success=result['success'],
                    reward=10.0 if result['success'] else -1.0
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return {
                'action_id': action.action_id,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _execute_chest_opening(self, parameters: Dict[str, Any]) -> str:
        """Execute chest opening action"""
        try:
            if self.automation_engine:
                chest_positions = parameters.get('chest_positions', [])
                return self.automation_engine.open_chests(chest_positions)
            else:
                return "No automation engine available"
        except Exception as e:
            self.logger.error(f"Chest opening failed: {e}")
            return f"Chest opening failed: {e}"
    
    def _execute_egg_hatching(self, parameters: Dict[str, Any]) -> str:
        """Execute egg hatching action"""
        try:
            if self.automation_engine:
                egg_positions = parameters.get('egg_positions', [])
                return self.automation_engine.hatch_eggs(egg_positions)
            else:
                return "No automation engine available"
        except Exception as e:
            self.logger.error(f"Egg hatching failed: {e}")
            return f"Egg hatching failed: {e}"
    
    def _execute_farming(self, parameters: Dict[str, Any]) -> str:
        """Execute farming action"""
        try:
            if self.automation_engine:
                zone_info = parameters.get('zone_info', {})
                if zone_info and 'center' in zone_info:
                    return self.automation_engine.stay_in_breakables_area([zone_info['center']])
                else:
                    return "No valid farming zone"
            else:
                return "No automation engine available"
        except Exception as e:
            self.logger.error(f"Farming failed: {e}")
            return f"Farming failed: {e}"
    
    def _execute_minigame(self, parameters: Dict[str, Any]) -> str:
        """Execute minigame action"""
        # This would involve specific minigame logic
        minigame_type = parameters.get('minigame_type', 'unknown')
        return f"Engaged with {minigame_type} minigame"
    
    def _execute_exploration(self, parameters: Dict[str, Any]) -> str:
        """Execute exploration action"""
        try:
            if self.automation_engine:
                # Move to a random location for exploration
                target_pos = parameters.get('target_position', (400, 300))
                return self.automation_engine.move_to_area(target_pos)
            else:
                return "Exploring area"
        except Exception as e:
            return f"Exploration failed: {e}"
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get current autonomous play status"""
        status = {
            'is_autonomous': self.is_autonomous,
            'current_mode': self.current_mode.value if self.current_mode else 'none',
            'runtime': 0,
            'active_goals': len(self.active_goals),
            'completed_goals': len(self.completed_goals),
            'failed_goals': len(self.failed_goals),
            'error_count': self.error_count,
            'performance_metrics': self.performance_metrics.copy(),
            'behavior_parameters': self.behavior_parameters.copy()
        }
        
        if self.is_autonomous:
            status['runtime'] = time.time() - self.start_time
            status['remaining_time'] = max(0, self.max_runtime - status['runtime'])
        
        return status
    
    # Helper methods for goal management, safety, etc.
    def _initialize_goals(self, mode: PlayMode, objectives: Optional[List[str]]):
        """Initialize goals based on play mode"""
        if mode == PlayMode.FARMING:
            self._add_goal("continuous_farming", "Farm resources continuously", Priority.HIGH)
        elif mode == PlayMode.COLLECTION:
            self._add_goal("collect_treasures", "Collect all available treasures", Priority.HIGH)
            self._add_goal("hatch_all_eggs", "Hatch all available eggs", Priority.MEDIUM)
        elif mode == PlayMode.EXPLORATION:
            self._add_goal("explore_map", "Explore unknown areas", Priority.MEDIUM)
        
        # Add custom objectives
        if objectives:
            for obj in objectives:
                self._add_goal(f"custom_{obj}", obj, Priority.MEDIUM)
    
    def _add_goal(self, goal_id: str, objective: str, priority: Priority):
        """Add a new goal"""
        goal = AutonomousGoal(
            goal_id=goal_id,
            objective=objective,
            priority=priority,
            estimated_duration=300.0,  # 5 minutes default
            success_criteria={},
            current_progress=0.0,
            max_attempts=10,
            current_attempts=0,
            strategies=[],
            performance_history=[]
        )
        self.active_goals[goal_id] = goal
    
    def _perform_safety_checks(self) -> bool:
        """Perform safety checks"""
        try:
            # Check if automation engine is still responsive
            if self.automation_engine and not self.automation_engine.is_active():
                self.logger.warning("Automation engine not active")
                return False
            
            # Check for game crashes or unexpected states
            if self.enhanced_vision:
                game_state = self.enhanced_vision.analyze_game_state()
                if 'error' in game_state:
                    self.logger.warning(f"Vision system error: {game_state['error']}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False
    
    def _adaptive_pause(self):
        """Adaptive pause between decision cycles"""
        base_pause = 1.0
        
        # Adjust based on current mode
        if self.current_mode == PlayMode.FARMING:
            base_pause = 0.5  # Faster for farming
        elif self.current_mode == PlayMode.EXPLORATION:
            base_pause = 2.0  # Slower for exploration
        
        # Add some randomness for human-like behavior
        pause_time = base_pause + random.uniform(-0.2, 0.2)
        time.sleep(max(0.1, pause_time))
    
    def _update_goal_progress(self, game_state: Dict[str, Any]):
        """Update progress on active goals"""
        # Placeholder implementation
        pass
    
    def _reassess_goals(self, game_state: Dict[str, Any]):
        """Reassess goals when no immediate actions available"""
        # Placeholder implementation
        pass
    
    def _learn_from_action_result(self, action: AutonomousAction, 
                                result: Dict[str, Any], game_state: Dict[str, Any]):
        """Learn from action execution results"""
        # Update strategy effectiveness
        strategy_key = f"{action.action_type}_{action.parameters.get('strategy', 'default')}"
        if result.get('success', False):
            self.strategy_effectiveness[strategy_key]['success'] += 1
        else:
            self.strategy_effectiveness[strategy_key]['failure'] += 1
    
    def _adapt_behavior(self, result: Dict[str, Any]):
        """Adapt behavior based on recent results"""
        # Adjust behavior parameters based on success/failure
        if result.get('success', False):
            # Successful action - slightly increase aggression
            self.behavior_parameters['aggression'] = min(1.0, 
                self.behavior_parameters['aggression'] + 0.01)
        else:
            # Failed action - be more cautious
            self.behavior_parameters['aggression'] = max(0.1, 
                self.behavior_parameters['aggression'] - 0.02)
    
    def _create_threat_response_action(self, threat: Dict[str, Any]) -> Optional[AutonomousAction]:
        """Create action to respond to a threat"""
        # Placeholder implementation
        return None
    
    def _create_goal_action(self, goal: AutonomousGoal, 
                          game_state: Dict[str, Any]) -> Optional[AutonomousAction]:
        """Create action to pursue a goal"""
        # Placeholder implementation
        return None
    
    def _create_exploration_action(self, game_state: Dict[str, Any]) -> Optional[AutonomousAction]:
        """Create exploration action"""
        return AutonomousAction(
            action_id=f"explore_{int(time.time())}",
            action_type='explore_area',
            parameters={
                'target_position': (random.randint(100, 700), random.randint(100, 500)),
                'exploration_radius': 100
            },
            expected_outcome='area_explored',
            confidence=0.8,
            estimated_time=10.0
        )
    
    def _save_autonomous_session(self):
        """Save autonomous session data"""
        try:
            session_data = {
                'timestamp': time.time(),
                'mode': self.current_mode.value,
                'duration': time.time() - self.start_time,
                'completed_goals': len(self.completed_goals),
                'failed_goals': len(self.failed_goals),
                'error_count': self.error_count,
                'performance_metrics': self.performance_metrics,
                'strategy_effectiveness': dict(self.strategy_effectiveness)
            }
            
            sessions_dir = Path("data/autonomous_sessions")
            sessions_dir.mkdir(exist_ok=True)
            
            session_file = sessions_dir / f"session_{int(self.start_time)}.json"
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info("Autonomous session data saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save session data: {e}")
    
    def _load_autonomous_knowledge(self):
        """Load autonomous knowledge from previous sessions"""
        try:
            knowledge_file = Path("data/autonomous_sessions/knowledge.json")
            if knowledge_file.exists():
                with open(knowledge_file, 'r') as f:
                    knowledge = json.load(f)
                
                self.known_strategies = knowledge.get('strategies', {})
                self.environmental_memory = knowledge.get('environment', {})
                self.player_preferences = knowledge.get('preferences', {})
                
                self.logger.info("Autonomous knowledge loaded")
        except Exception as e:
            self.logger.error(f"Failed to load autonomous knowledge: {e}")