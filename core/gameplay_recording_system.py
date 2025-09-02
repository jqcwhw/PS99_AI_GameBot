"""
Gameplay Recording System for AI Game Bot
Records, analyzes, and learns from gameplay sessions to create optimized automation
"""

import time
import json
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import pickle
import cv2

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False

@dataclass
class GameplayAction:
    """Represents a single gameplay action"""
    action_type: str  # 'click', 'key', 'move', 'drag', 'scroll'
    timestamp: float
    position: Optional[Tuple[int, int]]
    key: Optional[str]
    button: Optional[str]
    duration: float
    screen_context: Dict[str, Any]
    game_state: Dict[str, Any]
    success: Optional[bool] = None
    reward: Optional[float] = None

@dataclass
class GameplaySequence:
    """Represents a sequence of gameplay actions"""
    sequence_id: str
    actions: List[GameplayAction]
    start_time: float
    end_time: float
    objective: str
    success_rate: float
    optimization_level: int
    tags: List[str]
    performance_metrics: Dict[str, float]

@dataclass
class RecordingSession:
    """Represents a complete recording session"""
    session_id: str
    start_time: float
    end_time: float
    sequences: List[GameplaySequence]
    total_actions: int
    session_type: str  # 'manual', 'guided', 'optimization'
    metadata: Dict[str, Any]

class GameplayRecordingSystem:
    """System for recording and learning from gameplay sessions"""
    
    def __init__(self, enhanced_vision=None):
        self.logger = logging.getLogger(__name__)
        self.enhanced_vision = enhanced_vision
        
        # Recording state
        self.is_recording = False
        self.current_session: Optional[RecordingSession] = None
        self.recorded_actions = deque(maxlen=10000)
        self.recording_thread = None
        
        # Analysis and learning
        self.learned_sequences = {}
        self.optimization_patterns = {}
        self.performance_baselines = {}
        
        # Mouse/keyboard monitoring
        self.last_mouse_pos = (0, 0)
        self.last_action_time = 0
        self.action_threshold = 0.1  # Minimum time between recorded actions
        
        # Session management
        self.sessions_dir = Path("data/gameplay_sessions")
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.action_timing = {}
        self.efficiency_metrics = {}
        
        self._load_learned_sequences()
        self.logger.info("Gameplay Recording System initialized")
    
    def start_recording(self, objective: str = "general_gameplay", 
                       session_type: str = "manual") -> str:
        """Start recording a gameplay session"""
        if self.is_recording:
            return "Recording already in progress"
        
        try:
            session_id = f"session_{int(time.time())}"
            
            self.current_session = RecordingSession(
                session_id=session_id,
                start_time=time.time(),
                end_time=0,
                sequences=[],
                total_actions=0,
                session_type=session_type,
                metadata={'objective': objective}
            )
            
            self.is_recording = True
            self.recorded_actions.clear()
            
            # Start monitoring thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            self.logger.info(f"Started recording session: {session_id}")
            return f"Recording started: {session_id}"
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            return f"Recording failed: {e}"
    
    def stop_recording(self) -> str:
        """Stop recording and analyze the session"""
        if not self.is_recording:
            return "No recording in progress"
        
        try:
            self.is_recording = False
            
            if self.recording_thread:
                self.recording_thread.join(timeout=5.0)
            
            if self.current_session:
                self.current_session.end_time = time.time()
                self.current_session.total_actions = len(self.recorded_actions)
                
                # Analyze recorded actions
                sequences = self._analyze_recorded_actions()
                self.current_session.sequences = sequences
                
                # Save session
                self._save_session(self.current_session)
                
                # Learn from session
                learning_results = self._learn_from_session(self.current_session)
                
                session_id = self.current_session.session_id
                self.current_session = None
                
                self.logger.info(f"Recording stopped and analyzed: {session_id}")
                return f"Recording completed: {session_id}. {learning_results}"
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}")
            return f"Stop recording failed: {e}"
    
    def _recording_loop(self):
        """Main recording loop that monitors user input"""
        while self.is_recording:
            try:
                current_time = time.time()
                
                # Capture current game state
                game_state = self._capture_game_state()
                
                # Monitor for mouse/keyboard changes
                if PYAUTOGUI_AVAILABLE:
                    current_mouse_pos = pyautogui.position()
                    
                    # Detect mouse movement
                    if current_mouse_pos != self.last_mouse_pos:
                        if current_time - self.last_action_time > self.action_threshold:
                            self._record_action(
                                action_type='move',
                                position=current_mouse_pos,
                                game_state=game_state,
                                timestamp=current_time
                            )
                            self.last_mouse_pos = current_mouse_pos
                            self.last_action_time = current_time
                
                time.sleep(0.05)  # 20 FPS monitoring
                
            except Exception as e:
                self.logger.error(f"Recording loop error: {e}")
                time.sleep(0.1)
    
    def record_manual_action(self, action_type: str, **kwargs) -> bool:
        """Manually record an action (for integration with automation engine)"""
        try:
            if not self.is_recording:
                return False
            
            game_state = self._capture_game_state()
            
            action = GameplayAction(
                action_type=action_type,
                timestamp=time.time(),
                position=kwargs.get('position'),
                key=kwargs.get('key'),
                button=kwargs.get('button', 'left'),
                duration=kwargs.get('duration', 0.1),
                screen_context=game_state.get('screen_context', {}),
                game_state=game_state,
                success=kwargs.get('success'),
                reward=kwargs.get('reward')
            )
            
            self.recorded_actions.append(action)
            return True
            
        except Exception as e:
            self.logger.error(f"Manual action recording failed: {e}")
            return False
    
    def _capture_game_state(self) -> Dict[str, Any]:
        """Capture current game state for context"""
        try:
            game_state = {
                'timestamp': time.time(),
                'screen_context': {},
                'detected_elements': {},
                'zones': {},
                'ui_state': {}
            }
            
            if self.enhanced_vision:
                # Use enhanced vision for comprehensive analysis
                vision_analysis = self.enhanced_vision.analyze_game_state()
                game_state.update(vision_analysis)
            else:
                # Basic screenshot capture
                if PYAUTOGUI_AVAILABLE:
                    screenshot = pyautogui.screenshot()
                    game_state['screenshot_size'] = screenshot.size
                
            return game_state
            
        except Exception as e:
            self.logger.error(f"Game state capture failed: {e}")
            return {'timestamp': time.time(), 'error': str(e)}
    
    def _record_action(self, action_type: str, position: Optional[Tuple[int, int]] = None,
                      key: Optional[str] = None, game_state: Optional[Dict] = None,
                      timestamp: Optional[float] = None, **kwargs):
        """Record a single action"""
        try:
            if not game_state:
                game_state = self._capture_game_state()
            
            action = GameplayAction(
                action_type=action_type,
                timestamp=timestamp or time.time(),
                position=position,
                key=key,
                button=kwargs.get('button'),
                duration=kwargs.get('duration', 0.1),
                screen_context=game_state.get('screen_context', {}),
                game_state=game_state,
                success=kwargs.get('success'),
                reward=kwargs.get('reward')
            )
            
            self.recorded_actions.append(action)
            
        except Exception as e:
            self.logger.error(f"Action recording failed: {e}")
    
    def _analyze_recorded_actions(self) -> List[GameplaySequence]:
        """Analyze recorded actions to identify sequences and patterns"""
        sequences = []
        
        try:
            if not self.recorded_actions:
                return sequences
            
            # Convert to list for processing
            actions = list(self.recorded_actions)
            
            # Identify meaningful sequences
            sequences = self._identify_action_sequences(actions)
            
            # Analyze each sequence
            for sequence in sequences:
                sequence.performance_metrics = self._calculate_sequence_metrics(sequence)
                sequence.tags = self._generate_sequence_tags(sequence)
            
            self.logger.info(f"Identified {len(sequences)} action sequences")
            
        except Exception as e:
            self.logger.error(f"Action analysis failed: {e}")
        
        return sequences
    
    def _identify_action_sequences(self, actions: List[GameplayAction]) -> List[GameplaySequence]:
        """Identify coherent action sequences from recorded actions"""
        sequences = []
        
        try:
            current_sequence = []
            sequence_start_time = None
            last_action_time = 0
            sequence_counter = 0
            
            for action in actions:
                # Define sequence boundaries (pauses > 2 seconds or context changes)
                if (action.timestamp - last_action_time > 2.0 or 
                    self._is_context_change(action, current_sequence)):
                    
                    # Save previous sequence if it has enough actions
                    if len(current_sequence) >= 3:
                        sequence_id = f"seq_{sequence_counter}_{int(sequence_start_time)}"
                        
                        sequences.append(GameplaySequence(
                            sequence_id=sequence_id,
                            actions=current_sequence.copy(),
                            start_time=sequence_start_time,
                            end_time=current_sequence[-1].timestamp,
                            objective=self._infer_sequence_objective(current_sequence),
                            success_rate=1.0,  # Initial assumption
                            optimization_level=0,
                            tags=[],
                            performance_metrics={}
                        ))
                        
                        sequence_counter += 1
                    
                    # Start new sequence
                    current_sequence = [action]
                    sequence_start_time = action.timestamp
                else:
                    current_sequence.append(action)
                
                last_action_time = action.timestamp
            
            # Don't forget the last sequence
            if len(current_sequence) >= 3:
                sequence_id = f"seq_{sequence_counter}_{int(sequence_start_time)}"
                sequences.append(GameplaySequence(
                    sequence_id=sequence_id,
                    actions=current_sequence,
                    start_time=sequence_start_time,
                    end_time=current_sequence[-1].timestamp,
                    objective=self._infer_sequence_objective(current_sequence),
                    success_rate=1.0,
                    optimization_level=0,
                    tags=[],
                    performance_metrics={}
                ))
            
        except Exception as e:
            self.logger.error(f"Sequence identification failed: {e}")
        
        return sequences
    
    def _is_context_change(self, action: GameplayAction, 
                          current_sequence: List[GameplayAction]) -> bool:
        """Determine if action represents a context change"""
        if not current_sequence:
            return False
        
        last_action = current_sequence[-1]
        
        # Check for significant UI changes
        if (action.game_state.get('ui_state') != 
            last_action.game_state.get('ui_state')):
            return True
        
        # Check for zone changes
        current_zones = action.game_state.get('zones', {})
        last_zones = last_action.game_state.get('zones', {})
        if set(current_zones.keys()) != set(last_zones.keys()):
            return True
        
        return False
    
    def _infer_sequence_objective(self, actions: List[GameplayAction]) -> str:
        """Infer the objective of an action sequence"""
        try:
            # Analyze action patterns and game state changes
            action_types = [action.action_type for action in actions]
            
            # Look for common patterns
            if 'click' in action_types and len([a for a in actions if a.action_type == 'click']) > 5:
                return "collection_sequence"
            
            if any('chest' in str(action.game_state.get('detected_elements', {})) for action in actions):
                return "chest_opening"
            
            if any('egg' in str(action.game_state.get('detected_elements', {})) for action in actions):
                return "egg_hatching"
            
            if 'move' in action_types and len(actions) > 10:
                return "navigation_sequence"
            
            # Default
            return "general_gameplay"
            
        except Exception as e:
            self.logger.error(f"Objective inference failed: {e}")
            return "unknown"
    
    def _calculate_sequence_metrics(self, sequence: GameplaySequence) -> Dict[str, float]:
        """Calculate performance metrics for a sequence"""
        try:
            metrics = {
                'duration': sequence.end_time - sequence.start_time,
                'action_count': len(sequence.actions),
                'actions_per_second': 0,
                'efficiency_score': 0,
                'accuracy_score': 0,
                'smoothness_score': 0
            }
            
            if metrics['duration'] > 0:
                metrics['actions_per_second'] = metrics['action_count'] / metrics['duration']
            
            # Calculate efficiency (actions that led to progress)
            successful_actions = len([a for a in sequence.actions if a.success is True])
            if sequence.actions:
                metrics['efficiency_score'] = successful_actions / len(sequence.actions)
            
            # Calculate smoothness (consistent timing)
            if len(sequence.actions) > 1:
                intervals = []
                for i in range(1, len(sequence.actions)):
                    interval = sequence.actions[i].timestamp - sequence.actions[i-1].timestamp
                    intervals.append(interval)
                
                if intervals:
                    metrics['smoothness_score'] = 1.0 / (1.0 + np.std(intervals))
            
            # Calculate accuracy (precise targeting)
            click_actions = [a for a in sequence.actions if a.action_type == 'click' and a.position]
            if len(click_actions) > 1:
                positions = [a.position for a in click_actions]
                # Measure consistency of click positions
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                position_variance = np.var(x_coords) + np.var(y_coords)
                metrics['accuracy_score'] = 1.0 / (1.0 + position_variance / 1000)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return {}
    
    def _generate_sequence_tags(self, sequence: GameplaySequence) -> List[str]:
        """Generate descriptive tags for a sequence"""
        tags = []
        
        try:
            # Objective-based tags
            tags.append(sequence.objective)
            
            # Performance-based tags
            metrics = sequence.performance_metrics
            if metrics.get('efficiency_score', 0) > 0.8:
                tags.append('high_efficiency')
            if metrics.get('actions_per_second', 0) > 2.0:
                tags.append('fast_execution')
            if metrics.get('smoothness_score', 0) > 0.7:
                tags.append('smooth_execution')
            
            # Content-based tags
            action_types = set(action.action_type for action in sequence.actions)
            if 'click' in action_types:
                tags.append('clicking_focused')
            if 'move' in action_types:
                tags.append('movement_heavy')
            if 'key' in action_types:
                tags.append('keyboard_usage')
            
            # Complexity tags
            if len(sequence.actions) > 20:
                tags.append('complex_sequence')
            elif len(sequence.actions) < 5:
                tags.append('simple_sequence')
            
        except Exception as e:
            self.logger.error(f"Tag generation failed: {e}")
        
        return tags
    
    def _learn_from_session(self, session: RecordingSession) -> str:
        """Learn patterns and optimizations from a completed session"""
        try:
            learning_results = []
            
            # Analyze sequences for optimization opportunities
            for sequence in session.sequences:
                optimization = self._optimize_sequence(sequence)
                if optimization:
                    self.learned_sequences[sequence.sequence_id] = optimization
                    learning_results.append(f"Optimized {sequence.objective}")
            
            # Update performance baselines
            self._update_performance_baselines(session)
            
            # Extract reusable patterns
            patterns = self._extract_reusable_patterns(session)
            self.optimization_patterns.update(patterns)
            
            self._save_learned_sequences()
            
            result = f"Learned {len(learning_results)} optimizations from session"
            self.logger.info(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Session learning failed: {e}")
            return f"Learning failed: {e}"
    
    def _optimize_sequence(self, sequence: GameplaySequence) -> Optional[Dict[str, Any]]:
        """Optimize a gameplay sequence for better performance"""
        try:
            optimization = {
                'original_sequence': sequence.sequence_id,
                'optimized_actions': [],
                'improvements': [],
                'estimated_speedup': 1.0
            }
            
            # Remove redundant actions
            filtered_actions = self._remove_redundant_actions(sequence.actions)
            optimization['optimized_actions'] = filtered_actions
            
            if len(filtered_actions) < len(sequence.actions):
                reduction = len(sequence.actions) - len(filtered_actions)
                optimization['improvements'].append(f"Removed {reduction} redundant actions")
            
            # Optimize timing
            timing_optimized = self._optimize_action_timing(filtered_actions)
            optimization['optimized_actions'] = timing_optimized
            
            # Calculate estimated speedup
            original_duration = sequence.end_time - sequence.start_time
            optimized_duration = self._estimate_optimized_duration(timing_optimized)
            if optimized_duration > 0:
                optimization['estimated_speedup'] = original_duration / optimized_duration
            
            return optimization
            
        except Exception as e:
            self.logger.error(f"Sequence optimization failed: {e}")
            return None
    
    def _remove_redundant_actions(self, actions: List[GameplayAction]) -> List[GameplayAction]:
        """Remove redundant or unnecessary actions"""
        filtered = []
        
        try:
            for i, action in enumerate(actions):
                # Skip redundant moves
                if action.action_type == 'move' and i > 0:
                    if actions[i-1].action_type == 'move' and actions[i-1].position == action.position:
                        continue
                
                # Skip very quick repeated clicks
                if action.action_type == 'click' and i > 0:
                    if (actions[i-1].action_type == 'click' and 
                        actions[i-1].position == action.position and
                        action.timestamp - actions[i-1].timestamp < 0.1):
                        continue
                
                filtered.append(action)
            
        except Exception as e:
            self.logger.error(f"Action filtering failed: {e}")
            return actions
        
        return filtered
    
    def _optimize_action_timing(self, actions: List[GameplayAction]) -> List[GameplayAction]:
        """Optimize timing between actions"""
        optimized = []
        
        try:
            for i, action in enumerate(actions):
                optimized_action = action
                
                # Reduce unnecessary delays
                if i > 0:
                    time_gap = action.timestamp - actions[i-1].timestamp
                    if time_gap > 1.0 and action.action_type in ['click', 'key']:
                        # Reduce gap to optimal timing
                        optimal_gap = 0.3 if action.action_type == 'click' else 0.1
                        optimized_action.timestamp = actions[i-1].timestamp + optimal_gap
                
                optimized.append(optimized_action)
            
        except Exception as e:
            self.logger.error(f"Timing optimization failed: {e}")
            return actions
        
        return optimized
    
    def create_automation_from_recording(self, session_id: str, 
                                       sequence_id: Optional[str] = None) -> Dict[str, Any]:
        """Create automation script from recorded session"""
        try:
            session = self._load_session(session_id)
            if not session:
                return {'error': f'Session {session_id} not found'}
            
            # Get specific sequence or best sequence
            if sequence_id:
                target_sequence = next((s for s in session.sequences if s.sequence_id == sequence_id), None)
                if not target_sequence:
                    return {'error': f'Sequence {sequence_id} not found'}
                sequences = [target_sequence]
            else:
                # Select best performing sequences
                sequences = sorted(session.sequences, 
                                 key=lambda s: s.performance_metrics.get('efficiency_score', 0),
                                 reverse=True)[:3]
            
            automation_script = {
                'name': f"automation_{session_id}",
                'description': f"Generated from session {session_id}",
                'sequences': [],
                'loop_optimization': True,
                'estimated_performance': {}
            }
            
            for sequence in sequences:
                script_sequence = {
                    'id': sequence.sequence_id,
                    'objective': sequence.objective,
                    'actions': [asdict(action) for action in sequence.actions],
                    'optimized_actions': self.learned_sequences.get(sequence.sequence_id, {}).get('optimized_actions', []),
                    'tags': sequence.tags,
                    'metrics': sequence.performance_metrics
                }
                automation_script['sequences'].append(script_sequence)
            
            # Save automation script
            script_path = self.sessions_dir / f"automation_{session_id}.json"
            with open(script_path, 'w') as f:
                json.dump(automation_script, f, indent=2, default=str)
            
            return {
                'success': True,
                'automation_file': str(script_path),
                'sequences_count': len(sequences),
                'estimated_speedup': sum(self.learned_sequences.get(s.sequence_id, {}).get('estimated_speedup', 1.0) 
                                       for s in sequences) / len(sequences)
            }
            
        except Exception as e:
            self.logger.error(f"Automation creation failed: {e}")
            return {'error': str(e)}
    
    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status"""
        status = {
            'is_recording': self.is_recording,
            'current_session': None,
            'recorded_actions_count': len(self.recorded_actions),
            'learned_sequences_count': len(self.learned_sequences),
            'optimization_patterns_count': len(self.optimization_patterns)
        }
        
        if self.current_session:
            status['current_session'] = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time,
                'duration': time.time() - self.current_session.start_time,
                'session_type': self.current_session.session_type,
                'objective': self.current_session.metadata.get('objective', 'unknown')
            }
        
        return status
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all recorded sessions"""
        sessions = []
        
        try:
            for session_file in self.sessions_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    sessions.append({
                        'session_id': session_data['session_id'],
                        'start_time': session_data['start_time'],
                        'duration': session_data['end_time'] - session_data['start_time'],
                        'total_actions': session_data['total_actions'],
                        'sequences_count': len(session_data.get('sequences', [])),
                        'session_type': session_data.get('session_type', 'unknown'),
                        'objective': session_data.get('metadata', {}).get('objective', 'unknown')
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to load session {session_file}: {e}")
            
            # Sort by start time (newest first)
            sessions.sort(key=lambda s: s['start_time'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Session listing failed: {e}")
        
        return sessions
    
    # Helper methods for data management
    def _save_session(self, session: RecordingSession):
        """Save recording session to file"""
        try:
            session_file = self.sessions_dir / f"{session.session_id}.json"
            
            session_data = asdict(session)
            # Convert actions to serializable format
            for sequence in session_data['sequences']:
                sequence['actions'] = [asdict(action) for action in sequence['actions']]
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            self.logger.info(f"Session saved: {session.session_id}")
            
        except Exception as e:
            self.logger.error(f"Session save failed: {e}")
    
    def _load_session(self, session_id: str) -> Optional[RecordingSession]:
        """Load recording session from file"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                return None
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Convert back to objects (simplified for now)
            return RecordingSession(**session_data)
            
        except Exception as e:
            self.logger.error(f"Session load failed: {e}")
            return None
    
    def _save_learned_sequences(self):
        """Save learned sequences to file"""
        try:
            learned_file = self.sessions_dir / "learned_sequences.json"
            with open(learned_file, 'w') as f:
                json.dump(self.learned_sequences, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save learned sequences: {e}")
    
    def _load_learned_sequences(self):
        """Load learned sequences from file"""
        try:
            learned_file = self.sessions_dir / "learned_sequences.json"
            if learned_file.exists():
                with open(learned_file, 'r') as f:
                    self.learned_sequences = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load learned sequences: {e}")
    
    def _update_performance_baselines(self, session: RecordingSession):
        """Update performance baselines from session"""
        for sequence in session.sequences:
            objective = sequence.objective
            metrics = sequence.performance_metrics
            
            if objective not in self.performance_baselines:
                self.performance_baselines[objective] = metrics
            else:
                # Update with running average
                baseline = self.performance_baselines[objective]
                for metric, value in metrics.items():
                    if metric in baseline:
                        baseline[metric] = (baseline[metric] + value) / 2
                    else:
                        baseline[metric] = value
    
    def _extract_reusable_patterns(self, session: RecordingSession) -> Dict[str, Any]:
        """Extract reusable patterns from session"""
        patterns = {}
        
        # Find common action patterns across sequences
        for sequence in session.sequences:
            pattern_key = f"{sequence.objective}_pattern"
            if pattern_key not in patterns:
                patterns[pattern_key] = {
                    'action_types': [a.action_type for a in sequence.actions],
                    'timing_pattern': [a.duration for a in sequence.actions],
                    'success_indicators': []
                }
        
        return patterns
    
    def _estimate_optimized_duration(self, actions: List[GameplayAction]) -> float:
        """Estimate duration of optimized action sequence"""
        if not actions:
            return 0
        
        # Simple estimation based on action types and optimized timing
        total_duration = 0
        for action in actions:
            if action.action_type == 'click':
                total_duration += 0.2
            elif action.action_type == 'move':
                total_duration += 0.1
            elif action.action_type == 'key':
                total_duration += 0.1
            else:
                total_duration += action.duration
        
        return total_duration