"""
Macro System for AI Game Bot
Handles recording, storing, and playback of automation macros
"""

import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from collections import deque

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False
    # PyAutoGUI not available - macro functionality disabled
    pyautogui = None

class MacroSystem:
    """System for recording and playing automation macros with different sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.macros_file = Path("data/macros.json")
        
        # Macro storage with categorization
        self.macros = {
            'user_recorded': {},      # Macros recorded from user gameplay
            'ai_generated': {},       # Macros created by AI
            'ai_learned': {},         # Macros learned by watching user
            'imported': {},           # Macros imported from external sources
            'templates': {}           # Template/reference macros
        }
        
        # Recording state
        self.is_recording = False
        self.current_recording = None
        self.recording_thread = None
        self.recording_mode = 'user_recorded'  # Current recording mode
        
        # AI Learning state
        self.is_watching_for_learning = False
        self.learning_session = None
        self.watching_thread = None
        
        # Recording settings
        self.recording_interval = 0.1  # Record every 100ms
        self.max_recording_duration = 300  # 5 minutes max
        self.action_threshold = 5  # Minimum pixels for movement to count
        
        # Playback settings
        self.playback_speed = 1.0
        self.is_playing = False
        self.playback_thread = None
        
        # AI settings
        self.enable_ai_creation = True  # Enable AI macro creation
        self.enable_ai_learning = True  # Enable AI learning from user actions
        self.auto_optimize_macros = True  # Auto-optimize recorded macros
        
        # Load existing macros
        self._load_macros()
        
        self.logger.info("Enhanced macro system initialized with AI capabilities")
    
    def _load_macros(self):
        """Load macros from storage"""
        try:
            if self.macros_file.exists():
                with open(self.macros_file, 'r') as f:
                    self.macros = json.load(f)
                self.logger.info(f"Loaded {len(self.macros)} macros")
        except Exception as e:
            self.logger.error(f"Failed to load macros: {e}")
            self.macros = {}
    
    def _save_macros(self):
        """Save macros to storage"""
        try:
            self.macros_file.parent.mkdir(exist_ok=True)
            with open(self.macros_file, 'w') as f:
                json.dump(self.macros, f, indent=2)
            self.logger.debug("Macros saved")
        except Exception as e:
            self.logger.error(f"Failed to save macros: {e}")
    
    def start_recording(self, macro_name: str, description: str = "", recording_type: str = "user_recorded") -> str:
        """
        Start recording a new macro
        
        Args:
            macro_name: Name for the macro
            description: Optional description
            recording_type: Type of recording ('user_recorded', 'ai_learned', etc.)
            
        Returns:
            Result message
        """
        try:
            if self.is_recording:
                return "Already recording a macro. Stop current recording first."
            
            if recording_type not in self.macros:
                return f"Invalid recording type '{recording_type}'. Use: {list(self.macros.keys())}"
            
            if macro_name in self.macros[recording_type]:
                return f"Macro '{macro_name}' already exists in {recording_type}. Choose a different name."
            
            # Initialize recording
            self.is_recording = True
            self.recording_mode = recording_type
            self.current_recording = {
                'name': macro_name,
                'description': description,
                'type': recording_type,
                'actions': [],
                'start_time': time.time(),
                'created_by': 'user' if recording_type == 'user_recorded' else 'ai',
                'version': '1.0',
                'tags': [],
                'success_rate': 0.0,
                'last_mouse_pos': (0, 0) if not PYAUTOGUI_AVAILABLE else pyautogui.position(),
                'last_action_time': time.time()
            }
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            result = f"Started recording {recording_type} macro '{macro_name}'"
            self.logger.info(result)
            return result
            
        except Exception as e:
            self.is_recording = False
            error_msg = f"Failed to start recording: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def stop_recording(self) -> str:
        """
        Stop current recording and save the macro
        
        Returns:
            Result message
        """
        try:
            if not self.is_recording:
                return "No recording in progress"
            
            # Stop recording
            self.is_recording = False
            
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
            
            if not self.current_recording:
                return "No recording data available"
            
            # Finalize macro
            macro_name = self.current_recording['name']
            total_duration = time.time() - self.current_recording['start_time']
            
            macro_data = {
                'name': macro_name,
                'description': self.current_recording['description'],
                'actions': self.current_recording['actions'],
                'duration': total_duration,
                'created_at': time.time(),
                'action_count': len(self.current_recording['actions'])
            }
            
            # Save macro to appropriate category
            recording_type = self.current_recording.get('type', 'user_recorded')
            self.macros[recording_type][macro_name] = macro_data
            self._save_macros()
            
            result = f"Stopped recording. Saved {recording_type} macro '{macro_name}' with {len(self.current_recording['actions'])} actions"
            self.logger.info(result)
            
            self.current_recording = None
            return result
            
        except Exception as e:
            self.is_recording = False
            error_msg = f"Failed to stop recording: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _recording_loop(self):
        """Main recording loop"""
        try:
            while self.is_recording and self.current_recording is not None:
                current_time = time.time()
                
                # Check for max duration
                if current_time - self.current_recording['start_time'] > self.max_recording_duration:
                    self.logger.warning("Recording stopped: maximum duration reached")
                    break
                
                # Record mouse position
                if not PYAUTOGUI_AVAILABLE:
                    current_mouse_pos = (0, 0)  # Dummy position
                else:
                    current_mouse_pos = pyautogui.position()
                last_mouse_pos = self.current_recording['last_mouse_pos']
                
                # Check if mouse moved significantly
                if abs(current_mouse_pos[0] - last_mouse_pos[0]) > self.action_threshold or \
                   abs(current_mouse_pos[1] - last_mouse_pos[1]) > self.action_threshold:
                    
                    action = {
                        'type': 'mouse_move',
                        'x': current_mouse_pos[0],
                        'y': current_mouse_pos[1],
                        'timestamp': current_time,
                        'delay': current_time - self.current_recording['last_action_time']
                    }
                    
                    self.current_recording['actions'].append(action)
                    self.current_recording['last_mouse_pos'] = current_mouse_pos
                    self.current_recording['last_action_time'] = current_time
                
                time.sleep(self.recording_interval)
                
        except Exception as e:
            self.logger.error(f"Recording loop error: {e}")
            self.is_recording = False
    
    def play_macro(self, macro_name: str, speed: float = 1.0) -> str:
        """
        Play a recorded macro
        
        Args:
            macro_name: Name of macro to play
            speed: Playback speed multiplier (1.0 = normal speed)
            
        Returns:
            Result message
        """
        try:
            if self.is_playing:
                return "Already playing a macro. Wait for completion."
            
            if macro_name not in self.macros:
                return f"Macro '{macro_name}' not found"
            
            if self.is_recording:
                return "Cannot play macro while recording"
            
            # Start playback
            self.is_playing = True
            self.playback_speed = speed
            
            macro_data = self.macros[macro_name]
            self.playback_thread = threading.Thread(
                target=self._playback_loop, 
                args=(macro_data,)
            )
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
            result = f"Started playing macro '{macro_name}' at {speed}x speed"
            self.logger.info(result)
            return result
            
        except Exception as e:
            self.is_playing = False
            error_msg = f"Failed to play macro: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _playback_loop(self, macro_data: Dict[str, Any]):
        """Main playback loop"""
        try:
            actions = macro_data['actions']
            self.logger.info(f"Playing macro with {len(actions)} actions")
            
            for i, action in enumerate(actions):
                if not self.is_playing:
                    break
                
                # Apply speed adjustment to delay
                delay = action.get('delay', 0) / self.playback_speed
                if delay > 0:
                    time.sleep(min(delay, 5.0))  # Cap delay at 5 seconds
                
                # Execute action
                self._execute_macro_action(action)
                
                self.logger.debug(f"Executed action {i+1}/{len(actions)}: {action['type']}")
            
            self.is_playing = False
            self.logger.info("Macro playback completed")
            
        except Exception as e:
            self.is_playing = False
            self.logger.error(f"Macro playback error: {e}")
    
    def _execute_macro_action(self, action: Dict[str, Any]):
        """Execute a single macro action"""
        try:
            action_type = action['type']
            
            if not PYAUTOGUI_AVAILABLE:
                self.logger.info(f"Demo mode: Would execute {action_type}")
                return
                
            if action_type == 'mouse_move':
                pyautogui.moveTo(action['x'], action['y'], duration=0.1)
            
            elif action_type == 'mouse_click':
                pyautogui.click(
                    action['x'], 
                    action['y'], 
                    button=action.get('button', 'left'),
                    clicks=action.get('clicks', 1)
                )
            
            elif action_type == 'mouse_drag':
                pyautogui.drag(
                    action['end_x'] - action['start_x'],
                    action['end_y'] - action['start_y'],
                    duration=action.get('duration', 0.5),
                    button=action.get('button', 'left')
                )
            
            elif action_type == 'key_press':
                if action.get('key'):
                    pyautogui.press(action['key'])
            
            elif action_type == 'key_combination':
                keys = action.get('keys', [])
                if keys:
                    pyautogui.hotkey(*keys)
            
            elif action_type == 'scroll':
                pyautogui.scroll(
                    action.get('clicks', 1),
                    x=action.get('x'),
                    y=action.get('y')
                )
            
            elif action_type == 'wait':
                time.sleep(action.get('duration', 1.0))
                
        except Exception as e:
            self.logger.error(f"Failed to execute macro action {action}: {e}")
    
    def get_macro_list(self) -> List[str]:
        """Get list of available macro names"""
        return list(self.macros.keys())
    
    def clear_macros(self):
        """Clear all macros"""
        self.macros = {}
        self._save_macros()
        self.logger.info("All macros cleared")
    
    def stop_playback(self) -> str:
        """Stop current macro playback"""
        try:
            if not self.is_playing:
                return "No macro currently playing"
            
            self.is_playing = False
            
            if self.playback_thread:
                self.playback_thread.join(timeout=2.0)
            
            result = "Macro playback stopped"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to stop playback: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def list_macros(self) -> List[str]:
        """Get list of available macro names"""
        return list(self.macros.keys())
    
    def get_macro_info(self, macro_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific macro"""
        if macro_name not in self.macros:
            return None
        
        macro = self.macros[macro_name]
        return {
            'name': macro['name'],
            'description': macro.get('description', ''),
            'duration': macro.get('duration', 0),
            'action_count': macro.get('action_count', 0),
            'created_at': macro.get('created_at', 0)
        }
    
    def delete_macro(self, macro_name: str) -> str:
        """Delete a macro"""
        try:
            if macro_name not in self.macros:
                return f"Macro '{macro_name}' not found"
            
            del self.macros[macro_name]
            self._save_macros()
            
            result = f"Deleted macro '{macro_name}'"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to delete macro: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def create_macro_from_actions(self, macro_name: str, actions: List[Dict[str, Any]], 
                                 description: str = "") -> str:
        """
        Create a macro from a list of actions
        
        Args:
            macro_name: Name for the macro
            actions: List of action dictionaries
            description: Optional description
            
        Returns:
            Result message
        """
        try:
            if macro_name in self.macros:
                return f"Macro '{macro_name}' already exists"
            
            # Calculate total duration
            total_duration = sum(action.get('delay', 0) for action in actions)
            
            macro_data = {
                'name': macro_name,
                'description': description,
                'actions': actions,
                'duration': total_duration,
                'created_at': time.time(),
                'action_count': len(actions)
            }
            
            self.macros[macro_name] = macro_data
            self._save_macros()
            
            result = f"Created macro '{macro_name}' with {len(actions)} actions"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to create macro: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def export_macro(self, macro_name: str, file_path: str) -> str:
        """Export a macro to file"""
        try:
            if macro_name not in self.macros:
                return f"Macro '{macro_name}' not found"
            
            macro_data = self.macros[macro_name]
            
            with open(file_path, 'w') as f:
                json.dump(macro_data, f, indent=2)
            
            result = f"Exported macro '{macro_name}' to {file_path}"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to export macro: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def import_macro(self, file_path: str) -> str:
        """Import a macro from file"""
        try:
            with open(file_path, 'r') as f:
                macro_data = json.load(f)
            
            macro_name = macro_data['name']
            
            if macro_name in self.macros:
                return f"Macro '{macro_name}' already exists"
            
            self.macros[macro_name] = macro_data
            self._save_macros()
            
            result = f"Imported macro '{macro_name}' from {file_path}"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to import macro: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def is_recording_active(self) -> bool:
        """Check if currently recording"""
        return self.is_recording
    
    def is_playback_active(self) -> bool:
        """Check if currently playing back"""
        return self.is_playing
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of macro system"""
        total_macros = sum(len(macros) for macros in self.macros.values())
        return {
            'total_macros': total_macros,
            'is_recording': self.is_recording,
            'is_playing': self.is_playing,
            'is_ai_learning': self.is_watching_for_learning,
            'current_recording': self.current_recording['name'] if self.current_recording else None,
            'recording_duration': time.time() - self.current_recording['start_time'] if self.current_recording else 0,
            'macros_by_type': {macro_type: len(macros) for macro_type, macros in self.macros.items()}
        }
    
    def start_ai_learning_mode(self, session_name: str = None) -> str:
        """
        Start AI learning mode to watch user actions and create macros
        
        Args:
            session_name: Optional name for the learning session
            
        Returns:
            Result message
        """
        try:
            if not self.enable_ai_learning:
                return "AI learning is disabled. Enable it first."
            
            if self.is_watching_for_learning:
                return "AI learning mode is already active. Stop current session first."
            
            session_name = session_name or f"ai_learning_{int(time.time())}"
            
            self.is_watching_for_learning = True
            self.learning_session = {
                'session_name': session_name,
                'start_time': time.time(),
                'observed_actions': [],
                'patterns_identified': [],
                'potential_macros': []
            }
            
            # Start watching thread
            self.watching_thread = threading.Thread(target=self._ai_learning_loop)
            self.watching_thread.daemon = True
            self.watching_thread.start()
            
            result = f"Started AI learning session '{session_name}'"
            self.logger.info(result)
            return result
            
        except Exception as e:
            self.is_watching_for_learning = False
            error_msg = f"Failed to start AI learning mode: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def stop_ai_learning_mode(self) -> str:
        """
        Stop AI learning mode and generate learned macros
        
        Returns:
            Result message with summary
        """
        try:
            if not self.is_watching_for_learning:
                return "AI learning mode is not active"
            
            self.is_watching_for_learning = False
            
            if self.watching_thread:
                self.watching_thread.join(timeout=2.0)
            
            if not self.learning_session:
                return "No learning session data available"
            
            # Analyze learned patterns and create macros
            created_macros = self._analyze_and_create_learned_macros()
            
            session_name = self.learning_session['session_name']
            duration = time.time() - self.learning_session['start_time']
            actions_observed = len(self.learning_session['observed_actions'])
            
            result = f"Stopped AI learning session '{session_name}'. "
            result += f"Observed {actions_observed} actions in {duration:.1f}s. "
            result += f"Created {len(created_macros)} learned macros."
            
            self.logger.info(result)
            self.learning_session = None
            return result
            
        except Exception as e:
            self.is_watching_for_learning = False
            error_msg = f"Failed to stop AI learning mode: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def create_ai_macro(self, macro_name: str, task_description: str, context: Dict[str, Any] = None) -> str:
        """
        Create a macro using AI based on task description
        
        Args:
            macro_name: Name for the new macro
            task_description: Description of what the macro should do
            context: Optional context information
            
        Returns:
            Result message
        """
        try:
            if not self.enable_ai_creation:
                return "AI macro creation is disabled. Enable it first."
            
            if macro_name in self.macros['ai_generated']:
                return f"AI macro '{macro_name}' already exists. Choose a different name."
            
            # Generate macro based on description and context
            generated_macro = self._generate_ai_macro(task_description, context or {})
            
            if not generated_macro:
                return "Failed to generate AI macro from description"
            
            # Create macro data
            macro_data = {
                'name': macro_name,
                'description': task_description,
                'type': 'ai_generated',
                'created_by': 'ai',
                'actions': generated_macro['actions'],
                'confidence': generated_macro.get('confidence', 0.7),
                'context_requirements': generated_macro.get('context_requirements', {}),
                'created_at': time.time(),
                'version': '1.0',
                'tags': ['ai_generated', 'automated'],
                'estimated_duration': generated_macro.get('estimated_duration', 0)
            }
            
            # Save AI generated macro
            self.macros['ai_generated'][macro_name] = macro_data
            self._save_macros()
            
            result = f"Created AI macro '{macro_name}' with {len(generated_macro['actions'])} actions"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to create AI macro: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def get_macros_by_type(self, macro_type: str = None) -> Dict[str, Any]:
        """
        Get macros filtered by type
        
        Args:
            macro_type: Type to filter ('user_recorded', 'ai_generated', etc.)
                       None returns all types
            
        Returns:
            Dictionary of macros
        """
        try:
            if macro_type is None:
                return self.macros
            
            if macro_type not in self.macros:
                return {}
            
            return {macro_type: self.macros[macro_type]}
            
        except Exception as e:
            self.logger.error(f"Failed to get macros by type: {e}")
            return {}
    
    def get_macro_statistics(self) -> Dict[str, Any]:
        """Get statistics about all macro types"""
        try:
            stats = {
                'total_macros': 0,
                'by_type': {},
                'most_recent': None,
                'most_used': None
            }
            
            for macro_type, macros in self.macros.items():
                count = len(macros)
                stats['by_type'][macro_type] = {
                    'count': count,
                    'macros': list(macros.keys())
                }
                stats['total_macros'] += count
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get macro statistics: {e}")
            return {'error': str(e)}
    
    def _ai_learning_loop(self):
        """AI learning observation loop"""
        try:
            while self.is_watching_for_learning and self.learning_session:
                # Observe user actions (screen state, mouse, keyboard)
                current_time = time.time()
                
                # Capture current state for learning
                observation = {
                    'timestamp': current_time,
                    'mouse_pos': (0, 0) if not PYAUTOGUI_AVAILABLE else pyautogui.position(),
                    'screen_region': None,  # Would capture screen in real implementation
                    'action_type': 'observation'
                }
                
                self.learning_session['observed_actions'].append(observation)
                
                # Look for patterns every 10 observations
                if len(self.learning_session['observed_actions']) % 10 == 0:
                    self._identify_action_patterns()
                
                time.sleep(0.2)  # Observe every 200ms
                
        except Exception as e:
            self.logger.error(f"AI learning loop error: {e}")
        finally:
            self.is_watching_for_learning = False
    
    def _identify_action_patterns(self):
        """Identify patterns in observed actions"""
        try:
            if not self.learning_session:
                return
            
            actions = self.learning_session['observed_actions']
            if len(actions) < 3:
                return
            
            # Simple pattern detection - look for repeated sequences
            # In real implementation, this would use more sophisticated ML
            pattern = {
                'type': 'repeated_sequence',
                'confidence': 0.8,
                'action_count': len(actions),
                'identified_at': time.time()
            }
            
            self.learning_session['patterns_identified'].append(pattern)
            
        except Exception as e:
            self.logger.error(f"Pattern identification error: {e}")
    
    def _analyze_and_create_learned_macros(self) -> List[str]:
        """Analyze learning session and create macros from patterns"""
        try:
            if not self.learning_session:
                return []
            
            created_macros = []
            patterns = self.learning_session['patterns_identified']
            
            for i, pattern in enumerate(patterns):
                macro_name = f"learned_pattern_{i+1}_{int(time.time())}"
                
                # Create macro from pattern
                macro_data = {
                    'name': macro_name,
                    'description': f"Learned pattern from AI observation",
                    'type': 'ai_learned',
                    'created_by': 'ai_learning',
                    'actions': self._convert_pattern_to_actions(pattern),
                    'confidence': pattern.get('confidence', 0.7),
                    'learning_session': self.learning_session['session_name'],
                    'created_at': time.time(),
                    'version': '1.0',
                    'tags': ['ai_learned', 'pattern_based']
                }
                
                self.macros['ai_learned'][macro_name] = macro_data
                created_macros.append(macro_name)
            
            if created_macros:
                self._save_macros()
            
            return created_macros
            
        except Exception as e:
            self.logger.error(f"Failed to analyze and create learned macros: {e}")
            return []
    
    def _generate_ai_macro(self, task_description: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a macro using AI based on task description"""
        try:
            # This would use actual AI/ML in real implementation
            # For now, create a simple template based on common tasks
            
            actions = []
            confidence = 0.7
            
            # Simple rule-based generation based on keywords
            if 'chest' in task_description.lower():
                actions = [
                    {'type': 'click', 'x': 400, 'y': 300, 'duration': 0.1},
                    {'type': 'wait', 'duration': 1.0},
                    {'type': 'click', 'x': 500, 'y': 400, 'duration': 0.1}
                ]
                confidence = 0.8
            elif 'egg' in task_description.lower():
                actions = [
                    {'type': 'click', 'x': 350, 'y': 250, 'duration': 0.1},
                    {'type': 'wait', 'duration': 0.5},
                    {'type': 'click', 'x': 350, 'y': 250, 'duration': 0.1}
                ]
                confidence = 0.8
            else:
                # Generic action sequence
                actions = [
                    {'type': 'click', 'x': 400, 'y': 300, 'duration': 0.1},
                    {'type': 'wait', 'duration': 0.5}
                ]
                confidence = 0.5
            
            return {
                'actions': actions,
                'confidence': confidence,
                'context_requirements': context,
                'estimated_duration': sum(a.get('duration', 0.1) for a in actions)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate AI macro: {e}")
            return None
    
    def _convert_pattern_to_actions(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert an identified pattern into executable actions"""
        try:
            # Simple conversion - in real implementation this would be more sophisticated
            actions = [
                {'type': 'click', 'x': 400, 'y': 300, 'duration': 0.1},
                {'type': 'wait', 'duration': 0.5}
            ]
            return actions
            
        except Exception as e:
            self.logger.error(f"Failed to convert pattern to actions: {e}")
            return []
    
    def record_task_to_macro(self, task_name: str, description: str = "") -> str:
        """
        Start recording a task that will be converted to a downloadable macro
        
        Args:
            task_name: Name for the task/macro
            description: Description of what the task does
            
        Returns:
            Result message
        """
        try:
            # Start recording with task metadata
            result = self.start_recording(task_name, description, "user_recorded")
            
            if "Started recording" in result:
                # Add task-specific metadata
                if self.current_recording:
                    self.current_recording['is_task_recording'] = True
                    self.current_recording['task_metadata'] = {
                        'recorded_for_export': True,
                        'export_format': 'downloadable',
                        'recording_purpose': 'task_automation'
                    }
                result = f"Started task recording '{task_name}'. Perform your gameplay actions now, then call stop_task_recording()."
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to start task recording: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def stop_task_recording(self) -> Dict[str, Any]:
        """
        Stop task recording and prepare macro for export
        
        Returns:
            Dictionary with macro data and export information
        """
        try:
            if not self.is_recording:
                return {"error": "No task recording in progress"}
            
            if not self.current_recording or not self.current_recording.get('is_task_recording'):
                return {"error": "Current recording is not a task recording"}
            
            # Stop the recording
            stop_result = self.stop_recording()
            
            if "Saved" not in stop_result:
                return {"error": f"Failed to save recording: {stop_result}"}
            
            # Get the recorded task data
            task_name = self.current_recording['name'] if self.current_recording else "unknown_task"
            
            # Find the macro in user_recorded category
            if task_name in self.macros['user_recorded']:
                macro_data = self.macros['user_recorded'][task_name]
                
                # Generate exportable formats
                export_data = self._generate_exportable_macro(macro_data)
                
                return {
                    "success": True,
                    "task_name": task_name,
                    "macro_data": macro_data,
                    "export_formats": export_data,
                    "download_ready": True,
                    "message": f"Task '{task_name}' recorded successfully. Ready for download."
                }
            else:
                return {"error": f"Task macro '{task_name}' not found after recording"}
            
        except Exception as e:
            error_msg = f"Failed to stop task recording: {e}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def export_task_as_file(self, task_name: str, export_format: str = "json") -> Dict[str, Any]:
        """
        Export a recorded task as a downloadable file
        
        Args:
            task_name: Name of the task to export
            export_format: Format for export ('json', 'python', 'javascript', 'ahk')
            
        Returns:
            Dictionary with file content and metadata
        """
        try:
            # Find the task macro
            macro_data = None
            for macro_type, macros in self.macros.items():
                if task_name in macros:
                    macro_data = macros[task_name]
                    break
            
            if not macro_data:
                return {"error": f"Task '{task_name}' not found"}
            
            # Generate file content based on format
            if export_format.lower() == "json":
                file_content = self._generate_json_export(macro_data)
                file_extension = "json"
                mime_type = "application/json"
            
            elif export_format.lower() == "python":
                file_content = self._generate_python_script(macro_data)
                file_extension = "py"
                mime_type = "text/plain"
            
            elif export_format.lower() == "javascript":
                file_content = self._generate_javascript_script(macro_data)
                file_extension = "js"
                mime_type = "text/plain"
            
            elif export_format.lower() == "ahk":
                file_content = self._generate_autohotkey_script(macro_data)
                file_extension = "ahk"
                mime_type = "text/plain"
            
            elif export_format.lower() == "ahk2":
                file_content = self._generate_autohotkey_v2_script(macro_data)
                file_extension = "ahk"
                mime_type = "text/plain"
            
            else:
                return {"error": f"Unsupported export format: {export_format}"}
            
            # Create filename
            safe_name = "".join(c for c in task_name if c.isalnum() or c in ('-', '_')).strip()
            filename = f"{safe_name}_macro.{file_extension}"
            
            return {
                "success": True,
                "filename": filename,
                "content": file_content,
                "mime_type": mime_type,
                "size": len(file_content.encode('utf-8')),
                "format": export_format,
                "task_name": task_name
            }
            
        except Exception as e:
            error_msg = f"Failed to export task as file: {e}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def _generate_exportable_macro(self, macro_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate multiple export formats for a macro"""
        try:
            exports = {}
            
            # JSON format (raw macro data)
            exports['json'] = json.dumps(macro_data, indent=2)
            
            # Python script format
            exports['python'] = self._generate_python_script(macro_data)
            
            # JavaScript format
            exports['javascript'] = self._generate_javascript_script(macro_data)
            
            # AutoHotkey format
            exports['autohotkey'] = self._generate_autohotkey_script(macro_data)
            
            # AutoHotkey v2 format
            exports['autohotkey_v2'] = self._generate_autohotkey_v2_script(macro_data)
            
            return exports
            
        except Exception as e:
            self.logger.error(f"Failed to generate exportable formats: {e}")
            return {}
    
    def _generate_json_export(self, macro_data: Dict[str, Any]) -> str:
        """Generate JSON export of macro data"""
        try:
            export_data = {
                "macro_name": macro_data.get('name', 'Unnamed'),
                "description": macro_data.get('description', ''),
                "created_at": macro_data.get('created_at', time.time()),
                "format_version": "1.0",
                "exported_by": "AI Game Bot",
                "actions": macro_data.get('actions', []),
                "metadata": {
                    "total_actions": len(macro_data.get('actions', [])),
                    "estimated_duration": macro_data.get('duration', 0),
                    "macro_type": macro_data.get('type', 'user_recorded')
                }
            }
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON export: {e}")
            return "{}"
    
    def _generate_python_script(self, macro_data: Dict[str, Any]) -> str:
        """Generate Python script that can execute the macro"""
        try:
            script_lines = [
                "#!/usr/bin/env python3",
                "\"\"\"",
                f"Macro: {macro_data.get('name', 'Unnamed')}",
                f"Description: {macro_data.get('description', 'No description')}",
                f"Generated by: AI Game Bot",
                f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(macro_data.get('created_at', time.time())))}",
                "\"\"\"",
                "",
                "import time",
                "import pyautogui",
                "",
                "def execute_macro():",
                "    \"\"\"Execute the recorded macro actions\"\"\"",
                "    print(f\"Starting macro: {macro_data.get('name', 'Unnamed')}\")",
                ""
            ]
            
            actions = macro_data.get('actions', [])
            for i, action in enumerate(actions):
                action_type = action.get('type', 'unknown')
                
                if action_type == 'mouse_move':
                    script_lines.append(f"    # Action {i+1}: Mouse move")
                    script_lines.append(f"    pyautogui.moveTo({action.get('x', 0)}, {action.get('y', 0)})")
                
                elif action_type == 'mouse_click':
                    script_lines.append(f"    # Action {i+1}: Mouse click")
                    button = action.get('button', 'left')
                    clicks = action.get('clicks', 1)
                    script_lines.append(f"    pyautogui.click({action.get('x', 0)}, {action.get('y', 0)}, button='{button}', clicks={clicks})")
                
                elif action_type == 'key_press':
                    script_lines.append(f"    # Action {i+1}: Key press")
                    key = action.get('key', 'space')
                    script_lines.append(f"    pyautogui.press('{key}')")
                
                elif action_type == 'wait':
                    script_lines.append(f"    # Action {i+1}: Wait")
                    duration = action.get('duration', 1.0)
                    script_lines.append(f"    time.sleep({duration})")
                
                # Add delay between actions
                delay = action.get('delay', 0.1)
                if delay > 0 and i < len(actions) - 1:
                    script_lines.append(f"    time.sleep({delay})")
                
                script_lines.append("")
            
            script_lines.extend([
                "    print(\"Macro execution completed\")",
                "",
                "if __name__ == '__main__':",
                "    try:",
                "        execute_macro()",
                "    except KeyboardInterrupt:",
                "        print(\"Macro interrupted by user\")",
                "    except Exception as e:",
                "        print(f\"Error executing macro: {e}\")"
            ])
            
            return "\n".join(script_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to generate Python script: {e}")
            return "# Error generating Python script"
    
    def _generate_javascript_script(self, macro_data: Dict[str, Any]) -> str:
        """Generate JavaScript script for the macro"""
        try:
            script_lines = [
                "/*",
                f" * Macro: {macro_data.get('name', 'Unnamed')}",
                f" * Description: {macro_data.get('description', 'No description')}",
                f" * Generated by: AI Game Bot",
                f" * Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(macro_data.get('created_at', time.time())))}",
                " */",
                "",
                "// Note: This JavaScript version requires a browser automation library",
                "// such as Puppeteer, Selenium, or similar to execute mouse/keyboard actions",
                "",
                "async function executeMacro() {",
                f"    console.log('Starting macro: {macro_data.get('name', 'Unnamed')}');",
                ""
            ]
            
            actions = macro_data.get('actions', [])
            for i, action in enumerate(actions):
                action_type = action.get('type', 'unknown')
                
                if action_type == 'mouse_move':
                    script_lines.append(f"    // Action {i+1}: Mouse move")
                    script_lines.append(f"    await moveMouse({action.get('x', 0)}, {action.get('y', 0)});")
                
                elif action_type == 'mouse_click':
                    script_lines.append(f"    // Action {i+1}: Mouse click")
                    script_lines.append(f"    await clickMouse({action.get('x', 0)}, {action.get('y', 0)});")
                
                elif action_type == 'key_press':
                    script_lines.append(f"    // Action {i+1}: Key press")
                    key = action.get('key', 'space')
                    script_lines.append(f"    await pressKey('{key}');")
                
                elif action_type == 'wait':
                    script_lines.append(f"    // Action {i+1}: Wait")
                    duration = action.get('duration', 1.0) * 1000  # Convert to milliseconds
                    script_lines.append(f"    await sleep({int(duration)});")
                
                # Add delay between actions
                delay = action.get('delay', 0.1) * 1000
                if delay > 0 and i < len(actions) - 1:
                    script_lines.append(f"    await sleep({int(delay)});")
                
                script_lines.append("")
            
            script_lines.extend([
                "    console.log('Macro execution completed');",
                "}",
                "",
                "// Helper functions (implement based on your automation library)",
                "async function sleep(ms) {",
                "    return new Promise(resolve => setTimeout(resolve, ms));",
                "}",
                "",
                "async function moveMouse(x, y) {",
                "    // Implement mouse movement",
                "}",
                "",
                "async function clickMouse(x, y) {",
                "    // Implement mouse click",
                "}",
                "",
                "async function pressKey(key) {",
                "    // Implement key press",
                "}",
                "",
                "// Execute the macro",
                "executeMacro().catch(console.error);"
            ])
            
            return "\n".join(script_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to generate JavaScript script: {e}")
            return "// Error generating JavaScript script"
    
    def _generate_autohotkey_script(self, macro_data: Dict[str, Any]) -> str:
        """Generate AutoHotkey script for the macro"""
        try:
            script_lines = [
                "; AutoHotkey Macro Script",
                f"; Macro: {macro_data.get('name', 'Unnamed')}",
                f"; Description: {macro_data.get('description', 'No description')}",
                f"; Generated by: AI Game Bot",
                f"; Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(macro_data.get('created_at', time.time())))}",
                "",
                "#NoEnv",
                "#SingleInstance Force",
                "SendMode Input",
                "SetWorkingDir %A_ScriptDir%",
                "",
                "; Press F1 to start the macro",
                "F1::",
                f"MsgBox, Starting macro: {macro_data.get('name', 'Unnamed')}",
                ""
            ]
            
            actions = macro_data.get('actions', [])
            for i, action in enumerate(actions):
                action_type = action.get('type', 'unknown')
                
                if action_type == 'mouse_move':
                    script_lines.append(f"; Action {i+1}: Mouse move")
                    script_lines.append(f"MouseMove, {action.get('x', 0)}, {action.get('y', 0)}")
                
                elif action_type == 'mouse_click':
                    script_lines.append(f"; Action {i+1}: Mouse click")
                    button = action.get('button', 'Left')
                    clicks = action.get('clicks', 1)
                    script_lines.append(f"Click, {action.get('x', 0)}, {action.get('y', 0)}, {button}, {clicks}")
                
                elif action_type == 'key_press':
                    script_lines.append(f"; Action {i+1}: Key press")
                    key = action.get('key', 'Space')
                    script_lines.append(f"Send, {{{key}}}")
                
                elif action_type == 'wait':
                    script_lines.append(f"; Action {i+1}: Wait")
                    duration = int(action.get('duration', 1.0) * 1000)  # Convert to milliseconds
                    script_lines.append(f"Sleep, {duration}")
                
                # Add delay between actions
                delay = int(action.get('delay', 0.1) * 1000)
                if delay > 0 and i < len(actions) - 1:
                    script_lines.append(f"Sleep, {delay}")
                
                script_lines.append("")
            
            script_lines.extend([
                "MsgBox, Macro execution completed",
                "return",
                "",
                "; Press F2 to exit",
                "F2::",
                "ExitApp"
            ])
            
            return "\n".join(script_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to generate AutoHotkey script: {e}")
            return "; Error generating AutoHotkey script"
    
    def _generate_autohotkey_v2_script(self, macro_data: Dict[str, Any]) -> str:
        """Generate AutoHotkey v2 script for the macro"""
        try:
            script_lines = [
                "; AutoHotkey v2 Macro Script",
                f"; Macro: {macro_data.get('name', 'Unnamed')}",
                f"; Description: {macro_data.get('description', 'No description')}",
                f"; Generated by: AI Game Bot",
                f"; Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(macro_data.get('created_at', time.time())))}",
                "",
                "; AutoHotkey v2 syntax",
                "#Requires AutoHotkey v2.0",
                "#SingleInstance Force",
                "",
                "; Press F1 to start the macro",
                "F1:: {",
                f'    MsgBox("Starting macro: {macro_data.get("name", "Unnamed")}", "AI Game Bot")',
                "    ExecuteMacro()",
                "}",
                "",
                "; Press F2 to exit",
                "F2:: {",
                "    ExitApp()",
                "}",
                "",
                "ExecuteMacro() {",
                ""
            ]
            
            actions = macro_data.get('actions', [])
            for i, action in enumerate(actions):
                action_type = action.get('type', 'unknown')
                
                if action_type == 'mouse_move':
                    script_lines.append(f"    ; Action {i+1}: Mouse move")
                    script_lines.append(f"    MouseMove({action.get('x', 0)}, {action.get('y', 0)})")
                
                elif action_type == 'mouse_click':
                    script_lines.append(f"    ; Action {i+1}: Mouse click")
                    button = action.get('button', 'Left')
                    clicks = action.get('clicks', 1)
                    script_lines.append(f"    Click({action.get('x', 0)}, {action.get('y', 0)}, \"{button}\", {clicks})")
                
                elif action_type == 'key_press':
                    script_lines.append(f"    ; Action {i+1}: Key press")
                    key = action.get('key', 'Space')
                    script_lines.append(f"    Send(\"{{{key}}}\")")
                
                elif action_type == 'wait':
                    script_lines.append(f"    ; Action {i+1}: Wait")
                    duration = int(action.get('duration', 1.0) * 1000)  # Convert to milliseconds
                    script_lines.append(f"    Sleep({duration})")
                
                # Add delay between actions
                delay = int(action.get('delay', 0.1) * 1000)
                if delay > 0 and i < len(actions) - 1:
                    script_lines.append(f"    Sleep({delay})")
                
                script_lines.append("")
            
            script_lines.extend([
                '    MsgBox("Macro execution completed", "AI Game Bot")',
                "}"
            ])
            
            return "\n".join(script_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to generate AutoHotkey v2 script: {e}")
            return "; Error generating AutoHotkey v2 script"
