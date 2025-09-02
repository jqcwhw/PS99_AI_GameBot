"""
PS99 Action Recorder - Python Implementation
Real action recording and macro generation for PS99 automation
"""

import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import pyautogui
import keyboard
import mouse
from datetime import datetime

@dataclass
class RecordedAction:
    """Represents a recorded user action"""
    timestamp: float
    action_type: str  # 'mouse_move', 'mouse_click', 'key_press', 'key_release', 'wait'
    parameters: Dict[str, Any]
    relative_time: float  # Time since recording started
    window_info: Optional[Dict[str, Any]] = None

@dataclass
class WindowInfo:
    """Information about the game window"""
    title: str
    pid: int
    position: Tuple[int, int]
    size: Tuple[int, int]
    is_active: bool

class PS99ActionRecorder:
    """Real action recorder for PS99 game automation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Recording state
        self.recording = False
        self.recording_start_time = 0
        self.last_action_time = 0
        self.recorded_actions = []
        
        # Configuration
        self.capture_interval = 0.1  # 100ms
        self.min_mouse_move = 5      # Minimum pixels to record movement
        self.max_recording_time = 3600  # 1 hour max
        
        # Window tracking
        self.target_window = None
        self.window_info = None
        
        # Mouse tracking
        self.last_mouse_pos = None
        self.mouse_tracking = False
        
        # Output directory
        self.output_dir = Path("data/recorded_actions")
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info("PS99 Action Recorder initialized")
    
    def start_recording(self, target_window_title: str = "Roblox") -> bool:
        """Start recording user actions"""
        if self.recording:
            self.logger.warning("Recording already in progress")
            return False
        
        try:
            # Find target window
            self.window_info = self._find_target_window(target_window_title)
            if not self.window_info:
                self.logger.error(f"Could not find window with title: {target_window_title}")
                return False
            
            # Reset recording state
            self.recorded_actions.clear()
            self.recording_start_time = time.time()
            self.last_action_time = self.recording_start_time
            self.recording = True
            
            # Start monitoring threads
            self._start_monitoring()
            
            self.logger.info(f"Started recording actions for window: {self.window_info.title}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self) -> bool:
        """Stop recording and finalize actions"""
        if not self.recording:
            self.logger.warning("No recording in progress")
            return False
        
        self.recording = False
        self._stop_monitoring()
        
        # Add final timing info
        total_duration = time.time() - self.recording_start_time
        
        self.logger.info(f"Recording stopped. Duration: {total_duration:.2f}s, Actions: {len(self.recorded_actions)}")
        return True
    
    def _find_target_window(self, title: str) -> Optional[WindowInfo]:
        """Find the target game window"""
        try:
            # Use pyautogui to get window list (basic implementation)
            # In a real implementation, you'd use platform-specific window APIs
            
            # For now, create a mock window info based on current screen
            current_window = pyautogui.getActiveWindow()
            if current_window and title.lower() in current_window.title.lower():
                return WindowInfo(
                    title=current_window.title,
                    pid=0,  # Would need platform-specific code to get PID
                    position=(current_window.left, current_window.top),
                    size=(current_window.width, current_window.height),
                    is_active=True
                )
            
            # Fallback: assume game window covers full screen
            screen_size = pyautogui.size()
            return WindowInfo(
                title="Roblox Game Window",
                pid=0,
                position=(0, 0),
                size=(screen_size.width, screen_size.height),
                is_active=True
            )
            
        except Exception as e:
            self.logger.error(f"Error finding target window: {e}")
            return None
    
    def _start_monitoring(self):
        """Start monitoring mouse and keyboard"""
        # Start mouse monitoring
        self.mouse_tracking = True
        mouse_thread = threading.Thread(target=self._monitor_mouse, daemon=True)
        mouse_thread.start()
        
        # Start keyboard monitoring
        keyboard_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
        keyboard_thread.start()
    
    def _stop_monitoring(self):
        """Stop monitoring threads"""
        self.mouse_tracking = False
        # Keyboard monitoring will stop when recording flag is False
    
    def _monitor_mouse(self):
        """Monitor mouse movements and clicks"""
        while self.recording and self.mouse_tracking:
            try:
                current_pos = pyautogui.position()
                
                # Check for mouse movement
                if self.last_mouse_pos is not None:
                    distance = ((current_pos[0] - self.last_mouse_pos[0])**2 + 
                              (current_pos[1] - self.last_mouse_pos[1])**2)**0.5
                    
                    if distance >= self.min_mouse_move:
                        self._record_action(
                            action_type='mouse_move',
                            parameters={
                                'x': current_pos[0],
                                'y': current_pos[1],
                                'from_x': self.last_mouse_pos[0],
                                'from_y': self.last_mouse_pos[1]
                            }
                        )
                
                self.last_mouse_pos = current_pos
                
                # Monitor for mouse clicks using mouse library
                if mouse.is_pressed('left'):
                    self._record_action(
                        action_type='mouse_click',
                        parameters={
                            'button': 'left',
                            'x': current_pos[0],
                            'y': current_pos[1]
                        }
                    )
                    time.sleep(0.1)  # Debounce
                
                elif mouse.is_pressed('right'):
                    self._record_action(
                        action_type='mouse_click',
                        parameters={
                            'button': 'right',
                            'x': current_pos[0],
                            'y': current_pos[1]
                        }
                    )
                    time.sleep(0.1)  # Debounce
                
                time.sleep(self.capture_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring mouse: {e}")
                time.sleep(1.0)
    
    def _monitor_keyboard(self):
        """Monitor keyboard presses"""
        def on_key_event(event):
            if not self.recording:
                return
            
            try:
                if event.event_type == keyboard.KEY_DOWN:
                    self._record_action(
                        action_type='key_press',
                        parameters={
                            'key': event.name,
                            'scan_code': event.scan_code
                        }
                    )
                elif event.event_type == keyboard.KEY_UP:
                    self._record_action(
                        action_type='key_release',
                        parameters={
                            'key': event.name,
                            'scan_code': event.scan_code
                        }
                    )
            except Exception as e:
                self.logger.error(f"Error recording keyboard event: {e}")
        
        # Hook keyboard events
        keyboard.hook(on_key_event)
        
        # Keep thread alive while recording
        while self.recording:
            time.sleep(0.1)
        
        # Unhook when done
        keyboard.unhook_all()
    
    def _record_action(self, action_type: str, parameters: Dict[str, Any]):
        """Record a single action"""
        current_time = time.time()
        relative_time = current_time - self.recording_start_time
        
        action = RecordedAction(
            timestamp=current_time,
            action_type=action_type,
            parameters=parameters,
            relative_time=relative_time,
            window_info=asdict(self.window_info) if self.window_info else None
        )
        
        self.recorded_actions.append(action)
        self.last_action_time = current_time
        
        # Log important actions
        if action_type in ['mouse_click', 'key_press']:
            self.logger.debug(f"Recorded {action_type}: {parameters}")
    
    def save_recording(self, filename: Optional[str] = None) -> str:
        """Save recorded actions to file"""
        if not self.recorded_actions:
            raise ValueError("No actions recorded")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ps99_recording_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare data for saving
        recording_data = {
            'meta': {
                'recording_start': self.recording_start_time,
                'total_duration': time.time() - self.recording_start_time,
                'total_actions': len(self.recorded_actions),
                'window_info': asdict(self.window_info) if self.window_info else None,
                'created_at': datetime.now().isoformat()
            },
            'actions': [asdict(action) for action in self.recorded_actions]
        }
        
        with open(filepath, 'w') as f:
            json.dump(recording_data, f, indent=2)
        
        self.logger.info(f"Recording saved to: {filepath}")
        return str(filepath)
    
    def load_recording(self, filepath: str) -> bool:
        """Load a previously saved recording"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load actions
            self.recorded_actions = [
                RecordedAction(**action_data) 
                for action_data in data['actions']
            ]
            
            # Load metadata
            meta = data.get('meta', {})
            self.recording_start_time = meta.get('recording_start', 0)
            
            self.logger.info(f"Loaded recording from: {filepath} ({len(self.recorded_actions)} actions)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading recording: {e}")
            return False
    
    def generate_python_macro(self, output_file: Optional[str] = None) -> str:
        """Generate Python macro code from recorded actions"""
        if not self.recorded_actions:
            raise ValueError("No actions recorded")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"ps99_macro_{timestamp}.py"
        
        # Generate Python code
        code_lines = [
            "#!/usr/bin/env python3",
            '"""',
            f"Generated PS99 Macro - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total actions: {len(self.recorded_actions)}",
            '"""',
            "",
            "import pyautogui",
            "import time",
            "",
            "def run_ps99_macro():",
            '    """Execute recorded PS99 actions"""',
            "    print('Starting PS99 macro execution...')",
            "    ",
        ]
        
        last_time = 0
        for i, action in enumerate(self.recorded_actions):
            # Calculate delay since last action
            if i > 0:
                delay = action.relative_time - last_time
                if delay > 0.05:  # Only add delays > 50ms
                    code_lines.append(f"    time.sleep({delay:.3f})")
            
            # Generate action code
            if action.action_type == 'mouse_move':
                x, y = action.parameters['x'], action.parameters['y']
                code_lines.append(f"    pyautogui.moveTo({x}, {y})")
            
            elif action.action_type == 'mouse_click':
                x, y = action.parameters['x'], action.parameters['y']
                button = action.parameters['button']
                code_lines.append(f"    pyautogui.click({x}, {y}, button='{button}')")
            
            elif action.action_type == 'key_press':
                key = action.parameters['key']
                code_lines.append(f"    pyautogui.keyDown('{key}')")
            
            elif action.action_type == 'key_release':
                key = action.parameters['key']
                code_lines.append(f"    pyautogui.keyUp('{key}')")
            
            last_time = action.relative_time
        
        code_lines.extend([
            "",
            "    print('PS99 macro execution completed!')",
            "",
            "if __name__ == '__main__':",
            "    run_ps99_macro()"
        ])
        
        # Write to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write('\n'.join(code_lines))
        
        self.logger.info(f"Python macro generated: {output_path}")
        return str(output_path)
    
    def generate_autohotkey_macro(self, output_file: Optional[str] = None) -> str:
        """Generate AutoHotkey macro from recorded actions"""
        if not self.recorded_actions:
            raise ValueError("No actions recorded")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"ps99_macro_{timestamp}.ahk"
        
        # Generate AutoHotkey code
        ahk_lines = [
            "#SingleInstance Force",
            "#NoEnv",
            "SetWorkingDir %A_ScriptDir%",
            "SendMode Input",
            "SetBatchLines -1",
            "CoordMode, Mouse, Screen",
            "",
            f"; Generated PS99 Macro - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"; Total actions: {len(self.recorded_actions)}",
            "",
            "F1::StartMacro()",
            "F2::ExitApp",
            "",
            "StartMacro():",
            "{"
        ]
        
        last_time = 0
        for i, action in enumerate(self.recorded_actions):
            # Calculate delay since last action
            if i > 0:
                delay = int((action.relative_time - last_time) * 1000)  # Convert to ms
                if delay > 50:  # Only add delays > 50ms
                    ahk_lines.append(f"    Sleep, {delay}")
            
            # Generate action code
            if action.action_type == 'mouse_move':
                x, y = action.parameters['x'], action.parameters['y']
                ahk_lines.append(f"    MouseMove, {x}, {y}, 0")
            
            elif action.action_type == 'mouse_click':
                x, y = action.parameters['x'], action.parameters['y']
                button = action.parameters['button']
                click_type = "Left" if button == "left" else "Right"
                ahk_lines.append(f"    Click, {x}, {y}, {click_type}")
            
            elif action.action_type == 'key_press':
                key = action.parameters['key']
                # Map common keys
                ahk_key = key.upper() if len(key) == 1 else key
                ahk_lines.append(f"    Send, {{{ahk_key} down}}")
            
            elif action.action_type == 'key_release':
                key = action.parameters['key']
                ahk_key = key.upper() if len(key) == 1 else key
                ahk_lines.append(f"    Send, {{{ahk_key} up}}")
            
            last_time = action.relative_time
        
        ahk_lines.extend([
            "    MsgBox, PS99 macro execution completed!",
            "    return",
            "}"
        ])
        
        # Write to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write('\n'.join(ahk_lines))
        
        self.logger.info(f"AutoHotkey macro generated: {output_path}")
        return str(output_path)
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get statistics about current recording"""
        total_duration = 0
        if self.recording and self.recording_start_time > 0:
            total_duration = time.time() - self.recording_start_time
        elif self.recorded_actions:
            total_duration = self.recorded_actions[-1].relative_time
        
        action_counts = {}
        for action in self.recorded_actions:
            action_type = action.action_type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        return {
            'recording': self.recording,
            'total_duration': total_duration,
            'total_actions': len(self.recorded_actions),
            'action_breakdown': action_counts,
            'window_info': asdict(self.window_info) if self.window_info else None
        }