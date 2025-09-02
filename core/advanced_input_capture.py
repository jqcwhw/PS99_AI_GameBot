"""
Advanced Input Capture System
Based on SerpentAI sneakysnek for cross-platform input recording
"""

import time
import logging
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import json
from pathlib import Path

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False

class KeyboardEvents(Enum):
    DOWN = "down"
    UP = "up"

class MouseEvents(Enum):
    CLICK = "click"
    SCROLL = "scroll"
    MOVE = "move"

class MouseButtons(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"

@dataclass
class KeyboardEvent:
    """Keyboard event object"""
    event: KeyboardEvents
    key: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'keyboard',
            'event': self.event.value,
            'key': self.key,
            'timestamp': self.timestamp
        }

@dataclass
class MouseEvent:
    """Mouse event object"""
    event: MouseEvents
    button: Optional[MouseButtons]
    x: int
    y: int
    direction: Optional[str] = None
    velocity: int = 1
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'mouse',
            'event': self.event.value,
            'button': self.button.value if self.button else None,
            'x': self.x,
            'y': self.y,
            'direction': self.direction,
            'velocity': self.velocity,
            'timestamp': self.timestamp
        }

class InputRecorder:
    """Advanced input recording system inspired by SerpentAI sneakysnek"""
    
    def __init__(self, buffer_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        
        # Event storage
        self.event_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.callback = None
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'keyboard_events': 0,
            'mouse_events': 0,
            'recording_start_time': 0,
            'recording_duration': 0
        }
        
        # Input monitoring settings
        self.monitor_interval = 0.01  # 10ms polling
        self.last_mouse_pos = (0, 0)
        self.mouse_move_threshold = 5  # Minimum pixel movement to record
        
        self.logger.info("Advanced Input Recorder initialized")
    
    @classmethod
    def record(cls, callback: Callable[[Any], None], **kwargs) -> 'InputRecorder':
        """
        Create and start recording with callback (SerpentAI style)
        
        Args:
            callback: Function to call with each captured event
            **kwargs: Additional arguments for recorder initialization
            
        Returns:
            InputRecorder instance
        """
        recorder = cls(**kwargs)
        recorder.start_recording(callback)
        return recorder
    
    def start_recording(self, callback: Optional[Callable[[Any], None]] = None):
        """Start input recording"""
        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return
        
        self.callback = callback
        self.is_recording = True
        
        # Reset statistics
        self.stats['total_events'] = 0
        self.stats['keyboard_events'] = 0
        self.stats['mouse_events'] = 0
        self.stats['recording_start_time'] = time.time()
        
        # Clear buffer
        with self.buffer_lock:
            self.event_buffer.clear()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        
        self.logger.info("Input recording started")
    
    def _recording_loop(self):
        """Main recording loop for capturing input events"""
        while self.is_recording:
            try:
                current_time = time.time()
                
                # Monitor mouse position
                if PYAUTOGUI_AVAILABLE:
                    try:
                        current_pos = pyautogui.position()
                        
                        # Check for mouse movement
                        if abs(current_pos.x - self.last_mouse_pos[0]) > self.mouse_move_threshold or \
                           abs(current_pos.y - self.last_mouse_pos[1]) > self.mouse_move_threshold:
                            
                            mouse_event = MouseEvent(
                                event=MouseEvents.MOVE,
                                button=None,
                                x=current_pos.x,
                                y=current_pos.y,
                                timestamp=current_time
                            )
                            
                            self._process_event(mouse_event)
                            self.last_mouse_pos = (current_pos.x, current_pos.y)
                    
                    except Exception as e:
                        # PyAutoGUI may fail in some environments
                        pass
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in recording loop: {e}")
                time.sleep(0.1)
        
        # Update final statistics
        self.stats['recording_duration'] = time.time() - self.stats['recording_start_time']
        self.logger.info(f"Recording completed. Total events: {self.stats['total_events']}")
    
    def _process_event(self, event):
        """Process and store a captured event"""
        try:
            # Add to buffer
            with self.buffer_lock:
                self.event_buffer.append(event)
            
            # Update statistics
            self.stats['total_events'] += 1
            if isinstance(event, KeyboardEvent):
                self.stats['keyboard_events'] += 1
            elif isinstance(event, MouseEvent):
                self.stats['mouse_events'] += 1
            
            # Call callback if provided
            if self.callback:
                try:
                    self.callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
    
    def simulate_keyboard_event(self, key: str, event_type: KeyboardEvents):
        """Simulate a keyboard event for testing"""
        event = KeyboardEvent(
            event=event_type,
            key=key,
            timestamp=time.time()
        )
        self._process_event(event)
    
    def simulate_mouse_event(self, event_type: MouseEvents, x: int = 0, y: int = 0, 
                           button: Optional[MouseButtons] = None):
        """Simulate a mouse event for testing"""
        event = MouseEvent(
            event=event_type,
            button=button,
            x=x,
            y=y,
            timestamp=time.time()
        )
        self._process_event(event)
    
    def stop_recording(self):
        """Stop input recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        self.stats['recording_duration'] = time.time() - self.stats['recording_start_time']
        self.logger.info("Input recording stopped")
    
    def stop(self):
        """Alias for stop_recording (SerpentAI compatibility)"""
        self.stop_recording()
    
    def get_events(self, count: Optional[int] = None) -> List[Any]:
        """
        Get recorded events
        
        Args:
            count: Number of recent events to return (all if None)
            
        Returns:
            List of event objects
        """
        with self.buffer_lock:
            events = list(self.event_buffer)
            
            if count is not None:
                events = events[-count:]
            
            return events
    
    def get_events_by_type(self, event_type: str) -> List[Any]:
        """Get events filtered by type"""
        events = self.get_events()
        
        if event_type.lower() == 'keyboard':
            return [e for e in events if isinstance(e, KeyboardEvent)]
        elif event_type.lower() == 'mouse':
            return [e for e in events if isinstance(e, MouseEvent)]
        else:
            return []
    
    def export_events(self, filepath: str, format: str = "json") -> bool:
        """
        Export recorded events to file
        
        Args:
            filepath: Output file path
            format: Export format ("json" or "csv")
            
        Returns:
            True if successful
        """
        try:
            events = self.get_events()
            
            if format.lower() == "json":
                export_data = {
                    'metadata': {
                        'total_events': len(events),
                        'recording_duration': self.stats['recording_duration'],
                        'export_timestamp': time.time()
                    },
                    'events': [event.to_dict() for event in events]
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format.lower() == "csv":
                import csv
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['type', 'event', 'key_or_button', 'x', 'y', 'direction', 'velocity', 'timestamp'])
                    
                    # Write events
                    for event in events:
                        if isinstance(event, KeyboardEvent):
                            writer.writerow([
                                'keyboard', event.event.value, event.key, 
                                '', '', '', '', event.timestamp
                            ])
                        elif isinstance(event, MouseEvent):
                            writer.writerow([
                                'mouse', event.event.value, 
                                event.button.value if event.button else '',
                                event.x, event.y, event.direction or '', 
                                event.velocity, event.timestamp
                            ])
            
            self.logger.info(f"Events exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export events: {e}")
            return False
    
    def analyze_input_patterns(self) -> Dict[str, Any]:
        """Analyze recorded input patterns for insights"""
        events = self.get_events()
        
        if not events:
            return {'error': 'No events to analyze'}
        
        analysis = {
            'total_events': len(events),
            'event_types': {
                'keyboard': len([e for e in events if isinstance(e, KeyboardEvent)]),
                'mouse': len([e for e in events if isinstance(e, MouseEvent)])
            },
            'time_span': 0,
            'average_events_per_second': 0,
            'most_common_keys': {},
            'mouse_activity': {
                'total_clicks': 0,
                'total_moves': 0,
                'average_click_interval': 0
            }
        }
        
        if events:
            # Time analysis
            first_time = events[0].timestamp
            last_time = events[-1].timestamp
            analysis['time_span'] = last_time - first_time
            
            if analysis['time_span'] > 0:
                analysis['average_events_per_second'] = len(events) / analysis['time_span']
        
        # Keyboard analysis
        key_counts = {}
        for event in events:
            if isinstance(event, KeyboardEvent) and event.event == KeyboardEvents.DOWN:
                key_counts[event.key] = key_counts.get(event.key, 0) + 1
        
        analysis['most_common_keys'] = dict(sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Mouse analysis
        mouse_events = [e for e in events if isinstance(e, MouseEvent)]
        click_events = [e for e in mouse_events if e.event == MouseEvents.CLICK]
        move_events = [e for e in mouse_events if e.event == MouseEvents.MOVE]
        
        analysis['mouse_activity']['total_clicks'] = len(click_events)
        analysis['mouse_activity']['total_moves'] = len(move_events)
        
        if len(click_events) > 1:
            click_intervals = []
            for i in range(1, len(click_events)):
                interval = click_events[i].timestamp - click_events[i-1].timestamp
                click_intervals.append(interval)
            
            if click_intervals:
                analysis['mouse_activity']['average_click_interval'] = sum(click_intervals) / len(click_intervals)
        
        return analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics"""
        stats = self.stats.copy()
        stats['buffer_size'] = len(self.event_buffer)
        stats['is_recording'] = self.is_recording
        return stats
    
    def clear_buffer(self):
        """Clear the event buffer"""
        with self.buffer_lock:
            self.event_buffer.clear()
        self.logger.info("Event buffer cleared")
    
    def set_mouse_sensitivity(self, threshold: int):
        """Set mouse movement sensitivity"""
        self.mouse_move_threshold = threshold
        self.logger.info(f"Mouse sensitivity set to {threshold} pixels")
    
    def set_polling_rate(self, rate_hz: float):
        """Set input polling rate"""
        self.monitor_interval = 1.0 / rate_hz
        self.logger.info(f"Polling rate set to {rate_hz} Hz")