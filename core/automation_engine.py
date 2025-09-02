"""
Automation Engine for Game Control
Handles mouse and keyboard automation for game interactions
"""

import time
import logging
from typing import Tuple, List, Optional, Dict, Any
import threading
import queue
import json
from pathlib import Path
import random
from collections import deque

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False
    # PyAutoGUI not available - automation disabled
    pyautogui = None

class AutomationEngine:
    """Main automation engine for game control"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._is_active = False
        self.action_queue = queue.Queue()
        self.worker_thread = None
        
        # Configure PyAutoGUI
        if PYAUTOGUI_AVAILABLE:
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.1  # Small pause between actions
        
        # Movement settings
        self.move_duration = 0.3
        self.click_duration = 0.1
        self.human_like_movement = True
        
        # Safety settings
        self.safe_regions = []  # Regions where clicks are safe
        self.forbidden_regions = []  # Regions to avoid
        
        # Additional automation properties
        self.enable_autoplay = True  # Enable autoplay by default for AI automation
        
        # SerpentAI inspired enhancements
        self.action_history = deque(maxlen=100)  # Track recent actions
        self.performance_stats = {
            'actions_executed': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'avg_action_time': 0.0
        }
        
        # Advanced movement patterns from SerpentAI
        self.movement_patterns = {
            'human_like': True,
            'bezier_curves': True,
            'randomization': True,
            'speed_variation': True
        }
        
        self.logger.info("Enhanced Automation engine initialized with SerpentAI optimizations")
    
    def start(self):
        """Start the automation engine"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self._is_active = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        self.logger.info("Automation engine started")
    
    def stop(self):
        """Stop the automation engine"""
        self._is_active = False
        
        # Clear pending actions
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for worker thread to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        self.logger.info("Automation engine stopped")
    
    def _worker_loop(self):
        """Main worker loop for processing automation actions"""
        while self._is_active:
            try:
                # Get next action from queue (timeout to allow periodic checks)
                action = self.action_queue.get(timeout=1.0)
                
                # Process the action
                self._execute_action(action)
                
                # Mark task as done
                self.action_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Action execution failed: {e}")
    
    def _execute_action(self, action: Dict[str, Any]):
        """Execute a single automation action"""
        try:
            action_type = action.get('type')
            
            if action_type == 'click':
                self._execute_click(action)
            elif action_type == 'move':
                self._execute_move(action)
            elif action_type == 'drag':
                self._execute_drag(action)
            elif action_type == 'key':
                self._execute_key(action)
            elif action_type == 'scroll':
                self._execute_scroll(action)
            elif action_type == 'wait':
                self._execute_wait(action)
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to execute action {action}: {e}")
    
    def _execute_click(self, action: Dict[str, Any]):
        """Execute a click action with SerpentAI enhancements"""
        action_start = time.time()
        x = action.get('x', 0)
        y = action.get('y', 0)
        button = action.get('button', 'left')
        clicks = action.get('clicks', 1)
        
        if not self._is_safe_position(x, y):
            self.logger.warning(f"Unsafe click position: ({x}, {y})")
            self._record_action_result(action, False, "unsafe_position")
            return
        
        try:
            # Advanced human-like movement with SerpentAI patterns
            if self.movement_patterns['human_like']:
                # Add natural randomness based on action importance
                noise_level = action.get('precision', 2)  # Lower = more precise
                x += random.randint(-noise_level, noise_level)
                y += random.randint(-noise_level, noise_level)
            
            # Enhanced movement to position
            self._enhanced_move_to_position(x, y)
            
            # Variable click duration for human-like behavior
            click_duration = self.click_duration
            if self.movement_patterns['speed_variation']:
                click_duration *= random.uniform(0.8, 1.2)
            
            # Perform click
            if PYAUTOGUI_AVAILABLE:
                pyautogui.click(x, y, clicks=clicks, button=button, duration=click_duration)
            else:
                self.logger.info(f"Demo mode: Would click at ({x}, {y})")
            
            # Record successful action
            action_time = time.time() - action_start
            self._record_action_result(action, True, "success", action_time)
            self.logger.debug(f"Enhanced click at ({x}, {y}) with {button} button")
            
        except Exception as e:
            action_time = time.time() - action_start
            self._record_action_result(action, False, str(e), action_time)
            self.logger.error(f"Click failed: {e}")
    
    def _execute_move(self, action: Dict[str, Any]):
        """Execute a move action"""
        x = action.get('x', 0)
        y = action.get('y', 0)
        
        if PYAUTOGUI_AVAILABLE:
            self._move_to_position(x, y)
        self.logger.debug(f"Moved to ({x}, {y})")
    
    def _execute_drag(self, action: Dict[str, Any]):
        """Execute a drag action"""
        from_x = action.get('from_x', 0)
        from_y = action.get('from_y', 0)
        to_x = action.get('to_x', 0)
        to_y = action.get('to_y', 0)
        button = action.get('button', 'left')
        
        if not (self._is_safe_position(from_x, from_y) and self._is_safe_position(to_x, to_y)):
            self.logger.warning(f"Unsafe drag positions: ({from_x}, {from_y}) to ({to_x}, {to_y})")
            return
        
        if PYAUTOGUI_AVAILABLE:
            pyautogui.drag(to_x - from_x, to_y - from_y, 
                          duration=self.move_duration, button=button)
        else:
            self.logger.info(f"Demo mode: Would drag from ({from_x}, {from_y}) to ({to_x}, {to_y})")
        
        self.logger.debug(f"Dragged from ({from_x}, {from_y}) to ({to_x}, {to_y})")
    
    def _execute_key(self, action: Dict[str, Any]):
        """Execute a keyboard action"""
        key = action.get('key', '')
        action_type = action.get('action', 'press')  # press, hold, release
        
        if not PYAUTOGUI_AVAILABLE:
            self.logger.info(f"Demo mode: Would press key {key}")
            return
            
        if action_type == 'press':
            pyautogui.press(key)
        elif action_type == 'hold':
            pyautogui.keyDown(key)
        elif action_type == 'release':
            pyautogui.keyUp(key)
        
        self.logger.debug(f"Key action: {action_type} {key}")
    
    def _execute_scroll(self, action: Dict[str, Any]):
        """Execute a scroll action"""
        x = action.get('x', None)
        y = action.get('y', None)
        scrolls = action.get('scrolls', 1)
        
        if not PYAUTOGUI_AVAILABLE:
            self.logger.info(f"Demo mode: Would scroll {scrolls}")
            return
            
        if x is not None and y is not None:
            pyautogui.scroll(scrolls, x=x, y=y)
        else:
            pyautogui.scroll(scrolls)
        
        self.logger.debug(f"Scrolled {scrolls} at ({x}, {y})")
    
    def _execute_wait(self, action: Dict[str, Any]):
        """Execute a wait action"""
        duration = action.get('duration', 1.0)
        time.sleep(duration)
        self.logger.debug(f"Waited {duration} seconds")
    
    def _move_to_position(self, x: int, y: int):
        """Move mouse to position with human-like movement"""
        if not PYAUTOGUI_AVAILABLE:
            return
            
        if self.human_like_movement:
            # Add slight curve to movement
            current_x, current_y = pyautogui.position()
            mid_x = (current_x + x) // 2 + random.randint(-10, 10)
            mid_y = (current_y + y) // 2 + random.randint(-10, 10)
            
            # Move in two steps for more natural movement
            pyautogui.moveTo(mid_x, mid_y, duration=self.move_duration / 2)
            pyautogui.moveTo(x, y, duration=self.move_duration / 2)
        else:
            pyautogui.moveTo(x, y, duration=self.move_duration)
    
    def _is_safe_position(self, x: int, y: int) -> bool:
        """Check if position is safe for automation"""
        # Check forbidden regions
        for region in self.forbidden_regions:
            if (region['x'] <= x <= region['x'] + region['width'] and
                region['y'] <= y <= region['y'] + region['height']):
                return False
        
        # If safe regions are defined, position must be in one of them
        if self.safe_regions:
            for region in self.safe_regions:
                if (region['x'] <= x <= region['x'] + region['width'] and
                    region['y'] <= y <= region['y'] + region['height']):
                    return True
            return False
        
        # No restrictions if no safe regions defined
        return True
    
    def queue_click(self, x: int, y: int, button: str = 'left', clicks: int = 1):
        """Queue a click action"""
        action = {
            'type': 'click',
            'x': x,
            'y': y,
            'button': button,
            'clicks': clicks
        }
        self.action_queue.put(action)
    
    def queue_move(self, x: int, y: int):
        """Queue a move action"""
        action = {
            'type': 'move',
            'x': x,
            'y': y
        }
        self.action_queue.put(action)
    
    def queue_drag(self, from_x: int, from_y: int, to_x: int, to_y: int, button: str = 'left'):
        """Queue a drag action"""
        action = {
            'type': 'drag',
            'from_x': from_x,
            'from_y': from_y,
            'to_x': to_x,
            'to_y': to_y,
            'button': button
        }
        self.action_queue.put(action)
    
    def queue_key(self, key: str, action_type: str = 'press'):
        """Queue a keyboard action"""
        action = {
            'type': 'key',
            'key': key,
            'action': action_type
        }
        self.action_queue.put(action)
    
    def queue_wait(self, duration: float):
        """Queue a wait action"""
        action = {
            'type': 'wait',
            'duration': duration
        }
        self.action_queue.put(action)
    
    def click_at_positions(self, positions: List[Tuple[int, int]], delay: float = 0.5):
        """Click at multiple positions with delay between clicks"""
        for i, (x, y) in enumerate(positions):
            self.queue_click(x, y)
            
            # Add delay between clicks (except after the last one)
            if i < len(positions) - 1:
                self.queue_wait(delay)
    
    def _enhanced_move_to_position(self, x: int, y: int):
        """Enhanced movement with SerpentAI inspired patterns"""
        if not PYAUTOGUI_AVAILABLE:
            return
        
        current_x, current_y = pyautogui.position()
        
        # Use Bezier curves for natural movement
        if self.movement_patterns['bezier_curves']:
            self._bezier_move(current_x, current_y, x, y)
        else:
            # Standard movement with enhanced timing
            duration = self.move_duration
            if self.movement_patterns['speed_variation']:
                duration *= random.uniform(0.7, 1.3)
            
            pyautogui.moveTo(x, y, duration=duration)
    
    def _bezier_move(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Move using Bezier curve for natural human-like movement"""
        if not PYAUTOGUI_AVAILABLE:
            return
        
        # Calculate control points for Bezier curve
        distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
        
        # Add some randomness to control points
        mid_x = (start_x + end_x) / 2 + random.randint(-50, 50)
        mid_y = (start_y + end_y) / 2 + random.randint(-50, 50)
        
        # Number of points based on distance
        num_points = max(10, int(distance / 20))
        
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            
            # Quadratic Bezier curve
            x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * mid_x + t ** 2 * end_x
            y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * mid_y + t ** 2 * end_y
            
            points.append((int(x), int(y)))
        
        # Move through points with variable speed
        base_duration = self.move_duration / num_points
        
        for point in points:
            if self.movement_patterns['speed_variation']:
                duration = base_duration * random.uniform(0.5, 1.5)
            else:
                duration = base_duration
            
            pyautogui.moveTo(point[0], point[1], duration=duration)
    
    def _record_action_result(self, action: Dict[str, Any], success: bool, 
                             result: str, duration: float = 0.0):
        """Record action result for performance tracking"""
        action_record = {
            'timestamp': time.time(),
            'action': action,
            'success': success,
            'result': result,
            'duration': duration
        }
        
        self.action_history.append(action_record)
        
        # Update performance stats
        self.performance_stats['actions_executed'] += 1
        if success:
            self.performance_stats['successful_actions'] += 1
        else:
            self.performance_stats['failed_actions'] += 1
        
        # Update average action time
        if duration > 0:
            total_time = (self.performance_stats['avg_action_time'] * 
                         (self.performance_stats['actions_executed'] - 1) + duration)
            self.performance_stats['avg_action_time'] = total_time / self.performance_stats['actions_executed']
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        
        # Calculate success rate
        total_actions = stats['actions_executed']
        if total_actions > 0:
            stats['success_rate'] = (stats['successful_actions'] / total_actions) * 100
        else:
            stats['success_rate'] = 0.0
        
        stats['recent_actions'] = len(self.action_history)
        return stats
    
    def adaptive_difficulty_adjustment(self):
        """Adjust automation parameters based on success rate"""
        stats = self.get_performance_stats()
        success_rate = stats['success_rate']
        
        if success_rate < 70:  # Low success rate
            # Make movements more precise
            self.move_duration *= 1.1
            self.click_duration *= 1.1
            self.movement_patterns['randomization'] = False
            self.logger.info("Adjusted for higher precision due to low success rate")
            
        elif success_rate > 90:  # High success rate
            # Allow faster, more natural movements
            self.move_duration *= 0.95
            self.click_duration *= 0.95
            self.movement_patterns['randomization'] = True
            self.logger.info("Increased movement speed due to high success rate")
    
    def enable_serpent_mode(self):
        """Enable SerpentAI enhanced movement patterns"""
        self.movement_patterns = {
            'human_like': True,
            'bezier_curves': True,
            'randomization': True,
            'speed_variation': True
        }
        self.logger.info("SerpentAI enhanced movement mode enabled")
    
    def disable_serpent_mode(self):
        """Disable SerpentAI enhancements for precise mode"""
        self.movement_patterns = {
            'human_like': False,
            'bezier_curves': False,
            'randomization': False,
            'speed_variation': False
        }
        self.logger.info("SerpentAI enhancements disabled - precision mode")
    
    def open_chests(self, chest_positions: List[Tuple[int, int]]) -> str:
        """Automate opening chests at given positions"""
        if not chest_positions:
            return "No chests found to open"
        
        self.logger.info(f"Opening {len(chest_positions)} chests")
        
        for i, position in enumerate(chest_positions):
            x, y = position[:2]
            confidence = getattr(position, '__getitem__', lambda i: 1.0)(2) if len(position) > 2 else 1.0
            self.logger.debug(f"Opening chest {i+1} at ({x}, {y}) with confidence {confidence}")
            
            # Move to chest position
            self.queue_move(x, y)
            self.queue_wait(0.2)
            
            # Double-click to open chest
            self.queue_click(x, y, clicks=2)
            self.queue_wait(0.8)  # Wait for chest animation
        
        return f"Queued actions to open {len(chest_positions)} chests"
    
    def hatch_eggs(self, egg_positions: List[Tuple[int, int]]) -> str:
        """Automate hatching eggs at given positions"""
        if not egg_positions:
            return "No eggs found to hatch"
        
        self.logger.info(f"Hatching {len(egg_positions)} eggs")
        
        for i, position in enumerate(egg_positions):
            x, y = position[:2]
            confidence = getattr(position, '__getitem__', lambda i: 1.0)(2) if len(position) > 2 else 1.0
            self.logger.debug(f"Hatching egg {i+1} at ({x}, {y}) with confidence {confidence}")
            
            # Move to egg position
            self.queue_move(x, y)
            self.queue_wait(0.2)
            
            # Click to select egg
            self.queue_click(x, y)
            self.queue_wait(0.5)
            
            # Press hatch key (assuming 'h' key hatches eggs)
            self.queue_key('h')
            self.queue_wait(1.0)  # Wait for hatching animation
        
        return f"Queued actions to hatch {len(egg_positions)} eggs"
    
    def move_to_area(self, target_position: Tuple[int, int]) -> str:
        """Move character to specific area"""
        if not target_position:
            return "No target position specified"
        
        x, y = target_position[:2]  # Handle confidence tuple
        
        self.logger.info(f"Moving to area at ({x}, {y})")
        
        # Move to target area
        self.queue_move(x, y)
        self.queue_click(x, y)  # Click to move character
        
        return f"Moving to area at ({x}, {y})"
    
    def stay_in_breakables_area(self, breakables_positions: List[Tuple[int, int]]) -> str:
        """Keep character in breakables area"""
        if not breakables_positions:
            return "No breakables area found"
        
        # Find center of breakables area
        if len(breakables_positions) == 1:
            center_x, center_y = breakables_positions[0][:2]
        else:
            # Calculate center of all breakables
            x_coords = [pos[0] for pos in breakables_positions]
            y_coords = [pos[1] for pos in breakables_positions]
            center_x = sum(x_coords) // len(x_coords)
            center_y = sum(y_coords) // len(y_coords)
        
        # Move to center of breakables area
        result = self.move_to_area((center_x, center_y))
        
        self.logger.info(f"Staying in breakables area at ({center_x}, {center_y})")
        return f"Moving to breakables area. {result}"
    
    def emergency_stop(self):
        """Emergency stop all automation"""
        self.logger.warning("Emergency stop activated")
        self.stop()
        
        # Move mouse to safe position (corner of screen)
        try:
            if PYAUTOGUI_AVAILABLE:
                pyautogui.moveTo(0, 0, duration=0.5)
        except:
            pass
    
    def get_queue_size(self) -> int:
        """Get current action queue size"""
        return self.action_queue.qsize()
    
    def clear_queue(self):
        """Clear all pending actions"""
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("Action queue cleared")
    
    def is_active(self) -> bool:
        """Check if automation engine is active"""
        return self._is_active and (self.worker_thread is not None and self.worker_thread.is_alive())
    
    def set_safe_region(self, x: int, y: int, width: int, height: int):
        """Set a safe region for automation"""
        region = {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
        self.safe_regions.append(region)
        self.logger.info(f"Added safe region: {region}")
    
    def set_forbidden_region(self, x: int, y: int, width: int, height: int):
        """Set a forbidden region for automation"""
        region = {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
        self.forbidden_regions.append(region)
        self.logger.info(f"Added forbidden region: {region}")
    
    def clear_regions(self):
        """Clear all safe and forbidden regions"""
        self.safe_regions.clear()
        self.forbidden_regions.clear()
        self.logger.info("Cleared all regions")
