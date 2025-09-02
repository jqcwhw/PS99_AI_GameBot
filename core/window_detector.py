"""
Window Detection System for AI Game Bot
Detects and manages external Roblox windows launched independently
"""

import logging
import psutil
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import pygetwindow as gw
    import pyautogui
    WINDOW_DETECTION_AVAILABLE = True
except ImportError:
    WINDOW_DETECTION_AVAILABLE = False
    # Create dummy classes for environments without window detection
    class DummyWindow:
        def __init__(self):
            self.title = "Dummy Window"
            self.pid = 0
            self.left = 0
            self.top = 0
            self.width = 800
            self.height = 600
        def activate(self): pass
        def maximize(self): pass
        def minimize(self): pass
        def close(self): pass

class WindowDetector:
    """Detects and manages external Roblox windows"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detected_windows = {}
        self.monitored_processes = set()
        self.window_configs = {}
        
        # Roblox process patterns
        self.roblox_processes = [
            "RobloxPlayerBeta.exe",
            "RobloxPlayer.exe", 
            "Roblox.exe"
        ]
        
        # Window title patterns
        self.roblox_window_patterns = [
            "Roblox",
            "Pet Simulator 99",
            "PS99",
            "RobloxPlayerBeta"
        ]
        
        self.logger.info("Window detection system initialized")
        if not WINDOW_DETECTION_AVAILABLE:
            self.logger.warning("Window detection libraries not available - using simulation mode")
    
    def scan_for_roblox_windows(self) -> List[Dict]:
        """Scan for all currently open Roblox windows"""
        detected = []
        
        if not WINDOW_DETECTION_AVAILABLE:
            # Return simulated windows for testing
            return [{
                'title': 'Pet Simulator 99 - Roblox',
                'pid': 12345,
                'window_id': 'sim_1',
                'position': (100, 100, 800, 600),
                'is_active': True,
                'process_name': 'RobloxPlayerBeta.exe'
            }]
        
        try:
            # Get all windows
            all_windows = gw.getAllWindows()
            
            for window in all_windows:
                if self._is_roblox_window(window):
                    window_info = {
                        'title': window.title,
                        'pid': self._get_window_pid(window),
                        'window_id': str(id(window)),
                        'position': (window.left, window.top, window.width, window.height),
                        'is_active': window == gw.getActiveWindow(),
                        'process_name': self._get_process_name(window),
                        'window_object': window
                    }
                    detected.append(window_info)
                    
            self.detected_windows = {w['window_id']: w for w in detected}
            self.logger.info(f"Detected {len(detected)} Roblox windows")
            
        except Exception as e:
            self.logger.error(f"Error scanning for windows: {e}")
            
        return detected
    
    def detect_game_windows(self) -> List[Dict]:
        """Detect game windows (alias for scan_for_roblox_windows for compatibility)"""
        return self.scan_for_roblox_windows()
    
    def get_roblox_processes(self) -> List[Dict]:
        """Get all running Roblox processes"""
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'create_time', 'memory_info']):
                if proc.info['name'] in self.roblox_processes:
                    process_info = {
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'create_time': proc.info['create_time'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'status': proc.status() if hasattr(proc, 'status') else 'running'
                    }
                    processes.append(process_info)
                    
        except Exception as e:
            self.logger.error(f"Error getting processes: {e}")
            
        return processes
    
    def monitor_window(self, window_id: str, config: Dict = None) -> bool:
        """Start monitoring a specific window"""
        try:
            if window_id not in self.detected_windows:
                self.logger.error(f"Window {window_id} not found")
                return False
            
            window_info = self.detected_windows[window_id]
            
            # Default monitoring configuration
            default_config = {
                'auto_focus': False,
                'keep_active': True,
                'capture_screenshots': False,
                'track_title_changes': True,
                'mutex_bypass_active': False
            }
            
            self.window_configs[window_id] = {**default_config, **(config or {})}
            self.monitored_processes.add(window_info['pid'])
            
            self.logger.info(f"Started monitoring window: {window_info['title']} (PID: {window_info['pid']})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error monitoring window {window_id}: {e}")
            return False
    
    def focus_window(self, window_id: str) -> bool:
        """Focus on a specific Roblox window"""
        try:
            if window_id not in self.detected_windows:
                return False
                
            if not WINDOW_DETECTION_AVAILABLE:
                self.logger.info(f"Simulating focus on window {window_id}")
                return True
                
            window_info = self.detected_windows[window_id]
            window = window_info.get('window_object')
            
            if window:
                window.activate()
                self.logger.info(f"Focused window: {window_info['title']}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error focusing window {window_id}: {e}")
            
        return False
    
    def get_window_screenshot(self, window_id: str) -> Optional[bytes]:
        """Capture screenshot of specific window"""
        try:
            if window_id not in self.detected_windows:
                return None
                
            if not WINDOW_DETECTION_AVAILABLE:
                # Return dummy screenshot data
                return b"dummy_screenshot_data"
                
            window_info = self.detected_windows[window_id]
            pos = window_info['position']
            
            # Capture screenshot of window region
            screenshot = pyautogui.screenshot(region=pos)
            
            # Convert to bytes
            import io
            img_bytes = io.BytesIO()
            screenshot.save(img_bytes, format='PNG')
            return img_bytes.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error capturing screenshot for window {window_id}: {e}")
            return None
    
    def is_window_responsive(self, window_id: str) -> bool:
        """Check if window is responsive"""
        try:
            if window_id not in self.detected_windows:
                return False
                
            window_info = self.detected_windows[window_id]
            pid = window_info['pid']
            
            # Check if process is still running
            try:
                proc = psutil.Process(pid)
                return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking window responsiveness: {e}")
            return False
    
    def get_window_title_changes(self, window_id: str) -> List[str]:
        """Get history of window title changes"""
        # This would track title changes over time
        # For now, return current title
        if window_id in self.detected_windows:
            return [self.detected_windows[window_id]['title']]
        return []
    
    def close_window(self, window_id: str) -> bool:
        """Close a specific window"""
        try:
            if window_id not in self.detected_windows:
                return False
                
            if not WINDOW_DETECTION_AVAILABLE:
                self.logger.info(f"Simulating close of window {window_id}")
                return True
                
            window_info = self.detected_windows[window_id]
            window = window_info.get('window_object')
            
            if window:
                window.close()
                self.logger.info(f"Closed window: {window_info['title']}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error closing window {window_id}: {e}")
            
        return False
    
    def get_active_window_info(self) -> Optional[Dict]:
        """Get information about currently active window"""
        try:
            if not WINDOW_DETECTION_AVAILABLE:
                return {
                    'title': 'Active Window (Simulated)',
                    'is_roblox': True,
                    'window_id': 'active_sim'
                }
                
            active_window = gw.getActiveWindow()
            if active_window and self._is_roblox_window(active_window):
                return {
                    'title': active_window.title,
                    'is_roblox': True,
                    'window_id': str(id(active_window)),
                    'position': (active_window.left, active_window.top, active_window.width, active_window.height)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting active window: {e}")
            
        return None
    
    def _is_roblox_window(self, window) -> bool:
        """Check if window is a Roblox window"""
        if not hasattr(window, 'title') or not window.title:
            return False
            
        title_lower = window.title.lower()
        return any(pattern.lower() in title_lower for pattern in self.roblox_window_patterns)
    
    def _get_window_pid(self, window) -> int:
        """Get PID of window process"""
        try:
            if hasattr(window, '_hWnd'):
                import win32process
                import win32gui
                _, pid = win32process.GetWindowThreadProcessId(window._hWnd)
                return pid
        except ImportError:
            pass
        return 0
    
    def _get_process_name(self, window) -> str:
        """Get process name for window"""
        try:
            pid = self._get_window_pid(window)
            if pid:
                proc = psutil.Process(pid)
                return proc.name()
        except Exception:
            pass
        return "Unknown"
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            'detected_windows': len(self.detected_windows),
            'monitored_processes': len(self.monitored_processes),
            'active_configurations': len(self.window_configs),
            'detection_available': WINDOW_DETECTION_AVAILABLE
        }