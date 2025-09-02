"""
Enhanced Screen Capture System
Based on SerpentAI D3DShot techniques for ultra-fast screen capture
"""

import time
import logging
import threading
from typing import Optional, Tuple, Any, Dict, List
from pathlib import Path
import numpy as np
from collections import deque
import cv2

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

class FrameBuffer:
    """Thread-safe frame buffer inspired by SerpentAI's design"""
    
    def __init__(self, max_size: int = 60):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def add_frame(self, frame: np.ndarray, timestamp: float = None):
        """Add frame to buffer with timestamp"""
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            self.buffer.append({
                'frame': frame,
                'timestamp': timestamp
            })
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Get the most recent frame"""
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
            return None
    
    def get_frame_stack(self, indices: Tuple[int, ...], stack_dimension: str = "last") -> Optional[np.ndarray]:
        """Get multiple frames as a stack (SerpentAI style)"""
        with self.lock:
            if not self.buffer or not indices:
                return None
                
            frames = []
            for idx in indices:
                if 0 <= idx < len(self.buffer):
                    frames.append(self.buffer[idx]['frame'])
            
            if not frames:
                return None
                
            if stack_dimension == "last":
                return np.stack(frames, axis=-1)
            elif stack_dimension == "first":
                return np.stack(frames, axis=0)
            else:
                return np.stack(frames, axis=int(stack_dimension))
    
    def clear(self):
        """Clear the frame buffer"""
        with self.lock:
            self.buffer.clear()
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

class EnhancedScreenCapture:
    """Enhanced screen capture system with SerpentAI optimizations"""
    
    def __init__(self, frame_buffer_size: int = 60, capture_output: str = "numpy"):
        self.logger = logging.getLogger(__name__)
        
        # Frame buffer for high-speed capture
        self.frame_buffer = FrameBuffer(frame_buffer_size)
        
        # Capture settings
        self.capture_output = capture_output
        self.is_capturing = False
        self.capture_thread = None
        
        # Performance tracking
        self.capture_stats = {
            'frames_captured': 0,
            'capture_start_time': 0,
            'avg_fps': 0,
            'last_capture_time': 0
        }
        
        # Display settings (simplified for cross-platform compatibility)
        self.current_region = None
        self.capture_interval = 0.033  # ~30 FPS default
        
        self.logger.info(f"Enhanced Screen Capture initialized with {capture_output} output")
    
    def screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Take a single screenshot
        
        Args:
            region: (left, top, right, bottom) for specific region
            
        Returns:
            Screenshot as numpy array or None if failed
        """
        try:
            if not PYAUTOGUI_AVAILABLE:
                self.logger.error("PyAutoGUI not available for screenshot capture")
                return None
            
            # Capture screenshot
            if region:
                # Convert region format: (left, top, right, bottom) -> (left, top, width, height)
                left, top, right, bottom = region
                width = right - left
                height = bottom - top
                screenshot = pyautogui.screenshot(region=(left, top, width, height))
            else:
                screenshot = pyautogui.screenshot()
            
            # Convert to numpy array
            if PIL_AVAILABLE and hasattr(screenshot, 'convert'):
                screenshot_rgb = screenshot.convert('RGB')
                frame = np.array(screenshot_rgb)
            else:
                frame = np.array(screenshot)
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def screenshot_to_disk(self, filename: Optional[str] = None, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
        """
        Take screenshot and save to disk
        
        Args:
            filename: Output filename (auto-generated if None)
            region: Capture region
            
        Returns:
            Filename of saved image
        """
        try:
            frame = self.screenshot(region)
            if frame is None:
                return None
            
            if filename is None:
                timestamp = time.time()
                filename = f"screenshot_{timestamp}.png"
            
            # Ensure output directory exists
            output_path = Path("screenshots")
            output_path.mkdir(exist_ok=True)
            
            full_path = output_path / filename
            
            # Convert BGR back to RGB for saving
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            cv2.imwrite(str(full_path), frame_rgb)
            self.logger.info(f"Screenshot saved to {full_path}")
            
            return str(full_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save screenshot: {e}")
            return None
    
    def capture(self, duration: Optional[float] = None, region: Optional[Tuple[int, int, int, int]] = None):
        """
        Start high-speed frame capture to buffer (SerpentAI style)
        
        Args:
            duration: Capture duration in seconds (None for indefinite)
            region: Screen region to capture
        """
        if self.is_capturing:
            self.logger.warning("Capture already in progress")
            return
        
        self.current_region = region
        self.is_capturing = True
        self.frame_buffer.clear()
        
        # Reset stats
        self.capture_stats['frames_captured'] = 0
        self.capture_stats['capture_start_time'] = time.time()
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(duration,),
            daemon=True
        )
        self.capture_thread.start()
        
        self.logger.info(f"Started high-speed capture (region={region}, duration={duration})")
    
    def _capture_loop(self, duration: Optional[float]):
        """Internal capture loop for high-speed frame capture"""
        end_time = None
        if duration:
            end_time = time.time() + duration
        
        while self.is_capturing:
            try:
                # Check duration limit
                if end_time and time.time() >= end_time:
                    break
                
                # Capture frame
                frame = self.screenshot(self.current_region)
                if frame is not None:
                    # Add to frame buffer
                    self.frame_buffer.add_frame(frame)
                    
                    # Update stats
                    self.capture_stats['frames_captured'] += 1
                    self.capture_stats['last_capture_time'] = time.time()
                    
                    # Calculate average FPS
                    elapsed = time.time() - self.capture_stats['capture_start_time']
                    if elapsed > 0:
                        self.capture_stats['avg_fps'] = self.capture_stats['frames_captured'] / elapsed
                
                # Control capture rate
                time.sleep(self.capture_interval)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
        
        self.is_capturing = False
        self.logger.info(f"Capture completed. Frames: {self.capture_stats['frames_captured']}, Avg FPS: {self.capture_stats['avg_fps']:.2f}")
    
    def stop(self):
        """Stop the capture process"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        self.logger.info("Screen capture stopped")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent captured frame"""
        frame_data = self.frame_buffer.get_latest_frame()
        if frame_data:
            return frame_data['frame']
        return None
    
    def get_frame_stack(self, indices: Tuple[int, ...], stack_dimension: str = "last") -> Optional[np.ndarray]:
        """
        Get multiple frames as a stack (SerpentAI style)
        
        Args:
            indices: Frame indices to stack
            stack_dimension: Where to add the stack dimension ("first", "last", or index)
            
        Returns:
            Stacked frames as numpy array
        """
        return self.frame_buffer.get_frame_stack(indices, stack_dimension)
    
    def screenshot_every(self, interval: float, duration: float, callback: Optional[callable] = None):
        """
        Take screenshots at regular intervals (SerpentAI style)
        
        Args:
            interval: Time between screenshots in seconds
            duration: Total duration in seconds
            callback: Optional callback function for each screenshot
        """
        def screenshot_worker():
            end_time = time.time() + duration
            screenshot_count = 0
            
            while time.time() < end_time:
                try:
                    frame = self.screenshot(self.current_region)
                    if frame is not None:
                        self.frame_buffer.add_frame(frame)
                        screenshot_count += 1
                        
                        if callback:
                            callback(frame, screenshot_count)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in screenshot_every: {e}")
                    time.sleep(interval)
            
            self.logger.info(f"screenshot_every completed: {screenshot_count} screenshots taken")
        
        # Start worker thread
        worker_thread = threading.Thread(target=screenshot_worker, daemon=True)
        worker_thread.start()
        
        return worker_thread
    
    def set_capture_rate(self, fps: float):
        """Set the capture frame rate"""
        self.capture_interval = 1.0 / fps
        self.logger.info(f"Capture rate set to {fps} FPS")
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get current capture statistics"""
        stats = self.capture_stats.copy()
        stats['buffer_size'] = self.frame_buffer.get_buffer_size()
        stats['is_capturing'] = self.is_capturing
        return stats
    
    def set_region(self, region: Optional[Tuple[int, int, int, int]]):
        """Set the default capture region"""
        self.current_region = region
        if region:
            self.logger.info(f"Capture region set to {region}")
        else:
            self.logger.info("Capture region reset to full screen")
    
    def get_optimal_capture_settings(self, target_fps: float = 30) -> Dict[str, Any]:
        """
        Calculate optimal capture settings based on system performance
        
        Args:
            target_fps: Desired frames per second
            
        Returns:
            Dictionary with recommended settings
        """
        # Test capture performance
        test_start = time.time()
        test_frames = 10
        
        for _ in range(test_frames):
            frame = self.screenshot()
            if frame is None:
                break
        
        test_duration = time.time() - test_start
        actual_fps = test_frames / test_duration if test_duration > 0 else 0
        
        # Calculate recommendations
        recommended_fps = min(actual_fps * 0.8, target_fps)  # 80% of max performance
        recommended_buffer_size = int(recommended_fps * 2)  # 2 seconds of frames
        
        recommendations = {
            'recommended_fps': recommended_fps,
            'recommended_buffer_size': recommended_buffer_size,
            'max_observed_fps': actual_fps,
            'test_duration': test_duration,
            'performance_rating': 'good' if actual_fps >= target_fps else 'limited'
        }
        
        self.logger.info(f"Performance test: {actual_fps:.2f} FPS, recommending {recommended_fps:.2f} FPS")
        
        return recommendations