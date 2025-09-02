"""
Computer Vision System for Game Automation
Handles screen capture, image analysis, and game element recognition
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
import time
import threading
from pathlib import Path
import json

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False
    pyautogui = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from collections import deque

class VisionSystem:
    """Main computer vision system for game automation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_capturing = False
        self.last_screenshot = None
        self.capture_thread = None
        
        # Disable PyAutoGUI failsafe to prevent interruption
        if PYAUTOGUI_AVAILABLE:
            pyautogui.FAILSAFE = False
        
        # Load game element templates
        self.templates = self._load_templates()
        
        # Vision settings
        self.match_threshold = 0.8
        self.screen_region = None  # Full screen by default
        
        # SerpentAI-inspired frame buffer for high-performance capture
        self.frame_buffer = deque(maxlen=60)  # Store last 2 seconds at 30fps
        self.frame_buffer_lock = threading.Lock()
        
        # Performance optimization settings
        self.fps = 30
        self.frame_time = 1.0 / self.fps
        self.capture_stats = {
            'frames_captured': 0,
            'avg_fps': 0.0,
            'last_capture_time': 0.0
        }
        
        # Game-specific detection regions (SerpentAI style)
        self.detection_regions = {}
        self.sprite_templates = {}
        
        # Frame transformation pipeline
        self.frame_transforms = []
        
        self.logger.info("Enhanced Vision system initialized with SerpentAI optimizations")
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load game element templates from file"""
        templates_file = Path("data/game_elements.json")
        if templates_file.exists():
            try:
                with open(templates_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load templates: {e}")
        
        # Return default templates structure
        return {
            "chests": [],
            "eggs": [],
            "breakables": [],
            "ui_elements": [],
            "areas": {}
        }
    
    def save_templates(self):
        """Save current templates to file"""
        try:
            templates_file = Path("data/game_elements.json")
            templates_file.parent.mkdir(exist_ok=True)
            
            with open(templates_file, 'w') as f:
                json.dump(self.templates, f, indent=2)
                
            self.logger.info("Templates saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save templates: {e}")
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Enhanced screen capture with SerpentAI optimizations
        
        Args:
            region: (x, y, width, height) tuple for specific region
            
        Returns:
            Screenshot as numpy array or None if failed
        """
        try:
            if not PYAUTOGUI_AVAILABLE:
                self.logger.error("PyAutoGUI not available - cannot capture real screenshots. Install with: pip install PyAutoGUI")
                return None
            
            capture_start = time.time()
                
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            # Convert PIL image to numpy array with optimizations
            if PIL_AVAILABLE and hasattr(screenshot, 'convert'):
                frame = np.array(screenshot.convert('RGB'))
            else:
                frame = np.array(screenshot)
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add to frame buffer with timestamp (SerpentAI style)
            frame_data = {
                'frame': frame,
                'timestamp': capture_start,
                'frame_id': self.capture_stats['frames_captured']
            }
            
            with self.frame_buffer_lock:
                self.frame_buffer.append(frame_data)
            
            # Update capture statistics
            self.capture_stats['frames_captured'] += 1
            self.capture_stats['last_capture_time'] = capture_start
            
            # Calculate FPS
            if len(self.frame_buffer) > 1:
                time_span = capture_start - self.frame_buffer[0]['timestamp']
                if time_span > 0:
                    self.capture_stats['avg_fps'] = len(self.frame_buffer) / time_span
            
            self.last_screenshot = frame
            return frame
            
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            return None
    
    def find_template(self, template_name: str, screenshot: Optional[np.ndarray] = None) -> List[Tuple[int, int, float]]:
        """
        Find template matches in screenshot
        
        Args:
            template_name: Name of template to find
            screenshot: Screenshot to search in (uses last capture if None)
            
        Returns:
            List of (x, y, confidence) tuples for matches
        """
        if screenshot is None:
            screenshot = self.last_screenshot
            
        if screenshot is None:
            self.logger.warning("No screenshot available for template matching")
            return []
        
        matches = []
        
        # Get template data
        template_data = self.templates.get(template_name, [])
        if not template_data:
            self.logger.warning(f"No template data found for: {template_name}")
            return []
        
        for template_info in template_data:
            try:
                # In a real implementation, this would load actual template images
                # For now, we'll use color-based detection as a placeholder
                matches.extend(self._find_by_color_range(
                    screenshot, 
                    template_info.get('color_range', {}),
                    template_info.get('size_range', {}),
                    template_name
                ))
            except Exception as e:
                self.logger.error(f"Template matching failed for {template_name}: {e}")
        
        return matches
    
    def _find_by_color_range(self, screenshot: np.ndarray, color_range: Dict, size_range: Dict, element_type: str) -> List[Tuple[int, int, float]]:
        """
        Find elements by color range detection
        
        This is a simplified approach for demonstration.
        In a real implementation, you'd use actual template matching with cv2.matchTemplate
        """
        matches = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Default color ranges for different game elements
            default_ranges = {
                'chests': {'lower': [10, 100, 100], 'upper': [30, 255, 255]},  # Golden/brown
                'eggs': {'lower': [0, 100, 100], 'upper': [10, 255, 255]},     # Red/pink
                'breakables': {'lower': [100, 50, 50], 'upper': [130, 255, 255]}, # Blue
                'ui_elements': {'lower': [0, 0, 200], 'upper': [180, 30, 255]}  # White/bright
            }
            
            # Use provided color range or default
            if color_range:
                lower = np.array(color_range.get('lower', [0, 0, 0]))
                upper = np.array(color_range.get('upper', [180, 255, 255]))
            else:
                range_data = default_ranges.get(element_type, default_ranges['ui_elements'])
                lower = np.array(range_data['lower'])
                upper = np.array(range_data['upper'])
            
            # Create mask
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            min_area = size_range.get('min_area', 100)
            max_area = size_range.get('max_area', 10000)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence based on area and shape
                    confidence = min(0.9, area / max_area)
                    
                    # Use center point
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    matches.append((center_x, center_y, confidence))
            
        except Exception as e:
            self.logger.error(f"Color range detection failed: {e}")
        
        return matches
    
    def find_chests(self, screenshot: Optional[np.ndarray] = None) -> List[Tuple[int, int, float]]:
        """Find treasure chests in the screenshot"""
        return self.find_template('chests', screenshot)
    
    def find_eggs(self, screenshot: Optional[np.ndarray] = None) -> List[Tuple[int, int, float]]:
        """Find eggs in the screenshot"""
        return self.find_template('eggs', screenshot)
    
    def find_breakables_area(self, screenshot: Optional[np.ndarray] = None) -> List[Tuple[int, int, float]]:
        """Find breakables area markers"""
        return self.find_template('breakables', screenshot)
    
    def analyze_screen(self) -> Dict[str, Any]:
        """
        Comprehensive screen analysis
        
        Returns:
            Dictionary containing analysis results
        """
        screenshot = self.capture_screen()
        if screenshot is None:
            return {"error": "Failed to capture screen"}
        
        analysis = {
            "timestamp": time.time(),
            "screen_size": screenshot.shape[:2],
            "elements_found": {}
        }
        
        # Find all types of game elements
        element_types = ['chests', 'eggs', 'breakables', 'ui_elements']
        
        for element_type in element_types:
            matches = self.find_template(element_type, screenshot)
            analysis["elements_found"][element_type] = {
                "count": len(matches),
                "positions": matches
            }
        
        # Calculate screen regions
        height, width = screenshot.shape[:2]
        analysis["regions"] = {
            "center": (width // 2, height // 2),
            "quadrants": {
                "top_left": (width // 4, height // 4),
                "top_right": (3 * width // 4, height // 4),
                "bottom_left": (width // 4, 3 * height // 4),
                "bottom_right": (3 * width // 4, 3 * height // 4)
            }
        }
        
        # Basic color analysis
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Calculate dominant colors
            colors = screenshot.reshape(-1, 3)
            unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
            
            # Get top 5 most common colors
            top_indices = np.argsort(counts)[-5:]
            dominant_colors = [unique_colors[i].tolist() for i in top_indices]
            
            analysis["color_analysis"] = {
                "dominant_colors": dominant_colors,
                "average_brightness": np.mean(cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)),
                "color_variance": np.var(colors.astype(np.float32))
            }
        except Exception as e:
            self.logger.error(f"Color analysis failed: {e}")
            analysis["color_analysis"] = {"error": str(e)}
        
        return analysis
    
    def learn_element(self, element_type: str, region: Tuple[int, int, int, int], name: str = "") -> bool:
        """
        Learn a new game element from a screen region
        
        Args:
            element_type: Type of element (chest, egg, etc.)
            region: (x, y, width, height) region containing the element
            name: Optional name for the element
            
        Returns:
            True if learning was successful
        """
        try:
            # Capture the specific region
            element_screenshot = self.capture_screen(region)
            if element_screenshot is None:
                return False
            
            # Analyze the element
            element_analysis = self._analyze_element(element_screenshot)
            
            # Create template entry
            template_entry = {
                "name": name or f"{element_type}_{len(self.templates.get(element_type, []))}",
                "type": element_type,
                "region": region,
                "color_range": element_analysis.get("color_range", {}),
                "size_range": element_analysis.get("size_range", {}),
                "features": element_analysis.get("features", {}),
                "learned_at": time.time()
            }
            
            # Add to templates
            if element_type not in self.templates:
                self.templates[element_type] = []
            
            self.templates[element_type].append(template_entry)
            
            # Save templates
            self.save_templates()
            
            self.logger.info(f"Learned new {element_type}: {template_entry['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to learn element: {e}")
            return False
    
    def _analyze_element(self, element_image: np.ndarray) -> Dict[str, Any]:
        """Analyze an element image to extract features"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(element_image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(element_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate color statistics
            h, s, v = cv2.split(hsv)
            color_stats = {
                "hue_mean": float(np.mean(h)),
                "hue_std": float(np.std(h)),
                "saturation_mean": float(np.mean(s)),
                "saturation_std": float(np.std(s)),
                "value_mean": float(np.mean(v)),
                "value_std": float(np.std(v))
            }
            
            # Determine color range
            color_range = {
                "lower": [
                    max(0, color_stats["hue_mean"] - 2 * color_stats["hue_std"]),
                    max(0, color_stats["saturation_mean"] - 2 * color_stats["saturation_std"]),
                    max(0, color_stats["value_mean"] - 2 * color_stats["value_std"])
                ],
                "upper": [
                    min(180, color_stats["hue_mean"] + 2 * color_stats["hue_std"]),
                    min(255, color_stats["saturation_mean"] + 2 * color_stats["saturation_std"]),
                    min(255, color_stats["value_mean"] + 2 * color_stats["value_std"])
                ]
            }
            
            # Calculate size range
            height, width = element_image.shape[:2]
            area = height * width
            size_range = {
                "min_area": max(50, int(area * 0.7)),
                "max_area": int(area * 1.3),
                "aspect_ratio": width / height if height > 0 else 1.0
            }
            
            # Extract basic features
            features = {
                "edges": len(cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]),
                "brightness": float(np.mean(gray)),
                "contrast": float(np.std(gray)),
                "size": (width, height)
            }
            
            return {
                "color_range": color_range,
                "size_range": size_range,
                "features": features,
                "color_stats": color_stats
            }
            
        except Exception as e:
            self.logger.error(f"Element analysis failed: {e}")
            return {}
    
    def is_active(self) -> bool:
        """Check if vision system is active"""
        return self.last_screenshot is not None
    
    def start_continuous_capture(self, interval: float = 1.0):
        """Start continuous screen capture in background thread"""
        if self.capture_thread and self.capture_thread.is_alive():
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._continuous_capture_loop, args=(interval,))
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.logger.info(f"Started continuous capture with {interval}s interval")
    
    def stop_continuous_capture(self):
        """Stop continuous screen capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
        
        self.logger.info("Stopped continuous capture")
    
    def _continuous_capture_loop(self, interval: float):
        """Background loop for continuous screen capture"""
        while self.is_capturing:
            try:
                self.capture_screen()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Continuous capture error: {e}")
                time.sleep(interval)
