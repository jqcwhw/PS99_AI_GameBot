"""
SerpentAI Enhanced Vision System
Integrates advanced computer vision techniques from SerpentAI and D3DShot
"""

import time
import logging
import threading
from typing import Optional, Tuple, Any, Dict, List, Union
from pathlib import Path
import numpy as np
from collections import deque
import cv2
import json

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

class GameFrameBuffer:
    """
    Game frame buffer similar to SerpentAI's GameFrameBuffer
    Optimized for game AI applications
    """
    
    def __init__(self, size: int = 60):
        self.frames = deque(maxlen=size)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def add_frame(self, frame: np.ndarray, timestamp: float = None, game_state: Dict = None):
        """Add a game frame with metadata"""
        if timestamp is None:
            timestamp = time.time()
            
        frame_data = {
            'frame': frame,
            'timestamp': timestamp,
            'game_state': game_state or {},
            'frame_id': len(self.frames)
        }
        
        with self.lock:
            self.frames.append(frame_data)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame"""
        with self.lock:
            if self.frames:
                return self.frames[-1]['frame']
            return None
    
    def get_frame_stack(self, indices: List[int], stack_dimension: str = "last") -> Optional[np.ndarray]:
        """Get frame stack for temporal learning"""
        with self.lock:
            if not self.frames or not indices:
                return None
                
            frames = []
            for idx in indices:
                if 0 <= idx < len(self.frames):
                    frames.append(self.frames[idx]['frame'])
            
            if not frames:
                return None
                
            if stack_dimension == "last":
                return np.stack(frames, axis=-1)
            elif stack_dimension == "first":
                return np.stack(frames, axis=0)
            else:
                return np.stack(frames, axis=int(stack_dimension))
    
    def get_frames_in_range(self, start_time: float, end_time: float) -> List[np.ndarray]:
        """Get frames within a time range"""
        with self.lock:
            frames = []
            for frame_data in self.frames:
                if start_time <= frame_data['timestamp'] <= end_time:
                    frames.append(frame_data['frame'])
            return frames

class FrameTransformationPipeline:
    """
    Frame transformation pipeline inspired by SerpentAI
    Supports multiple transformation stages for preprocessing
    """
    
    def __init__(self, pipeline_string: str = ""):
        self.pipeline_string = pipeline_string
        self.transformations = self._parse_pipeline(pipeline_string)
        self.logger = logging.getLogger(__name__)
    
    def _parse_pipeline(self, pipeline_string: str) -> List[Dict[str, Any]]:
        """Parse pipeline string into transformation steps"""
        if not pipeline_string:
            return []
        
        transformations = []
        steps = pipeline_string.split("|")
        
        for step in steps:
            step = step.strip()
            if step == "GRAYSCALE":
                transformations.append({"type": "grayscale"})
            elif step.startswith("RESIZE"):
                # Format: RESIZE:640x480
                parts = step.split(":")
                if len(parts) == 2:
                    try:
                        width, height = map(int, parts[1].split("x"))
                        transformations.append({"type": "resize", "width": width, "height": height})
                    except ValueError:
                        self.logger.warning(f"Invalid resize format: {step}")
            elif step == "NORMALIZE":
                transformations.append({"type": "normalize"})
            elif step == "PNG":
                transformations.append({"type": "png"})
            elif step.startswith("CROP"):
                # Format: CROP:x1,y1,x2,y2
                parts = step.split(":")
                if len(parts) == 2:
                    try:
                        coords = list(map(int, parts[1].split(",")))
                        if len(coords) == 4:
                            transformations.append({
                                "type": "crop", 
                                "coords": coords
                            })
                    except ValueError:
                        self.logger.warning(f"Invalid crop format: {step}")
        
        return transformations
    
    def transform(self, frame: np.ndarray) -> Union[np.ndarray, bytes]:
        """Apply transformation pipeline to frame"""
        current_frame = frame.copy()
        
        for transformation in self.transformations:
            try:
                if transformation["type"] == "grayscale":
                    if len(current_frame.shape) == 3:
                        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                
                elif transformation["type"] == "resize":
                    current_frame = cv2.resize(
                        current_frame, 
                        (transformation["width"], transformation["height"])
                    )
                
                elif transformation["type"] == "normalize":
                    current_frame = current_frame.astype(np.float32) / 255.0
                
                elif transformation["type"] == "crop":
                    x1, y1, x2, y2 = transformation["coords"]
                    current_frame = current_frame[y1:y2, x1:x2]
                
                elif transformation["type"] == "png":
                    # Convert to PNG bytes
                    if PIL_AVAILABLE:
                        if len(current_frame.shape) == 3:
                            img = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
                        else:
                            img = Image.fromarray(current_frame)
                        
                        import io
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        return buffer.getvalue()
                    else:
                        self.logger.warning("PIL not available for PNG conversion")
                        
            except Exception as e:
                self.logger.error(f"Error in transformation {transformation['type']}: {e}")
                continue
        
        return current_frame

class SerpentEnhancedVision:
    """
    Enhanced vision system combining SerpentAI and D3DShot optimizations
    Provides high-performance screen capture and processing for game AI
    """
    
    def __init__(self, width: int = 640, height: int = 480, x_offset: int = 0, y_offset: int = 0, 
                 fps: int = 30, pipeline_string: str = "", buffer_seconds: int = 5):
        self.logger = logging.getLogger(__name__)
        
        # Screen capture settings
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset
        
        # Performance settings
        self.frame_time = 1.0 / fps
        self.fps = fps
        
        # Frame management
        self.frame_buffer_size = buffer_seconds * fps
        self.game_frame_buffer = GameFrameBuffer(self.frame_buffer_size)
        
        # Frame processing
        self.frame_transformation_pipeline = None
        if pipeline_string:
            self.frame_transformation_pipeline = FrameTransformationPipeline(pipeline_string)
        
        # Capture state
        self.is_capturing = False
        self.capture_thread = None
        
        # Performance metrics
        self.capture_stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'avg_fps': 0.0,
            'processing_time': 0.0,
            'last_frame_time': 0.0
        }
        
        # Game-specific detection regions
        self.detection_regions = {}
        self.sprite_templates = {}
        
        self.logger.info(f"SerpentAI Enhanced Vision initialized: {width}x{height} @ {fps}FPS")
    
    def start_continuous_analysis(self):
        """Start continuous analysis mode for SerpentAI compatibility"""
        self.logger.info("Starting SerpentAI continuous analysis mode")
        self.start_capture()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        self.logger.info("Continuous analysis mode activated")
    
    def _analysis_loop(self):
        """Continuous analysis loop for game state detection"""
        while self.is_capturing:
            try:
                # Get latest frame for analysis
                latest_frame = self.game_frame_buffer.get_latest_frame()
                
                if latest_frame is not None:
                    # Perform game state analysis
                    self._analyze_game_state(latest_frame)
                
                # Control analysis frequency
                time.sleep(0.1)  # Analyze 10 times per second
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(0.5)
    
    def _analyze_game_state(self, frame: np.ndarray):
        """Analyze current frame for game state information"""
        try:
            # Basic game state detection
            game_state = {
                'timestamp': time.time(),
                'frame_shape': frame.shape,
                'detected_objects': [],
                'analysis_complete': True
            }
            
            # Add to buffer with game state metadata
            self.game_frame_buffer.add_frame(frame, game_state=game_state)
            
        except Exception as e:
            self.logger.error(f"Game state analysis failed: {e}")
    
    def start_capture(self):
        """Start high-performance frame capture loop"""
        if self.is_capturing:
            self.logger.warning("Capture already in progress")
            return
        
        self.is_capturing = True
        self.capture_stats['frames_captured'] = 0
        self.capture_stats['frames_processed'] = 0
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info("High-performance capture started")
    
    def _capture_loop(self):
        """Main capture loop optimized for real-time performance"""
        last_fps_check = time.time()
        fps_frame_count = 0
        
        while self.is_capturing:
            try:
                cycle_start = time.time()
                
                # Capture frame
                frame = self._grab_frame()
                if frame is not None:
                    # Process frame if pipeline exists
                    processed_frame = frame
                    if self.frame_transformation_pipeline:
                        processing_start = time.time()
                        processed_frame = self.frame_transformation_pipeline.transform(frame)
                        processing_time = time.time() - processing_start
                        self.capture_stats['processing_time'] += processing_time
                        self.capture_stats['frames_processed'] += 1
                    
                    # Add to buffer
                    self.game_frame_buffer.add_frame(frame, cycle_start)
                    
                    # Update stats
                    self.capture_stats['frames_captured'] += 1
                    self.capture_stats['last_frame_time'] = cycle_start
                    fps_frame_count += 1
                
                # Calculate FPS periodically
                current_time = time.time()
                if current_time - last_fps_check >= 1.0:
                    self.capture_stats['avg_fps'] = fps_frame_count / (current_time - last_fps_check)
                    fps_frame_count = 0
                    last_fps_check = current_time
                
                # Control frame rate
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, self.frame_time - cycle_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
        
        self.logger.info("Capture loop ended")
    
    def _grab_frame(self) -> Optional[np.ndarray]:
        """Optimized frame grabbing using PyAutoGUI"""
        try:
            if not PYAUTOGUI_AVAILABLE:
                return None
            
            # Define capture region
            if self.x_offset or self.y_offset or self.width or self.height:
                region = (self.x_offset, self.y_offset, self.width, self.height)
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            # Convert to numpy array
            if PIL_AVAILABLE and hasattr(screenshot, 'convert'):
                frame = np.array(screenshot.convert('RGB'))
            else:
                frame = np.array(screenshot)
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame grab failed: {e}")
            return None
    
    def stop_capture(self):
        """Stop frame capture"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        self.logger.info("Frame capture stopped")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent captured frame"""
        return self.game_frame_buffer.get_latest_frame()
    
    def get_frame_stack(self, indices: List[int], stack_dimension: str = "last") -> Optional[np.ndarray]:
        """Get frame stack for temporal analysis"""
        return self.game_frame_buffer.get_frame_stack(indices, stack_dimension)
    
    def detect_sprites(self, frame: np.ndarray, sprite_templates: Dict[str, np.ndarray], 
                      threshold: float = 0.8) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Detect sprites in frame using template matching
        Optimized for real-time game detection
        """
        detections = {}
        
        try:
            # Convert to grayscale for faster matching
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            for sprite_name, template in sprite_templates.items():
                # Ensure template is grayscale
                if len(template.shape) == 3:
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # Template matching
                result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)
                
                # Store detections with confidence scores
                sprite_detections = []
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    sprite_detections.append((pt[0], pt[1], confidence))
                
                detections[sprite_name] = sprite_detections
                
        except Exception as e:
            self.logger.error(f"Sprite detection failed: {e}")
        
        return detections
    
    def detect_color_regions(self, frame: np.ndarray, color_ranges: Dict[str, Tuple[Tuple, Tuple]], 
                           min_area: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect colored regions in frame using HSV color space
        Optimized for game element detection
        """
        detections = {}
        
        try:
            # Convert to HSV for better color detection
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            for color_name, (lower_bound, upper_bound) in color_ranges.items():
                # Create color mask
                mask = cv2.inRange(hsv_frame, np.array(lower_bound), np.array(upper_bound))
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter by area and extract properties
                color_detections = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area >= min_area:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Calculate center
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        color_detections.append({
                            'center': (center_x, center_y),
                            'bbox': (x, y, w, h),
                            'area': area,
                            'contour': contour
                        })
                
                detections[color_name] = color_detections
                
        except Exception as e:
            self.logger.error(f"Color detection failed: {e}")
        
        return detections
    
    def set_detection_region(self, name: str, region: Tuple[int, int, int, int]):
        """Set a named detection region for focused analysis"""
        self.detection_regions[name] = region
        self.logger.info(f"Detection region '{name}' set to {region}")
    
    def analyze_region(self, frame: np.ndarray, region_name: str) -> Optional[np.ndarray]:
        """Extract and return a specific region from frame"""
        if region_name not in self.detection_regions:
            self.logger.warning(f"Unknown detection region: {region_name}")
            return None
        
        try:
            x1, y1, x2, y2 = self.detection_regions[region_name]
            return frame[y1:y2, x1:x2]
        except Exception as e:
            self.logger.error(f"Region analysis failed: {e}")
            return None
    
    def load_sprite_templates(self, templates_dir: Path):
        """Load sprite templates from directory"""
        try:
            templates_dir = Path(templates_dir)
            if not templates_dir.exists():
                self.logger.warning(f"Templates directory not found: {templates_dir}")
                return
            
            for template_file in templates_dir.glob("*.png"):
                template = cv2.imread(str(template_file))
                if template is not None:
                    sprite_name = template_file.stem
                    self.sprite_templates[sprite_name] = template
                    self.logger.info(f"Loaded sprite template: {sprite_name}")
            
            self.logger.info(f"Loaded {len(self.sprite_templates)} sprite templates")
            
        except Exception as e:
            self.logger.error(f"Failed to load sprite templates: {e}")
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get current capture performance statistics"""
        stats = self.capture_stats.copy()
        stats['buffer_size'] = len(self.game_frame_buffer.frames)
        stats['is_capturing'] = self.is_capturing
        stats['fps_target'] = self.fps
        
        # Calculate average processing time
        if stats['frames_processed'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['frames_processed']
        else:
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    def set_capture_region(self, x: int, y: int, width: int, height: int):
        """Update capture region"""
        self.x_offset = x
        self.y_offset = y
        self.width = width
        self.height = height
        self.logger.info(f"Capture region updated: {x},{y} {width}x{height}")
    
    def set_fps(self, fps: int):
        """Update target FPS"""
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.logger.info(f"Target FPS updated to {fps}")
    
    def export_detection_config(self, filepath: str):
        """Export current detection configuration"""
        try:
            config = {
                'detection_regions': self.detection_regions,
                'capture_settings': {
                    'width': self.width,
                    'height': self.height,
                    'x_offset': self.x_offset,
                    'y_offset': self.y_offset,
                    'fps': self.fps
                },
                'pipeline_string': self.frame_transformation_pipeline.pipeline_string if self.frame_transformation_pipeline else ""
            }
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Detection config exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export config: {e}")
    
    def load_detection_config(self, filepath: str):
        """Load detection configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Load settings
            if 'detection_regions' in config:
                self.detection_regions = config['detection_regions']
            
            if 'capture_settings' in config:
                settings = config['capture_settings']
                self.width = settings.get('width', self.width)
                self.height = settings.get('height', self.height)
                self.x_offset = settings.get('x_offset', self.x_offset)
                self.y_offset = settings.get('y_offset', self.y_offset)
                self.fps = settings.get('fps', self.fps)
                self.frame_time = 1.0 / self.fps
            
            if 'pipeline_string' in config and config['pipeline_string']:
                self.frame_transformation_pipeline = FrameTransformationPipeline(config['pipeline_string'])
            
            self.logger.info(f"Detection config loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")