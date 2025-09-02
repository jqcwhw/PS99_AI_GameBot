"""
Enhanced Vision System for Advanced Game AI
Provides comprehensive game element detection, path learning, boundary detection, and spatial intelligence
"""

import cv2
import numpy as np
import logging
import time
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
import pickle

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False
    pyautogui = None

@dataclass
class GameElement:
    """Represents a detected game element"""
    element_type: str
    position: Tuple[int, int]
    confidence: float
    size: Tuple[int, int]
    color_signature: List[int]
    shape_features: Dict[str, float]
    timestamp: float
    stable_duration: float = 0.0

@dataclass
class GameZone:
    """Represents a detected game zone or area"""
    zone_type: str
    boundaries: List[Tuple[int, int]]
    center: Tuple[int, int]
    area: float
    color_characteristics: Dict[str, Any]
    detected_elements: List[str]
    access_points: List[Tuple[int, int]]

@dataclass
class GamePath:
    """Represents a learned path in the game"""
    path_id: str
    waypoints: List[Tuple[int, int]]
    path_type: str  # 'navigation', 'farming', 'collection', etc.
    success_rate: float
    average_time: float
    obstacles: List[Tuple[int, int]]
    preferred_route: bool = False

class EnhancedVisionSystem:
    """Advanced vision system with comprehensive game understanding"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Vision state
        self.is_active = False
        self.last_screenshot = None
        self.screenshot_history = deque(maxlen=30)  # Last 30 screenshots for analysis
        
        # Element detection
        self.detected_elements = {}
        self.element_history = deque(maxlen=1000)
        self.element_templates = self._load_element_templates()
        self.custom_detectors = []
        
        # Spatial intelligence
        self.game_zones = {}
        self.learned_paths = {}
        self.boundary_map = None
        self.obstacle_map = None
        self.navigation_grid = None
        
        # Dynamic learning
        self.ui_patterns = {}
        self.minigame_signatures = {}
        self.event_patterns = {}
        self.context_memory = deque(maxlen=500)
        
        # Analysis settings
        self.detection_sensitivity = 0.7
        self.stability_threshold = 3  # Frames an element must be stable
        self.zone_detection_threshold = 0.8
        
        # Background analysis
        self.analysis_thread = None
        self.continuous_analysis = False
        
        self.logger.info("Enhanced Vision System initialized with advanced capabilities")
    
    def start_enhanced_vision(self):
        """Start enhanced vision with continuous analysis"""
        if self.analysis_thread and self.analysis_thread.is_alive():
            return
        
        self.is_active = True
        self.continuous_analysis = True
        
        self.analysis_thread = threading.Thread(target=self._continuous_analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        self.logger.info("Enhanced vision started with continuous learning")
    
    def stop_enhanced_vision(self):
        """Stop enhanced vision"""
        self.continuous_analysis = False
        self.is_active = False
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5.0)
        
        self._save_learned_data()
        self.logger.info("Enhanced vision stopped and data saved")
    
    def _continuous_analysis_loop(self):
        """Continuous analysis loop for real-time learning"""
        while self.continuous_analysis:
            try:
                # Capture and analyze screen
                screenshot = self._capture_enhanced_screenshot()
                if screenshot is not None:
                    self._comprehensive_analysis(screenshot)
                
                time.sleep(0.5)  # Analyze every 500ms
                
            except Exception as e:
                self.logger.error(f"Continuous analysis error: {e}")
                time.sleep(1.0)
    
    def _capture_enhanced_screenshot(self) -> Optional[np.ndarray]:
        """Enhanced screenshot capture with metadata"""
        try:
            if not PYAUTOGUI_AVAILABLE:
                # Return dummy screenshot for demo
                dummy = np.zeros((600, 800, 3), dtype=np.uint8)
                # Add some random elements for demo
                cv2.rectangle(dummy, (100, 100), (150, 150), (0, 255, 0), -1)  # Green chest
                cv2.circle(dummy, (300, 200), 20, (0, 0, 255), -1)  # Red egg
                cv2.rectangle(dummy, (500, 300), (600, 400), (255, 0, 0), 2)  # Blue boundary
                return dummy
            
            screenshot = pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Add timestamp and store in history
            screenshot_data = {
                'image': screenshot_cv,
                'timestamp': time.time(),
                'analysis_complete': False
            }
            
            self.screenshot_history.append(screenshot_data)
            self.last_screenshot = screenshot_cv
            
            return screenshot_cv
            
        except Exception as e:
            self.logger.error(f"Enhanced screenshot capture failed: {e}")
            return None
    
    def _comprehensive_analysis(self, screenshot: np.ndarray):
        """Comprehensive analysis of game screen"""
        try:
            analysis_results = {
                'timestamp': time.time(),
                'elements': {},
                'zones': {},
                'ui_changes': {},
                'events': [],
                'navigation_info': {}
            }
            
            # Multi-level element detection
            analysis_results['elements'] = self._detect_all_elements(screenshot)
            
            # Zone and boundary detection
            analysis_results['zones'] = self._detect_game_zones(screenshot)
            
            # UI pattern analysis
            analysis_results['ui_changes'] = self._analyze_ui_patterns(screenshot)
            
            # Event detection
            analysis_results['events'] = self._detect_game_events(screenshot)
            
            # Navigation analysis
            analysis_results['navigation_info'] = self._analyze_navigation_context(screenshot)
            
            # Store analysis for learning
            self.context_memory.append(analysis_results)
            
            # Update learned knowledge
            self._update_learned_knowledge(analysis_results)
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
    
    def _detect_all_elements(self, screenshot: np.ndarray) -> Dict[str, List[GameElement]]:
        """Advanced multi-element detection"""
        detected = defaultdict(list)
        
        try:
            # Standard element detection (chests, eggs, etc.)
            standard_elements = self._detect_standard_elements(screenshot)
            for element_type, elements in standard_elements.items():
                detected[element_type].extend(elements)
            
            # Dynamic element discovery
            dynamic_elements = self._discover_unknown_elements(screenshot)
            for element in dynamic_elements:
                detected['unknown'].append(element)
            
            # UI element detection
            ui_elements = self._detect_ui_elements(screenshot)
            detected['ui'].extend(ui_elements)
            
            # Interactive element detection
            interactive_elements = self._detect_interactive_elements(screenshot)
            detected['interactive'].extend(interactive_elements)
            
            # Resource node detection
            resource_elements = self._detect_resource_nodes(screenshot)
            detected['resources'].extend(resource_elements)
            
            # Enemy/NPC detection
            character_elements = self._detect_characters(screenshot)
            detected['characters'].extend(character_elements)
            
        except Exception as e:
            self.logger.error(f"Element detection failed: {e}")
        
        return dict(detected)
    
    def _detect_standard_elements(self, screenshot: np.ndarray) -> Dict[str, List[GameElement]]:
        """Detect standard game elements (chests, eggs, breakables)"""
        elements = defaultdict(list)
        
        # Enhanced detection using multiple methods
        for element_type in ['chests', 'eggs', 'breakables', 'npcs', 'doors', 'collectibles']:
            # Color-based detection
            color_elements = self._color_based_detection(screenshot, element_type)
            
            # Shape-based detection
            shape_elements = self._shape_based_detection(screenshot, element_type)
            
            # Template matching (if templates exist)
            template_elements = self._template_based_detection(screenshot, element_type)
            
            # Combine and filter results
            all_candidates = color_elements + shape_elements + template_elements
            filtered_elements = self._filter_and_validate_elements(all_candidates, element_type)
            
            elements[element_type] = filtered_elements
        
        return dict(elements)
    
    def _discover_unknown_elements(self, screenshot: np.ndarray) -> List[GameElement]:
        """Discover new/unknown game elements dynamically"""
        unknown_elements = []
        
        try:
            # Use edge detection and contour analysis
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 5000:  # Reasonable size range
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract features
                    roi = screenshot[y:y+h, x:x+w]
                    features = self._extract_element_features(roi)
                    
                    # Check if this is a known element type
                    if not self._is_known_element_type(features):
                        element = GameElement(
                            element_type='unknown',
                            position=(x + w//2, y + h//2),
                            confidence=0.6,
                            size=(w, h),
                            color_signature=self._get_color_signature(roi),
                            shape_features=features,
                            timestamp=time.time()
                        )
                        unknown_elements.append(element)
            
        except Exception as e:
            self.logger.error(f"Unknown element discovery failed: {e}")
        
        return unknown_elements
    
    def _detect_game_zones(self, screenshot: np.ndarray) -> Dict[str, GameZone]:
        """Detect different zones and areas in the game"""
        zones = {}
        
        try:
            # Color-based zone detection
            zones.update(self._detect_color_zones(screenshot))
            
            # Texture-based zone detection
            zones.update(self._detect_texture_zones(screenshot))
            
            # UI-based zone detection
            zones.update(self._detect_ui_zones(screenshot))
            
            # Boundary detection
            boundaries = self._detect_boundaries(screenshot)
            for boundary_id, boundary_data in boundaries.items():
                if boundary_id not in zones:
                    zones[boundary_id] = GameZone(
                        zone_type='boundary',
                        boundaries=boundary_data['points'],
                        center=boundary_data['center'],
                        area=boundary_data['area'],
                        color_characteristics={},
                        detected_elements=[],
                        access_points=boundary_data.get('access_points', [])
                    )
            
        except Exception as e:
            self.logger.error(f"Zone detection failed: {e}")
        
        return zones
    
    def _detect_boundaries(self, screenshot: np.ndarray) -> Dict[str, Dict]:
        """Detect game boundaries and barriers"""
        boundaries = {}
        
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Detect walls and barriers using edge detection
            edges = cv2.Canny(gray, 100, 200)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                # Group lines into boundaries
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    if abs(angle) < 15 or abs(angle) > 165:  # Horizontal
                        horizontal_lines.append(line[0])
                    elif abs(abs(angle) - 90) < 15:  # Vertical
                        vertical_lines.append(line[0])
                
                # Create boundary objects
                if horizontal_lines:
                    boundaries['horizontal_barriers'] = {
                        'points': horizontal_lines,
                        'center': self._calculate_lines_center(horizontal_lines),
                        'area': len(horizontal_lines) * 50,  # Approximate
                        'access_points': self._find_gaps_in_lines(horizontal_lines)
                    }
                
                if vertical_lines:
                    boundaries['vertical_barriers'] = {
                        'points': vertical_lines,
                        'center': self._calculate_lines_center(vertical_lines),
                        'area': len(vertical_lines) * 50,
                        'access_points': self._find_gaps_in_lines(vertical_lines)
                    }
            
            # Detect water/lava boundaries using color
            water_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            lava_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
            
            for mask, boundary_type in [(water_mask, 'water'), (lava_mask, 'lava')]:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Significant area
                        moments = cv2.moments(contour)
                        if moments['m00'] != 0:
                            center_x = int(moments['m10'] / moments['m00'])
                            center_y = int(moments['m01'] / moments['m00'])
                            
                            boundaries[f'{boundary_type}_boundary_{len(boundaries)}'] = {
                                'points': contour.squeeze().tolist(),
                                'center': (center_x, center_y),
                                'area': area,
                                'access_points': []
                            }
            
        except Exception as e:
            self.logger.error(f"Boundary detection failed: {e}")
        
        return boundaries
    
    def learn_path(self, path_type: str, waypoints: List[Tuple[int, int]], 
                   execution_time: float, success: bool) -> str:
        """Learn a new path or update existing path knowledge"""
        try:
            path_id = f"{path_type}_{hash(str(waypoints))}"
            
            if path_id in self.learned_paths:
                # Update existing path
                path = self.learned_paths[path_id]
                total_attempts = path.success_rate * 100  # Approximate previous attempts
                
                if success:
                    new_success_rate = (path.success_rate * total_attempts + 1) / (total_attempts + 1)
                else:
                    new_success_rate = (path.success_rate * total_attempts) / (total_attempts + 1)
                
                path.success_rate = new_success_rate
                path.average_time = (path.average_time + execution_time) / 2
                
            else:
                # Create new path
                self.learned_paths[path_id] = GamePath(
                    path_id=path_id,
                    waypoints=waypoints,
                    path_type=path_type,
                    success_rate=1.0 if success else 0.0,
                    average_time=execution_time,
                    obstacles=self._detect_path_obstacles(waypoints)
                )
            
            # Mark as preferred if highly successful
            if self.learned_paths[path_id].success_rate > 0.9:
                self.learned_paths[path_id].preferred_route = True
            
            self.logger.info(f"Learned path {path_id}: {self.learned_paths[path_id].success_rate:.2f} success rate")
            return f"Path learned: {path_type} with {self.learned_paths[path_id].success_rate:.2f} success rate"
            
        except Exception as e:
            self.logger.error(f"Path learning failed: {e}")
            return f"Failed to learn path: {e}"
    
    def get_optimal_path(self, start: Tuple[int, int], end: Tuple[int, int], 
                        path_type: str = 'navigation') -> Optional[List[Tuple[int, int]]]:
        """Get optimal path between two points based on learned knowledge"""
        try:
            # Look for existing learned paths
            suitable_paths = [
                path for path in self.learned_paths.values()
                if path.path_type == path_type and path.success_rate > 0.7
            ]
            
            if suitable_paths:
                # Find path with start/end points closest to desired points
                best_path = None
                best_score = float('inf')
                
                for path in suitable_paths:
                    start_dist = self._distance(start, path.waypoints[0])
                    end_dist = self._distance(end, path.waypoints[-1])
                    score = start_dist + end_dist - (path.success_rate * 100)
                    
                    if score < best_score:
                        best_score = score
                        best_path = path
                
                if best_path:
                    return best_path.waypoints
            
            # Generate new path using A* with learned obstacles
            return self._generate_path_with_obstacles(start, end)
            
        except Exception as e:
            self.logger.error(f"Optimal path calculation failed: {e}")
            return None
    
    def detect_minigame(self, screenshot: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect if a minigame or special event is active"""
        if screenshot is None:
            screenshot = self.last_screenshot
            
        if screenshot is None:
            return {'detected': False, 'type': 'none'}
        
        try:
            minigame_info = {
                'detected': False,
                'type': 'none',
                'confidence': 0.0,
                'ui_elements': [],
                'instructions': '',
                'time_limit': None
            }
            
            # Analyze UI patterns for minigame indicators
            ui_analysis = self._analyze_ui_patterns(screenshot)
            
            # Check for common minigame patterns
            if self._has_timer_ui(screenshot):
                minigame_info['detected'] = True
                minigame_info['type'] = 'timed_event'
                minigame_info['confidence'] = 0.8
                minigame_info['time_limit'] = self._extract_timer_value(screenshot)
            
            if self._has_puzzle_elements(screenshot):
                minigame_info['detected'] = True
                minigame_info['type'] = 'puzzle'
                minigame_info['confidence'] = 0.7
            
            if self._has_combat_ui(screenshot):
                minigame_info['detected'] = True
                minigame_info['type'] = 'combat'
                minigame_info['confidence'] = 0.9
            
            if self._has_dialogue_box(screenshot):
                minigame_info['detected'] = True
                minigame_info['type'] = 'dialogue'
                minigame_info['confidence'] = 0.8
                minigame_info['instructions'] = self._extract_dialogue_text(screenshot)
            
            return minigame_info
            
        except Exception as e:
            self.logger.error(f"Minigame detection failed: {e}")
            return {'detected': False, 'type': 'error', 'error': str(e)}
    
    def analyze_game_state(self) -> Dict[str, Any]:
        """Comprehensive game state analysis"""
        if not self.last_screenshot is not None:
            return {'error': 'No screenshot available'}
        
        try:
            game_state = {
                'timestamp': time.time(),
                'screen_analysis': {},
                'detected_elements': {},
                'active_zones': {},
                'navigation_context': {},
                'minigame_status': {},
                'ui_state': {},
                'recommended_actions': []
            }
            
            # Comprehensive element detection
            game_state['detected_elements'] = self._detect_all_elements(self.last_screenshot)
            
            # Zone analysis
            game_state['active_zones'] = self._detect_game_zones(self.last_screenshot)
            
            # Navigation context
            game_state['navigation_context'] = self._analyze_navigation_context(self.last_screenshot)
            
            # Minigame detection
            game_state['minigame_status'] = self.detect_minigame(self.last_screenshot)
            
            # UI state analysis
            game_state['ui_state'] = self._analyze_ui_patterns(self.last_screenshot)
            
            # Generate action recommendations
            game_state['recommended_actions'] = self._generate_action_recommendations(game_state)
            
            return game_state
            
        except Exception as e:
            self.logger.error(f"Game state analysis failed: {e}")
            return {'error': str(e)}
    
    # Helper methods for various detection algorithms
    def _color_based_detection(self, screenshot: np.ndarray, element_type: str) -> List[GameElement]:
        """Color-based element detection"""
        elements = []
        
        # Define color ranges for different element types
        color_ranges = {
            'chests': [(np.array([10, 100, 100]), np.array([30, 255, 255]))],  # Golden
            'eggs': [(np.array([0, 100, 100]), np.array([10, 255, 255]))],     # Red/Pink
            'breakables': [(np.array([100, 50, 50]), np.array([130, 255, 255]))], # Blue
            'npcs': [(np.array([20, 50, 50]), np.array([40, 255, 255]))],      # Green
            'doors': [(np.array([60, 50, 50]), np.array([80, 255, 255]))],     # Cyan
            'collectibles': [(np.array([140, 100, 100]), np.array([180, 255, 255]))] # Purple
        }
        
        if element_type not in color_ranges:
            return elements
        
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        
        for lower, upper in color_ranges[element_type]:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 2000:  # Size filter
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    element = GameElement(
                        element_type=element_type,
                        position=(x + w//2, y + h//2),
                        confidence=0.7,
                        size=(w, h),
                        color_signature=self._get_color_signature(screenshot[y:y+h, x:x+w]),
                        shape_features=self._extract_element_features(screenshot[y:y+h, x:x+w]),
                        timestamp=time.time()
                    )
                    elements.append(element)
        
        return elements
    
    def _generate_action_recommendations(self, game_state: Dict[str, Any]) -> List[str]:
        """Generate intelligent action recommendations based on game state"""
        recommendations = []
        
        try:
            elements = game_state.get('detected_elements', {})
            zones = game_state.get('active_zones', {})
            minigame = game_state.get('minigame_status', {})
            
            # Priority: Handle active minigames first
            if minigame.get('detected', False):
                minigame_type = minigame.get('type', 'unknown')
                if minigame_type == 'timed_event':
                    recommendations.append(f"URGENT: Complete timed event (Time limit: {minigame.get('time_limit', 'unknown')})")
                elif minigame_type == 'combat':
                    recommendations.append("COMBAT: Engage in combat minigame")
                elif minigame_type == 'dialogue':
                    recommendations.append(f"DIALOGUE: {minigame.get('instructions', 'Respond to dialogue')}")
                elif minigame_type == 'puzzle':
                    recommendations.append("PUZZLE: Solve puzzle minigame")
            
            # Resource collection recommendations
            if 'chests' in elements and elements['chests']:
                chest_count = len(elements['chests'])
                recommendations.append(f"COLLECT: {chest_count} chest(s) available for opening")
            
            if 'eggs' in elements and elements['eggs']:
                egg_count = len(elements['eggs'])
                recommendations.append(f"HATCH: {egg_count} egg(s) ready for hatching")
            
            if 'collectibles' in elements and elements['collectibles']:
                collectible_count = len(elements['collectibles'])
                recommendations.append(f"GATHER: {collectible_count} collectible item(s) nearby")
            
            # Zone-based recommendations
            if 'breakables' in zones:
                recommendations.append("FARM: Breakables zone detected - good for farming")
            
            if 'safe_zone' in zones:
                recommendations.append("SAFE: Safe zone available for resting/planning")
            
            # Navigation recommendations
            nav_context = game_state.get('navigation_context', {})
            if nav_context.get('blocked_paths'):
                recommendations.append("NAVIGATE: Find alternative route - path blocked")
            
            if nav_context.get('optimal_farming_path'):
                recommendations.append("OPTIMIZE: Optimal farming path available")
            
            # Resource management
            if 'resources' in elements and len(elements['resources']) > 10:
                recommendations.append("MANAGE: High resource density - consider inventory management")
            
            # Default exploration if no specific actions
            if not recommendations:
                recommendations.append("EXPLORE: No immediate objectives - continue exploration")
            
        except Exception as e:
            self.logger.error(f"Action recommendation generation failed: {e}")
            recommendations.append(f"ERROR: Could not generate recommendations - {e}")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    # Additional helper methods for various detection algorithms
    def _shape_based_detection(self, screenshot: np.ndarray, element_type: str) -> List[GameElement]:
        """Shape-based element detection"""
        # Placeholder for shape-based detection
        return []
    
    def _template_based_detection(self, screenshot: np.ndarray, element_type: str) -> List[GameElement]:
        """Template-based element detection"""
        # Placeholder for template matching
        return []
    
    def _filter_and_validate_elements(self, candidates: List[GameElement], element_type: str) -> List[GameElement]:
        """Filter and validate detected elements"""
        # Remove duplicates and validate detections
        valid_elements = []
        for element in candidates:
            if element.confidence > self.detection_sensitivity:
                valid_elements.append(element)
        return valid_elements
    
    def _extract_element_features(self, roi: np.ndarray) -> Dict[str, float]:
        """Extract shape and texture features from element ROI"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Basic shape features
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 1
                
                return {
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'compactness': area / (w * h) if w * h > 0 else 0
                }
        except:
            pass
        
        return {'area': 0, 'perimeter': 0, 'circularity': 0, 'aspect_ratio': 1, 'compactness': 0}
    
    def _get_color_signature(self, roi: np.ndarray) -> List[int]:
        """Get color signature of ROI"""
        try:
            # Get dominant colors
            roi_resized = cv2.resize(roi, (50, 50))
            colors = roi_resized.reshape(-1, 3)
            dominant_color = np.mean(colors, axis=0).astype(int)
            return dominant_color.tolist()
        except:
            return [0, 0, 0]
    
    def _is_known_element_type(self, features: Dict[str, float]) -> bool:
        """Check if element features match known element types"""
        # Compare with learned element patterns
        return False  # Placeholder
    
    def _detect_color_zones(self, screenshot: np.ndarray) -> Dict[str, GameZone]:
        """Detect zones based on color characteristics"""
        return {}  # Placeholder
    
    def _detect_texture_zones(self, screenshot: np.ndarray) -> Dict[str, GameZone]:
        """Detect zones based on texture patterns"""
        return {}  # Placeholder
    
    def _detect_ui_zones(self, screenshot: np.ndarray) -> Dict[str, GameZone]:
        """Detect UI-defined zones"""
        return {}  # Placeholder
    
    def _analyze_ui_patterns(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyze UI patterns and changes"""
        return {}  # Placeholder
    
    def _detect_game_events(self, screenshot: np.ndarray) -> List[str]:
        """Detect game events"""
        return []  # Placeholder
    
    def _analyze_navigation_context(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyze navigation context"""
        return {}  # Placeholder
    
    def _update_learned_knowledge(self, analysis_results: Dict[str, Any]):
        """Update learned knowledge based on analysis"""
        pass  # Placeholder
    
    def _detect_ui_elements(self, screenshot: np.ndarray) -> List[GameElement]:
        """Detect UI elements"""
        return []  # Placeholder
    
    def _detect_interactive_elements(self, screenshot: np.ndarray) -> List[GameElement]:
        """Detect interactive elements"""
        return []  # Placeholder
    
    def _detect_resource_nodes(self, screenshot: np.ndarray) -> List[GameElement]:
        """Detect resource nodes"""
        return []  # Placeholder
    
    def _detect_characters(self, screenshot: np.ndarray) -> List[GameElement]:
        """Detect characters/NPCs"""
        return []  # Placeholder
    
    def _calculate_lines_center(self, lines: List[List[int]]) -> Tuple[int, int]:
        """Calculate center point of lines"""
        if not lines:
            return (0, 0)
        
        x_coords = []
        y_coords = []
        for line in lines:
            x_coords.extend([line[0], line[2]])
            y_coords.extend([line[1], line[3]])
        
        return (int(np.mean(x_coords)), int(np.mean(y_coords)))
    
    def _find_gaps_in_lines(self, lines: List[List[int]]) -> List[Tuple[int, int]]:
        """Find gaps in line barriers for access points"""
        return []  # Placeholder
    
    def _detect_path_obstacles(self, waypoints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Detect obstacles along a path"""
        return []  # Placeholder
    
    def _distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _generate_path_with_obstacles(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate path avoiding known obstacles"""
        # Simple straight line for placeholder
        return [start, end]
    
    def _has_timer_ui(self, screenshot: np.ndarray) -> bool:
        """Check for timer UI elements"""
        return False  # Placeholder
    
    def _extract_timer_value(self, screenshot: np.ndarray) -> Optional[int]:
        """Extract timer value from UI"""
        return None  # Placeholder
    
    def _has_puzzle_elements(self, screenshot: np.ndarray) -> bool:
        """Check for puzzle UI elements"""
        return False  # Placeholder
    
    def _has_combat_ui(self, screenshot: np.ndarray) -> bool:
        """Check for combat UI elements"""
        return False  # Placeholder
    
    def _has_dialogue_box(self, screenshot: np.ndarray) -> bool:
        """Check for dialogue box"""
        return False  # Placeholder
    
    def _extract_dialogue_text(self, screenshot: np.ndarray) -> str:
        """Extract text from dialogue box"""
        return ""  # Placeholder
    
    def _load_element_templates(self) -> Dict[str, Any]:
        """Load element templates"""
        return {}  # Placeholder
    
    def _save_learned_data(self):
        """Save learned vision data"""
        try:
            data = {
                'learned_paths': {k: {
                    'path_id': v.path_id,
                    'waypoints': v.waypoints,
                    'path_type': v.path_type,
                    'success_rate': v.success_rate,
                    'average_time': v.average_time,
                    'obstacles': v.obstacles,
                    'preferred_route': v.preferred_route
                } for k, v in self.learned_paths.items()},
                'game_zones': self.game_zones,
                'minigame_signatures': self.minigame_signatures,
                'ui_patterns': self.ui_patterns
            }
            
            save_path = Path("data/enhanced_vision")
            save_path.mkdir(exist_ok=True)
            
            with open(save_path / "learned_data.json", 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info("Enhanced vision data saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save vision data: {e}")