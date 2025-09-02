"""
Interactive Training System for AI Game Bot
Allows users to teach the bot through direct interaction and mapping
"""

import time
import json
import logging
import threading
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque

# Handle PyAutoGUI import for headless environments
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception as e:
    PYAUTOGUI_AVAILABLE = False
    class MockPyAutoGUI:
        @staticmethod
        def position():
            return (400, 300)  # Mock position
        @staticmethod
        def screenshot(region=None):
            # Return a mock PIL Image
            from PIL import Image
            return Image.new('RGB', (100, 100), color='blue')
    pyautogui = MockPyAutoGUI()

@dataclass
class InteractiveItem:
    """Represents an item learned through interaction"""
    item_id: str
    item_type: str  # 'chest', 'egg', 'breakable', 'resource', etc.
    screenshot: str  # base64 encoded image
    position: Tuple[int, int]
    size: Tuple[int, int]
    color_signature: List[int]  # RGB color signature
    learned_at: float
    confidence_score: float
    metadata: Dict[str, Any]

@dataclass
class GameZone:
    """Represents a mapped game zone"""
    zone_id: str
    zone_type: str  # 'breakables', 'fishing', 'safe', 'boundary'
    corners: List[Tuple[int, int]]  # 4 corner coordinates
    center: Tuple[int, int]
    restrictions: List[str]  # 'stay_inside', 'avoid', 'farming_zone'
    learned_items: List[str]  # item_ids found in this zone
    created_at: float
    last_used: float

class InteractiveTrainer:
    """System for interactive training and mapping"""
    
    def __init__(self, enhanced_vision=None, automation_engine=None):
        self.logger = logging.getLogger(__name__)
        
        # Core systems
        self.enhanced_vision = enhanced_vision
        self.automation_engine = automation_engine
        
        # Training data storage
        self.learned_items = {}
        self.game_zones = {}
        self.training_sessions = deque(maxlen=100)
        
        # Interactive state
        self.training_mode = False
        self.current_mode = "idle"  # 'mapping', 'item_learning', 'zone_creation'
        self.zone_creation_corners = []
        self.pending_item_type = None
        
        # File paths
        self.items_file = Path("data/learned_items.json")
        self.zones_file = Path("data/game_zones.json")
        
        # Load existing data
        self._load_training_data()
        
        # Natural language processing for commands
        self.command_patterns = {
            'map_area': ['map', 'area', 'zone', 'region', 'boundary'],
            'learn_item': ['learn', 'teach', 'store', 'remember', 'save'],
            'create_zone': ['create', 'zone', 'boundary', 'area', 'safe'],
            'start_recording': ['record', 'watch', 'observe', 'learn gameplay'],
            'analyze_similarity': ['similar', 'recognize', 'identify', 'detect']
        }
        
        self.logger.info("Interactive Trainer initialized")
    
    def _load_training_data(self):
        """Load previously learned items and zones"""
        try:
            # Load learned items
            if self.items_file.exists():
                with open(self.items_file, 'r') as f:
                    items_data = json.load(f)
                    for item_id, item_data in items_data.items():
                        self.learned_items[item_id] = InteractiveItem(**item_data)
                self.logger.info(f"Loaded {len(self.learned_items)} learned items")
            
            # Load game zones
            if self.zones_file.exists():
                with open(self.zones_file, 'r') as f:
                    zones_data = json.load(f)
                    for zone_id, zone_data in zones_data.items():
                        self.game_zones[zone_id] = GameZone(**zone_data)
                self.logger.info(f"Loaded {len(self.game_zones)} game zones")
                
        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
    
    def _save_training_data(self):
        """Save learned items and zones to files"""
        try:
            # Save learned items
            self.items_file.parent.mkdir(exist_ok=True)
            items_data = {item_id: asdict(item) for item_id, item in self.learned_items.items()}
            with open(self.items_file, 'w') as f:
                json.dump(items_data, f, indent=2)
            
            # Save game zones
            self.zones_file.parent.mkdir(exist_ok=True)
            zones_data = {zone_id: asdict(zone) for zone_id, zone in self.game_zones.items()}
            with open(self.zones_file, 'w') as f:
                json.dump(zones_data, f, indent=2)
                
            self.logger.debug("Training data saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save training data: {e}")
    
    def start_interactive_training(self, mode: str = "item_learning") -> str:
        """Start interactive training mode"""
        try:
            self.training_mode = True
            self.current_mode = mode
            
            if mode == "item_learning":
                return "Interactive item learning started! Hover over items and press SPACE to store them."
            elif mode == "zone_mapping":
                self.zone_creation_corners = []
                return "Zone mapping started! Click 4 corners to define a zone boundary."
            elif mode == "gameplay_recording":
                return "Gameplay recording started! The bot will watch and learn from your actions."
            else:
                return f"Unknown training mode: {mode}"
                
        except Exception as e:
            error_msg = f"Failed to start interactive training: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def stop_interactive_training(self) -> str:
        """Stop interactive training mode"""
        try:
            self.training_mode = False
            self.current_mode = "idle"
            self.zone_creation_corners = []
            self.pending_item_type = None
            
            self._save_training_data()
            
            return "Interactive training stopped and data saved."
            
        except Exception as e:
            error_msg = f"Failed to stop interactive training: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def process_spacebar_input(self, item_type: Optional[str] = None) -> str:
        """Process spacebar input for item learning"""
        try:
            if not self.training_mode or self.current_mode != "item_learning":
                return "Not in item learning mode"
            
            # Get current mouse position
            mouse_x, mouse_y = pyautogui.position()
            
            # Take screenshot of area around mouse
            screenshot_region = (int(mouse_x - 50), int(mouse_y - 50), 100, 100)
            screenshot = pyautogui.screenshot(region=screenshot_region)
            
            # Convert to OpenCV format
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Extract color signature
            color_signature = self._extract_color_signature(screenshot_cv)
            
            # Create item ID
            item_id = f"item_{int(time.time())}_{mouse_x}_{mouse_y}"
            
            # Save screenshot as base64
            import base64
            from io import BytesIO
            buffer = BytesIO()
            screenshot.save(buffer, format='PNG')
            screenshot_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create interactive item
            interactive_item = InteractiveItem(
                item_id=item_id,
                item_type=item_type or self.pending_item_type or "unknown",
                screenshot=screenshot_b64,
                position=(int(mouse_x), int(mouse_y)),
                size=(100, 100),
                color_signature=color_signature,
                learned_at=time.time(),
                confidence_score=1.0,
                metadata={
                    'learning_method': 'interactive_spacebar',
                    'screen_region': screenshot_region
                }
            )
            
            # Store the item
            self.learned_items[item_id] = interactive_item
            
            # Clear pending item type
            self.pending_item_type = None
            
            result = f"Learned new {interactive_item.item_type} at position ({mouse_x}, {mouse_y})"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to process spacebar input: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def process_corner_click(self, x: int, y: int) -> str:
        """Process corner click for zone creation"""
        try:
            if not self.training_mode or self.current_mode != "zone_mapping":
                return "Not in zone mapping mode"
            
            # Add corner to list
            self.zone_creation_corners.append((x, y))
            
            corners_needed = 4 - len(self.zone_creation_corners)
            
            if corners_needed > 0:
                return f"Corner {len(self.zone_creation_corners)} recorded. {corners_needed} more corners needed."
            else:
                # Create the zone
                return self._create_zone_from_corners()
                
        except Exception as e:
            error_msg = f"Failed to process corner click: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _create_zone_from_corners(self) -> str:
        """Create a game zone from the collected corners"""
        try:
            if len(self.zone_creation_corners) != 4:
                return "Need exactly 4 corners to create zone"
            
            # Calculate center point
            center_x = sum(corner[0] for corner in self.zone_creation_corners) // 4
            center_y = sum(corner[1] for corner in self.zone_creation_corners) // 4
            
            # Create zone ID
            zone_id = f"zone_{int(time.time())}_{center_x}_{center_y}"
            
            # Create game zone
            game_zone = GameZone(
                zone_id=zone_id,
                zone_type="custom",  # User can specify this later
                corners=self.zone_creation_corners.copy(),
                center=(center_x, center_y),
                restrictions=["stay_inside"],  # Default restriction
                learned_items=[],
                created_at=time.time(),
                last_used=0
            )
            
            # Store the zone
            self.game_zones[zone_id] = game_zone
            
            # Reset corners
            self.zone_creation_corners = []
            
            result = f"Created zone '{zone_id}' with center at ({center_x}, {center_y})"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to create zone: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _extract_color_signature(self, image: np.ndarray) -> List[int]:
        """Extract color signature from image"""
        try:
            # Calculate average color
            avg_color = np.mean(image, axis=(0, 1))
            return [int(c) for c in avg_color]
        except Exception as e:
            self.logger.error(f"Color signature extraction failed: {e}")
            return [0, 0, 0]
    
    def process_natural_language_command(self, command: str) -> str:
        """Process natural language commands for training"""
        try:
            command_lower = command.lower()
            
            # Detect command intent
            intent = self._detect_command_intent(command_lower)
            
            if intent == 'map_area':
                return self.start_interactive_training("zone_mapping")
            elif intent == 'learn_item':
                # Extract item type from command
                item_type = self._extract_item_type(command_lower)
                self.pending_item_type = item_type
                return self.start_interactive_training("item_learning")
            elif intent == 'create_zone':
                return self.start_interactive_training("zone_mapping")
            elif intent == 'start_recording':
                return self.start_interactive_training("gameplay_recording")
            elif intent == 'analyze_similarity':
                return self._analyze_item_similarities()
            else:
                # Try to process as regular bot command
                return f"Command processed: {command}"
                
        except Exception as e:
            error_msg = f"Failed to process command: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _detect_command_intent(self, command: str) -> str:
        """Detect the intent of a natural language command"""
        for intent, keywords in self.command_patterns.items():
            if any(keyword in command for keyword in keywords):
                return intent
        return 'unknown'
    
    def _extract_item_type(self, command: str) -> str:
        """Extract item type from command"""
        item_types = ['chest', 'egg', 'breakable', 'resource', 'pet', 'coin']
        for item_type in item_types:
            if item_type in command:
                return item_type
        return 'unknown'
    
    def _analyze_item_similarities(self) -> str:
        """Analyze similarities between learned items"""
        try:
            if len(self.learned_items) < 2:
                return "Need at least 2 learned items to analyze similarities"
            
            similarities = []
            items_list = list(self.learned_items.values())
            
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    item1, item2 = items_list[i], items_list[j]
                    
                    # Calculate color similarity
                    color_similarity = self._calculate_color_similarity(
                        item1.color_signature, item2.color_signature
                    )
                    
                    if color_similarity > 0.7:  # 70% similarity threshold
                        similarities.append({
                            'item1': item1.item_id,
                            'item2': item2.item_id,
                            'similarity': color_similarity,
                            'type1': item1.item_type,
                            'type2': item2.item_type
                        })
            
            if similarities:
                result = f"Found {len(similarities)} similar items:\n"
                for sim in similarities[:5]:  # Show top 5
                    result += f"- {sim['type1']} vs {sim['type2']}: {sim['similarity']:.2f} similarity\n"
                return result
            else:
                return "No significant similarities found between learned items"
                
        except Exception as e:
            error_msg = f"Similarity analysis failed: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _calculate_color_similarity(self, color1: List[int], color2: List[int]) -> float:
        """Calculate similarity between two color signatures"""
        try:
            if len(color1) != len(color2):
                return 0.0
            
            # Calculate normalized difference
            diff = sum(abs(c1 - c2) for c1, c2 in zip(color1, color2))
            max_diff = sum(255 for _ in color1)  # Maximum possible difference
            
            similarity = 1 - (diff / max_diff)
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Color similarity calculation failed: {e}")
            return 0.0
    
    def find_similar_items_in_screen(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find items in current screen that match learned items"""
        try:
            if not self.learned_items:
                return []
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            matches = []
            
            for item_id, learned_item in self.learned_items.items():
                # Decode the stored screenshot
                import base64
                from PIL import Image
                from io import BytesIO
                
                img_data = base64.b64decode(learned_item.screenshot)
                template = Image.open(BytesIO(img_data))
                template_cv = cv2.cvtColor(np.array(template), cv2.COLOR_RGB2BGR)
                
                # Template matching
                result = cv2.matchTemplate(screenshot_cv, template_cv, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):  # Switch x and y
                    matches.append({
                        'item_id': item_id,
                        'item_type': learned_item.item_type,
                        'position': pt,
                        'confidence': float(result[pt[1], pt[0]]),
                        'original_item': learned_item
                    })
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Screen matching failed: {e}")
            return []
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'training_mode': self.training_mode,
            'current_mode': self.current_mode,
            'learned_items_count': len(self.learned_items),
            'game_zones_count': len(self.game_zones),
            'pending_item_type': self.pending_item_type,
            'zone_corners_collected': len(self.zone_creation_corners),
            'recent_items': [
                {
                    'id': item.item_id,
                    'type': item.item_type,
                    'position': item.position,
                    'learned_at': item.learned_at
                }
                for item in list(self.learned_items.values())[-5:]
            ]
        }
    
    def set_zone_type(self, zone_id: str, zone_type: str, restrictions: List[str]) -> str:
        """Set the type and restrictions for a created zone"""
        try:
            if zone_id not in self.game_zones:
                return f"Zone {zone_id} not found"
            
            zone = self.game_zones[zone_id]
            zone.zone_type = zone_type
            zone.restrictions = restrictions
            
            self._save_training_data()
            
            return f"Zone {zone_id} updated: type={zone_type}, restrictions={restrictions}"
            
        except Exception as e:
            error_msg = f"Failed to set zone type: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def export_training_data(self) -> Dict[str, Any]:
        """Export all training data for backup or sharing"""
        return {
            'learned_items': {item_id: asdict(item) for item_id, item in self.learned_items.items()},
            'game_zones': {zone_id: asdict(zone) for zone_id, zone in self.game_zones.items()},
            'export_timestamp': time.time()
        }