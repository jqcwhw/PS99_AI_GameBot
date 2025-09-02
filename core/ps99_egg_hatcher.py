"""
PS99 Egg Hatching System
Real egg hatching functionality that interacts with Pet Simulator 99
"""

import cv2
import numpy as np
import pyautogui
import requests
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue

class PS99EggHatcher:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.eggs_data = {}
        self.pets_data = {}
        self.target_pets = set()  # Pets user wants to hatch
        self.hatching_results = []
        self.is_hatching = False
        self.hatch_thread = None
        self.detection_enabled = True
        
        # Screen detection settings
        self.game_window = None
        self.last_screenshot = None
        
        # Load PS99 data
        self.load_ps99_data()
        
        # Initialize pet detection templates
        self.pet_templates = {}
        self.load_pet_templates()
        
    def load_ps99_data(self):
        """Load eggs and pets data from PS99 API"""
        try:
            # Load eggs data
            response = requests.get('https://ps99.biggamesapi.io/api/collection/eggs', timeout=10)
            if response.status_code == 200:
                self.eggs_data = response.json()
                self.logger.info(f"Loaded {len(self.eggs_data)} eggs from PS99 API")
            else:
                self.logger.error(f"Failed to load eggs data: {response.status_code}")
                
            # Load pets data  
            response = requests.get('https://ps99.biggamesapi.io/api/collection/pets', timeout=10)
            if response.status_code == 200:
                self.pets_data = response.json()
                self.logger.info(f"Loaded {len(self.pets_data)} pets from PS99 API")
            else:
                self.logger.error(f"Failed to load pets data: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error loading PS99 data: {e}")
            
    def load_pet_templates(self):
        """Load pet image templates for detection"""
        templates_dir = Path("data/pet_templates")
        templates_dir.mkdir(exist_ok=True)
        
        try:
            for template_file in templates_dir.glob("*.png"):
                pet_name = template_file.stem
                template = cv2.imread(str(template_file))
                if template is not None:
                    self.pet_templates[pet_name] = template
                    
            self.logger.info(f"Loaded {len(self.pet_templates)} pet templates")
        except Exception as e:
            self.logger.error(f"Error loading pet templates: {e}")
            
    def find_game_window(self) -> bool:
        """Find and focus Pet Simulator 99 game window"""
        try:
            # Try to find Roblox window
            windows = pyautogui.getWindowsWithTitle("Roblox")
            if windows:
                self.game_window = windows[0]
                self.game_window.activate()
                self.logger.info("Found and focused Roblox window")
                return True
            else:
                self.logger.warning("Roblox window not found")
                return False
        except Exception as e:
            self.logger.error(f"Error finding game window: {e}")
            return False
            
    def take_screenshot(self) -> Optional[np.ndarray]:
        """Take screenshot of game window"""
        try:
            if not self.game_window:
                self.find_game_window()
                
            if self.game_window:
                # Get window region
                left, top, width, height = self.game_window.left, self.game_window.top, self.game_window.width, self.game_window.height
                screenshot = pyautogui.screenshot(region=(left, top, width, height))
                
                # Convert to OpenCV format
                screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                self.last_screenshot = screenshot_cv
                return screenshot_cv
            else:
                self.logger.warning("No game window available for screenshot")
                return None
                
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            return None
            
    def detect_eggs_on_screen(self, screenshot: np.ndarray) -> List[Dict[str, Any]]:
        """Detect eggs on screen using computer vision"""
        detected_eggs = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Define egg color ranges (these would need to be calibrated for PS99)
            egg_color_ranges = [
                # Common Egg (gray/white)
                {'name': 'Common Egg', 'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
                # Uncommon Egg (green)
                {'name': 'Uncommon Egg', 'lower': np.array([50, 100, 100]), 'upper': np.array([70, 255, 255])},
                # Rare Egg (blue)
                {'name': 'Rare Egg', 'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
                # Epic Egg (purple)
                {'name': 'Epic Egg', 'lower': np.array([130, 100, 100]), 'upper': np.array([160, 255, 255])},
                # Legendary Egg (gold/yellow)
                {'name': 'Legendary Egg', 'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
            ]
            
            for egg_range in egg_color_ranges:
                # Create mask for egg color
                mask = cv2.inRange(hsv, egg_range['lower'], egg_range['upper'])
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum egg size
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        detected_eggs.append({
                            'name': egg_range['name'],
                            'position': (center_x, center_y),
                            'area': area,
                            'bounds': (x, y, w, h)
                        })
                        
        except Exception as e:
            self.logger.error(f"Error detecting eggs: {e}")
            
        return detected_eggs
        
    def detect_hatched_pets(self, screenshot: np.ndarray) -> List[Dict[str, Any]]:
        """Detect pets that were just hatched using template matching"""
        detected_pets = []
        
        try:
            # Look for pet popup/notification areas (would need calibration for PS99 UI)
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Template matching for known pet appearances
            for pet_name, template in self.pet_templates.items():
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
                
                threshold = 0.8
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    detected_pets.append({
                        'name': pet_name,
                        'position': pt,
                        'confidence': result[pt[1], pt[0]],
                        'timestamp': datetime.now()
                    })
                    
                    # Check if this is a target pet
                    if pet_name in self.target_pets:
                        self.logger.info(f"🎉 TARGET PET HATCHED: {pet_name}!")
                        self.notify_target_pet_hatched(pet_name)
                        
        except Exception as e:
            self.logger.error(f"Error detecting hatched pets: {e}")
            
        return detected_pets
        
    def move_to_egg(self, egg_position: Tuple[int, int]) -> bool:
        """Move character to egg position and interact"""
        try:
            if not self.game_window:
                return False
                
            # Calculate screen coordinates relative to game window
            window_x = self.game_window.left + egg_position[0]
            window_y = self.game_window.top + egg_position[1]
            
            # Click to move to egg
            pyautogui.click(window_x, window_y)
            time.sleep(1)  # Wait for movement
            
            # Try to interact with egg (E key is common for Roblox games)
            pyautogui.press('e')
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving to egg: {e}")
            return False
            
    def hatch_egg(self, egg_name: str = None) -> Dict[str, Any]:
        """Perform egg hatching sequence"""
        try:
            screenshot = self.take_screenshot()
            if screenshot is None:
                return {'success': False, 'error': 'Could not take screenshot'}
                
            # Detect eggs on screen
            detected_eggs = self.detect_eggs_on_screen(screenshot)
            
            if not detected_eggs:
                return {'success': False, 'error': 'No eggs detected on screen'}
                
            # Choose egg to hatch
            target_egg = None
            if egg_name:
                for egg in detected_eggs:
                    if egg_name.lower() in egg['name'].lower():
                        target_egg = egg
                        break
            else:
                # Use closest egg
                target_egg = min(detected_eggs, key=lambda e: e['area'])
                
            if not target_egg:
                return {'success': False, 'error': f'Egg "{egg_name}" not found'}
                
            # Move to egg and hatch
            success = self.move_to_egg(target_egg['position'])
            
            if success:
                # Wait for hatching animation and detect results
                time.sleep(3)  # Wait for hatch animation
                
                new_screenshot = self.take_screenshot()
                if new_screenshot is not None:
                    hatched_pets = self.detect_hatched_pets(new_screenshot)
                    
                    result = {
                        'success': True,
                        'egg_hatched': target_egg['name'],
                        'pets_detected': hatched_pets,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Store result
                    self.hatching_results.append(result)
                    
                    return result
                    
            return {'success': False, 'error': 'Failed to hatch egg'}
            
        except Exception as e:
            self.logger.error(f"Error hatching egg: {e}")
            return {'success': False, 'error': str(e)}
            
    def start_auto_hatch(self, egg_types: List[str] = None, target_pets: List[str] = None):
        """Start automatic egg hatching"""
        if self.is_hatching:
            self.logger.warning("Auto-hatch already running")
            return
            
        self.is_hatching = True
        if target_pets:
            self.target_pets.update(target_pets)
            
        def hatch_loop():
            while self.is_hatching:
                try:
                    result = self.hatch_egg()
                    if result['success']:
                        self.logger.info(f"Hatched {result['egg_hatched']}")
                    else:
                        self.logger.warning(f"Hatch failed: {result.get('error', 'Unknown error')}")
                        
                    time.sleep(5)  # Wait between hatches
                    
                except Exception as e:
                    self.logger.error(f"Error in hatch loop: {e}")
                    time.sleep(10)  # Wait longer on error
                    
        self.hatch_thread = threading.Thread(target=hatch_loop, daemon=True)
        self.hatch_thread.start()
        self.logger.info("Auto-hatch started")
        
    def stop_auto_hatch(self):
        """Stop automatic egg hatching"""
        self.is_hatching = False
        if self.hatch_thread:
            self.hatch_thread.join(timeout=5)
        self.logger.info("Auto-hatch stopped")
        
    def add_target_pet(self, pet_name: str):
        """Add pet to target list"""
        self.target_pets.add(pet_name)
        self.logger.info(f"Added target pet: {pet_name}")
        
    def remove_target_pet(self, pet_name: str):
        """Remove pet from target list"""
        self.target_pets.discard(pet_name)
        self.logger.info(f"Removed target pet: {pet_name}")
        
    def notify_target_pet_hatched(self, pet_name: str):
        """Send notification when target pet is hatched"""
        # This could be expanded with sound alerts, notifications, etc.
        message = f"🎉 TARGET PET HATCHED: {pet_name}!"
        self.logger.info(message)
        print(f"\n{message}\n")  # Console notification
        
    def get_egg_info(self, egg_name: str) -> Optional[Dict]:
        """Get detailed information about an egg from PS99 API data"""
        for egg_id, egg_data in self.eggs_data.items():
            if egg_name.lower() in egg_data.get('configName', '').lower():
                return {
                    'id': egg_id,
                    'name': egg_data.get('configName', 'Unknown'),
                    'pets': egg_data.get('pets', []),
                    'cost': egg_data.get('cost', 0),
                    'currency': egg_data.get('currency', 'coins')
                }
        return None
        
    def get_pet_info(self, pet_name: str) -> Optional[Dict]:
        """Get detailed information about a pet from PS99 API data"""
        for pet_id, pet_data in self.pets_data.items():
            if pet_name.lower() in pet_data.get('configName', '').lower():
                return {
                    'id': pet_id,
                    'name': pet_data.get('configName', 'Unknown'),
                    'rarity': pet_data.get('rarity', 'Unknown'),
                    'damage': pet_data.get('damage', 0),
                    'speed': pet_data.get('walkSpeed', 0)
                }
        return None
        
    def get_hatching_stats(self) -> Dict[str, Any]:
        """Get statistics about hatching results"""
        if not self.hatching_results:
            return {'total_hatches': 0, 'pets_hatched': 0, 'target_pets_found': 0}
            
        total_hatches = len(self.hatching_results)
        pets_hatched = sum(len(result.get('pets_detected', [])) for result in self.hatching_results)
        target_pets_found = sum(
            1 for result in self.hatching_results
            for pet in result.get('pets_detected', [])
            if pet['name'] in self.target_pets
        )
        
        return {
            'total_hatches': total_hatches,
            'pets_hatched': pets_hatched,
            'target_pets_found': target_pets_found,
            'success_rate': (pets_hatched / total_hatches * 100) if total_hatches > 0 else 0
        }
        
    def save_results(self, filename: str = None):
        """Save hatching results to file"""
        if not filename:
            filename = f"data/hatching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        try:
            Path(filename).parent.mkdir(exist_ok=True)
            with open(filename, 'w') as f:
                json.dump({
                    'results': self.hatching_results,
                    'target_pets': list(self.target_pets),
                    'stats': self.get_hatching_stats()
                }, f, indent=2, default=str)
                
            self.logger.info(f"Saved hatching results to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")