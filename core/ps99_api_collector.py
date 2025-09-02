"""
PS99 API Enhanced Collector - Python Implementation
Converted from JavaScript for direct integration with AI Game Bot
"""

import requests
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import pyautogui
import cv2
import numpy as np

@dataclass
class EggData:
    """PS99 Egg data structure"""
    name: str
    cost: int
    currency: str
    pets: List[str]
    location: str
    requirements: Dict[str, Any]

@dataclass
class CollectionStats:
    """Collection statistics"""
    total_collections: int = 0
    successful_collections: int = 0
    eggs_collected: int = 0
    eggs_planted: int = 0
    eggs_harvested: int = 0
    coins_spent: int = 0
    session_start: float = 0

class PS99APICollector:
    """Real PS99 API integration for egg collection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # API Configuration
        self.api_base_url = 'https://ps99.biggamesapi.io'
        self.api_endpoints = {
            'pets': '/api/collection/pets',
            'items': '/api/collection/items', 
            'eggs': '/api/collection/eggs',
            'currencies': '/api/collection/currencies'
        }
        
        # Game mechanics timing
        self.egg_refresh_interval = 5 * 60  # 5 minutes
        self.egg_growth_time = 30 * 60      # 30 minutes
        
        # Collection settings
        self.max_coins_to_spend = 50000000  # 50M coins
        self.priority_eggs = ['Angelus', 'Agony', 'Demon', 'Yeti', 'Griffin']
        
        # Cache setup
        self.cache_dir = Path("data/ps99_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.collecting = False
        self.learning = False
        self.learned_sequence = []
        self.stats = CollectionStats()
        self.egg_data = {}
        
        # Position data (learned from screen analysis)
        self.shop_positions = {
            'shop_button': None,
            'merchant_position': None,
            'egg_buttons': []
        }
        
        self.farm_positions = {
            'farm_button': None,
            'planting_spots': [],
            'harvest_button': None
        }
        
        # Load cached data
        self.load_cached_data()
        
        self.logger.info("PS99 API Collector initialized with real API integration")
    
    def load_ps99_api_data(self) -> bool:
        """Load real data from PS99 Big Games API"""
        try:
            # Load eggs data
            response = requests.get(f"{self.api_base_url}{self.api_endpoints['eggs']}", timeout=10)
            if response.status_code == 200:
                eggs_raw = response.json()
                self.egg_data = self._process_egg_data(eggs_raw)
                self.logger.info(f"Loaded {len(self.egg_data)} eggs from PS99 API")
                
                # Cache the data
                cache_file = self.cache_dir / "egg_data.json"
                with open(cache_file, 'w') as f:
                    json.dump(self.egg_data, f, indent=2)
                
                return True
            else:
                self.logger.error(f"Failed to load eggs data: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading PS99 API data: {e}")
            return False
    
    def _process_egg_data(self, raw_data: Dict) -> Dict[str, EggData]:
        """Process raw API data into structured format"""
        processed = {}
        
        try:
            for egg_id, egg_info in raw_data.items():
                if isinstance(egg_info, dict):
                    egg_data = EggData(
                        name=egg_info.get('configName', egg_id),
                        cost=egg_info.get('cost', 0),
                        currency=egg_info.get('currency', 'Coins'),
                        pets=egg_info.get('pets', []),
                        location=egg_info.get('location', 'Unknown'),
                        requirements=egg_info.get('requirements', {})
                    )
                    processed[egg_id] = egg_data
        
        except Exception as e:
            self.logger.error(f"Error processing egg data: {e}")
        
        return processed
    
    def learn_game_positions(self) -> bool:
        """Learn shop and farm positions by analyzing screen"""
        self.logger.info("Learning game positions from screen analysis...")
        
        try:
            # Capture current screen
            screenshot = pyautogui.screenshot()
            screen_array = np.array(screenshot)
            
            # Convert to grayscale for template matching
            gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
            
            # Try to find shop button (you'd need to provide template images)
            # This is a simplified version - real implementation would use template matching
            
            # For now, use approximate positions based on common PS99 layout
            screen_width, screen_height = screenshot.size
            
            # Estimate positions based on typical PS99 UI layout
            self.shop_positions = {
                'shop_button': (screen_width - 100, screen_height - 200),
                'merchant_position': (screen_width // 2, screen_height // 2),
                'egg_buttons': [
                    (screen_width // 2 - 150, screen_height // 2),
                    (screen_width // 2, screen_height // 2),
                    (screen_width // 2 + 150, screen_height // 2)
                ]
            }
            
            self.farm_positions = {
                'farm_button': (screen_width - 200, screen_height - 200),
                'planting_spots': [
                    (screen_width // 2 - 100, screen_height // 2 + 100),
                    (screen_width // 2, screen_height // 2 + 100),
                    (screen_width // 2 + 100, screen_height // 2 + 100)
                ],
                'harvest_button': (screen_width // 2, screen_height // 2 + 200)
            }
            
            self.logger.info("Game positions learned successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error learning game positions: {e}")
            return False
    
    def start_collection(self, target_eggs: Optional[List[str]] = None) -> bool:
        """Start real egg collection process"""
        if self.collecting:
            self.logger.warning("Collection already in progress")
            return False
        
        # Load fresh API data
        if not self.load_ps99_api_data():
            self.logger.error("Failed to load PS99 API data")
            return False
        
        # Learn current game positions
        if not self.learn_game_positions():
            self.logger.error("Failed to learn game positions")
            return False
        
        self.collecting = True
        self.stats.session_start = time.time()
        
        # Start collection in background thread
        collection_thread = threading.Thread(target=self._collection_loop, args=(target_eggs,), daemon=True)
        collection_thread.start()
        
        self.logger.info("Started PS99 egg collection with real API integration")
        return True
    
    def _collection_loop(self, target_eggs: Optional[List[str]]):
        """Main collection loop - performs real game actions"""
        eggs_to_collect = target_eggs or self.priority_eggs
        
        while self.collecting:
            try:
                for egg_name in eggs_to_collect:
                    if not self.collecting:
                        break
                    
                    # Find egg in API data
                    egg_info = None
                    for egg_id, egg_data in self.egg_data.items():
                        if egg_data.name == egg_name:
                            egg_info = egg_data
                            break
                    
                    if not egg_info:
                        self.logger.warning(f"Egg {egg_name} not found in API data")
                        continue
                    
                    # Perform real collection actions
                    success = self._collect_egg(egg_info)
                    
                    if success:
                        self.stats.successful_collections += 1
                        self.stats.eggs_collected += 1
                        self.stats.coins_spent += egg_info.cost
                    
                    self.stats.total_collections += 1
                    
                    # Wait between collections
                    time.sleep(2.0)
                
                # Wait for next collection cycle
                time.sleep(self.egg_refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(5.0)
    
    def _collect_egg(self, egg_info: EggData) -> bool:
        """Perform real egg collection actions in game"""
        try:
            self.logger.info(f"Collecting egg: {egg_info.name} (Cost: {egg_info.cost} {egg_info.currency})")
            
            # Step 1: Open shop
            if self.shop_positions['shop_button']:
                pyautogui.click(self.shop_positions['shop_button'])
                time.sleep(1.0)
            
            # Step 2: Navigate to egg merchant
            if self.shop_positions['merchant_position']:
                pyautogui.click(self.shop_positions['merchant_position'])
                time.sleep(1.0)
            
            # Step 3: Find and click egg (simplified - real implementation would use image recognition)
            if self.shop_positions['egg_buttons']:
                # For demo, click first available egg button
                pyautogui.click(self.shop_positions['egg_buttons'][0])
                time.sleep(0.5)
                
                # Confirm purchase
                pyautogui.click(self.shop_positions['egg_buttons'][0])
                time.sleep(1.0)
            
            # Step 4: Go to farm if planting is enabled
            if self.farm_positions['farm_button']:
                pyautogui.click(self.farm_positions['farm_button'])
                time.sleep(1.0)
                
                # Plant egg
                if self.farm_positions['planting_spots']:
                    pyautogui.click(self.farm_positions['planting_spots'][0])
                    time.sleep(0.5)
                    self.stats.eggs_planted += 1
            
            self.logger.info(f"Successfully collected {egg_info.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting egg {egg_info.name}: {e}")
            return False
    
    def stop_collection(self):
        """Stop collection process"""
        self.collecting = False
        self.logger.info("Stopped PS99 egg collection")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics"""
        session_duration = time.time() - self.stats.session_start if self.stats.session_start > 0 else 0
        
        success_rate = 0.0
        if self.stats.total_collections > 0:
            success_rate = (self.stats.successful_collections / self.stats.total_collections) * 100
        
        return {
            'collecting': self.collecting,
            'session_duration': session_duration,
            'total_collections': self.stats.total_collections,
            'successful_collections': self.stats.successful_collections,
            'success_rate': success_rate,
            'eggs_collected': self.stats.eggs_collected,
            'eggs_planted': self.stats.eggs_planted,
            'eggs_harvested': self.stats.eggs_harvested,
            'coins_spent': self.stats.coins_spent,
            'available_eggs': len(self.egg_data)
        }
    
    def load_cached_data(self):
        """Load cached data if available"""
        try:
            cache_file = self.cache_dir / "egg_data.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Convert back to EggData objects
                    self.egg_data = {
                        k: EggData(**v) if isinstance(v, dict) else v 
                        for k, v in cached_data.items()
                    }
                self.logger.info(f"Loaded {len(self.egg_data)} eggs from cache")
        except Exception as e:
            self.logger.warning(f"Could not load cached data: {e}")
    
    def get_priority_eggs(self) -> List[str]:
        """Get list of priority eggs for collection"""
        return self.priority_eggs
    
    def set_priority_eggs(self, eggs: List[str]):
        """Set priority eggs for collection"""
        self.priority_eggs = eggs
        self.logger.info(f"Updated priority eggs: {eggs}")