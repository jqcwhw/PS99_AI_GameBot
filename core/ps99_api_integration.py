#!/usr/bin/env python3
"""
Pet Simulator 99 API Integration System
Real-time data monitoring and AI learning integration
"""

import requests
import json
import time
import logging
from datetime import datetime
from pathlib import Path

class PS99APIIntegration:
    """Pet Simulator 99 Official API Integration for AI Learning"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Official PS99 API endpoints
        self.api_endpoints = {
            'pets': 'https://ps99.biggamesapi.io/api/collection/pets',
            'eggs': 'https://ps99.biggamesapi.io/api/collection/eggs', 
            'zones': 'https://ps99.biggamesapi.io/api/collection/zones',
            'worlds': 'https://ps99.biggamesapi.io/api/collection/worlds'
        }
        
        # Cache for API data
        self.cache = {
            'pets': {},
            'eggs': {},
            'zones': {},
            'worlds': {},
            'last_updated': {}
        }
        
        # Cache expiry time (5 minutes)
        self.cache_expiry = 300
        
        # Data directory
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize API data
        self.refresh_all_data()
    
    def get_ps99_data(self, data_type):
        """Get PS99 data with caching"""
        if data_type not in self.api_endpoints:
            self.logger.error(f"Unknown data type: {data_type}")
            return None
        
        # Check cache freshness
        last_update = self.cache['last_updated'].get(data_type, 0)
        if time.time() - last_update < self.cache_expiry:
            return self.cache[data_type]
        
        # Fetch fresh data
        try:
            response = requests.get(self.api_endpoints[data_type], timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', [])
                
                # Update cache
                self.cache[data_type] = data
                self.cache['last_updated'][data_type] = time.time()
                
                # Save to disk
                self.save_data_to_disk(data_type, data)
                
                self.logger.info(f"PS99 {data_type} data updated: {len(data)} items")
                return data
            else:
                self.logger.error(f"PS99 API error for {data_type}: {response.status_code}")
                return self.cache.get(data_type, [])
                
        except Exception as e:
            self.logger.error(f"PS99 API request failed for {data_type}: {e}")
            return self.cache.get(data_type, [])
    
    def save_data_to_disk(self, data_type, data):
        """Save API data to disk for persistence"""
        try:
            file_path = self.data_dir / f'ps99_{data_type}.json'
            with open(file_path, 'w') as f:
                json.dump({
                    'data': data,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'ps99.biggamesapi.io'
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save {data_type} data: {e}")
    
    def load_data_from_disk(self, data_type):
        """Load cached data from disk"""
        try:
            file_path = self.data_dir / f'ps99_{data_type}.json'
            if file_path.exists():
                with open(file_path, 'r') as f:
                    cached = json.load(f)
                return cached.get('data', [])
        except Exception as e:
            self.logger.error(f"Failed to load {data_type} data: {e}")
        return []
    
    def refresh_all_data(self):
        """Refresh all PS99 API data"""
        self.logger.info("Refreshing all PS99 API data...")
        
        for data_type in self.api_endpoints.keys():
            # Try to load from disk first
            disk_data = self.load_data_from_disk(data_type)
            if disk_data:
                self.cache[data_type] = disk_data
                self.cache['last_updated'][data_type] = time.time() - self.cache_expiry + 60  # Give it 1 minute
            
            # Then refresh from API
            self.get_ps99_data(data_type)
    
    def find_pet_by_name(self, pet_name):
        """Find pet information by name"""
        pets = self.get_ps99_data('pets')
        if not pets:
            return None
        
        for pet in pets:
            if pet.get('configName', '').lower() == pet_name.lower():
                return pet
            if pet.get('displayName', '').lower() == pet_name.lower():
                return pet
        
        # Fuzzy search
        pet_name_lower = pet_name.lower()
        for pet in pets:
            if pet_name_lower in pet.get('configName', '').lower():
                return pet
            if pet_name_lower in pet.get('displayName', '').lower():
                return pet
        
        return None
    
    def find_egg_by_name(self, egg_name):
        """Find egg information by name"""
        eggs = self.get_ps99_data('eggs')
        if not eggs:
            return None
        
        for egg in eggs:
            if egg.get('configName', '').lower() == egg_name.lower():
                return egg
            if egg.get('displayName', '').lower() == egg_name.lower():
                return egg
        
        # Fuzzy search
        egg_name_lower = egg_name.lower()
        for egg in eggs:
            if egg_name_lower in egg.get('configName', '').lower():
                return egg
            if egg_name_lower in egg.get('displayName', '').lower():
                return egg
        
        return None
    
    def find_zone_by_name(self, zone_name):
        """Find zone information by name"""
        zones = self.get_ps99_data('zones')
        if not zones:
            return None
        
        for zone in zones:
            if zone.get('configName', '').lower() == zone_name.lower():
                return zone
            if zone.get('displayName', '').lower() == zone_name.lower():
                return zone
        
        return None
    
    def find_world_by_name(self, world_name):
        """Find world information by name"""
        worlds = self.get_ps99_data('worlds')
        if not worlds:
            return None
        
        for world in worlds:
            if world.get('configName', '').lower() == world_name.lower():
                return world
            if world.get('displayName', '').lower() == world_name.lower():
                return world
        
        return None
    
    def get_pets_from_egg(self, egg_name):
        """Get list of pets that can hatch from specific egg"""
        egg_info = self.find_egg_by_name(egg_name)
        if not egg_info:
            return []
        
        # Get pets that can hatch from this egg
        possible_pets = []
        pets = self.get_ps99_data('pets')
        
        for pet in pets:
            # Check if pet can come from this egg
            pet_sources = pet.get('sources', [])
            for source in pet_sources:
                if source.get('type') == 'egg' and source.get('name', '').lower() == egg_name.lower():
                    possible_pets.append(pet)
                    break
        
        return possible_pets
    
    def validate_hatched_pet(self, hatched_pet_name, expected_egg):
        """Validate if a hatched pet can actually come from the expected egg"""
        possible_pets = self.get_pets_from_egg(expected_egg)
        
        for pet in possible_pets:
            if (pet.get('configName', '').lower() == hatched_pet_name.lower() or 
                pet.get('displayName', '').lower() == hatched_pet_name.lower()):
                return True, pet
        
        return False, None
    
    def get_zone_requirements(self, zone_name):
        """Get requirements to access a zone"""
        zone_info = self.find_zone_by_name(zone_name)
        if not zone_info:
            return None
        
        return {
            'zone': zone_info,
            'requirements': zone_info.get('requirements', {}),
            'unlock_cost': zone_info.get('unlockCost', 0),
            'world': zone_info.get('world', '')
        }
    
    def create_ai_knowledge_update(self, action_type, data):
        """Create knowledge update for AI learning system"""
        timestamp = datetime.now().isoformat()
        
        knowledge_update = {
            'timestamp': timestamp,
            'source': 'ps99_api',
            'action_type': action_type,
            'data': data,
            'api_validation': True
        }
        
        # Add relevant context based on action type
        if action_type == 'pet_hatched':
            pet_info = self.find_pet_by_name(data.get('pet_name', ''))
            egg_info = self.find_egg_by_name(data.get('egg_name', ''))
            
            knowledge_update['validation'] = {
                'pet_exists': pet_info is not None,
                'egg_exists': egg_info is not None,
                'valid_hatch': False
            }
            
            if pet_info and egg_info:
                valid, _ = self.validate_hatched_pet(data.get('pet_name', ''), data.get('egg_name', ''))
                knowledge_update['validation']['valid_hatch'] = valid
                knowledge_update['pet_rarity'] = pet_info.get('rarity', 'Unknown')
                knowledge_update['pet_value'] = pet_info.get('value', 0)
        
        elif action_type == 'zone_entered':
            zone_info = self.find_zone_by_name(data.get('zone_name', ''))
            knowledge_update['validation'] = {
                'zone_exists': zone_info is not None
            }
            if zone_info:
                knowledge_update['zone_requirements'] = self.get_zone_requirements(data.get('zone_name', ''))
        
        return knowledge_update
    
    def get_api_statistics(self):
        """Get API usage and cache statistics"""
        stats = {
            'cache_status': {},
            'data_counts': {},
            'last_updates': {}
        }
        
        for data_type in self.api_endpoints.keys():
            data = self.cache.get(data_type, [])
            last_update = self.cache['last_updated'].get(data_type, 0)
            
            stats['data_counts'][data_type] = len(data)
            stats['last_updates'][data_type] = datetime.fromtimestamp(last_update).isoformat() if last_update else 'Never'
            stats['cache_status'][data_type] = 'Fresh' if time.time() - last_update < self.cache_expiry else 'Stale'
        
        return stats

# Global instance for easy access
ps99_api = None

def get_ps99_api(logger=None):
    """Get the global PS99 API instance"""
    global ps99_api
    if ps99_api is None:
        ps99_api = PS99APIIntegration(logger)
    return ps99_api

def initialize_ps99_integration(game_bot):
    """Initialize PS99 integration with the game bot"""
    try:
        game_bot.ps99_api = get_ps99_api(game_bot.logger)
        game_bot.logger.info("PS99 API integration initialized successfully")
        
        # Add PS99 monitoring to the AI learning system
        if hasattr(game_bot, 'learning_system'):
            game_bot.learning_system.ps99_api = game_bot.ps99_api
            
        return True
    except Exception as e:
        game_bot.logger.error(f"Failed to initialize PS99 integration: {e}")
        return False