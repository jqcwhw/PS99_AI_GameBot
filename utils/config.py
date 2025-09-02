"""
Configuration Management for AI Game Bot
Handles application configuration and settings
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    """Configuration manager for the AI Game Bot"""
    
    def __init__(self, config_file: str = "config.json"):
        self.logger = logging.getLogger(__name__)
        self.config_file = Path(config_file)
        
        # Default configuration
        self.defaults = {
            # Vision system settings
            "vision": {
                "match_threshold": 0.8,
                "capture_interval": 1.0,
                "screen_region": None,
                "auto_save_templates": True,
                "template_directory": "data/templates"
            },
            
            # Automation settings
            "automation": {
                "move_duration": 0.3,
                "click_duration": 0.1,
                "human_like_movement": True,
                "safety_checks": True,
                "max_queue_size": 100,
                "action_delay": 0.1
            },
            
            # Learning system settings
            "learning": {
                "max_memory_size": 1000,
                "similarity_threshold": 0.8,
                "min_pattern_occurrences": 3,
                "auto_adapt": True,
                "learning_rate": 0.1
            },
            
            # Knowledge management settings
            "knowledge": {
                "auto_update_interval": 3600,  # 1 hour
                "max_file_size": 10485760,  # 10MB
                "supported_formats": [".txt", ".md", ".json", ".html"],
                "developer_sources": [
                    "https://example-game-blog.com/rss",
                    "https://example-game.com/api/updates"
                ]
            },
            
            # Macro system settings
            "macros": {
                "max_recording_duration": 300,  # 5 minutes
                "recording_interval": 0.1,
                "auto_save": True,
                "backup_macros": True,
                "playback_speed": 1.0
            },
            
            # Web interface settings
            "web": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False,
                "auto_refresh_interval": 2000,  # 2 seconds
                "max_screenshot_width": 800
            },
            
            # Logging settings
            "logging": {
                "level": "INFO",
                "file_logging": True,
                "console_logging": True,
                "max_log_size": 10485760,  # 10MB
                "backup_count": 5
            },
            
            # Security settings
            "security": {
                "safe_regions": [],
                "forbidden_regions": [],
                "require_confirmation": True,
                "max_automation_time": 1800  # 30 minutes
            },
            
            # Game-specific settings
            "game": {
                "name": "Generic Game",
                "window_title": "",
                "process_name": "",
                "key_bindings": {
                    "pause": "esc",
                    "inventory": "i",
                    "map": "m",
                    "chat": "enter"
                },
                "ui_elements": {
                    "health_bar_region": None,
                    "minimap_region": None,
                    "inventory_region": None
                }
            }
        }
        
        # Load configuration
        self.config = self._load_config()
        
        # Environment variable overrides
        self._apply_env_overrides()
        
        self.logger.info("Configuration loaded")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with defaults
                config = self.defaults.copy()
                self._deep_update(config, file_config)
                
                self.logger.info(f"Configuration loaded from {self.config_file}")
                return config
            else:
                self.logger.info("No config file found, using defaults")
                return self.defaults.copy()
                
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self.defaults.copy()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Common environment variable mappings
        env_mappings = {
            'GAMEBOT_LOG_LEVEL': ('logging', 'level'),
            'GAMEBOT_WEB_PORT': ('web', 'port'),
            'GAMEBOT_WEB_HOST': ('web', 'host'),
            'GAMEBOT_DEBUG': ('web', 'debug'),
            'GAMEBOT_VISION_THRESHOLD': ('vision', 'match_threshold'),
            'GAMEBOT_AUTOMATION_SPEED': ('automation', 'move_duration'),
            'GAMEBOT_LEARNING_MEMORY': ('learning', 'max_memory_size'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert value to appropriate type
                    if key in ['port', 'max_memory_size']:
                        value = int(value)
                    elif key in ['match_threshold', 'move_duration']:
                        value = float(value)
                    elif key == 'debug':
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    
                    self.config[section][key] = value
                    self.logger.info(f"Applied environment override: {env_var} = {value}")
                    
                except ValueError as e:
                    self.logger.error(f"Invalid environment variable value {env_var}={value}: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            self.config_file.parent.mkdir(exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            section: Configuration section
            key: Configuration key (optional)
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        try:
            if key is None:
                return self.config.get(section, default)
            else:
                return self.config.get(section, {}).get(key, default)
        except Exception:
            return default
    
    def set(self, section: str, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        self.logger.debug(f"Configuration updated: {section}.{key} = {value}")
    
    def get_vision_config(self) -> Dict[str, Any]:
        """Get vision system configuration"""
        return self.config.get('vision', {})
    
    def get_automation_config(self) -> Dict[str, Any]:
        """Get automation system configuration"""
        return self.config.get('automation', {})
    
    def get_learning_config(self) -> Dict[str, Any]:
        """Get learning system configuration"""
        return self.config.get('learning', {})
    
    def get_knowledge_config(self) -> Dict[str, Any]:
        """Get knowledge management configuration"""
        return self.config.get('knowledge', {})
    
    def get_macro_config(self) -> Dict[str, Any]:
        """Get macro system configuration"""
        return self.config.get('macros', {})
    
    def get_web_config(self) -> Dict[str, Any]:
        """Get web interface configuration"""
        return self.config.get('web', {})
    
    def get_game_config(self) -> Dict[str, Any]:
        """Get game-specific configuration"""
        return self.config.get('game', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.config.get('security', {})
    
    def validate_config(self) -> Dict[str, list]:
        """
        Validate configuration values
        
        Returns:
            Dictionary of validation errors by section
        """
        errors = {}
        
        # Validate vision config
        vision_errors = []
        vision_config = self.get_vision_config()
        
        threshold = vision_config.get('match_threshold', 0.8)
        if not 0.0 <= threshold <= 1.0:
            vision_errors.append("match_threshold must be between 0.0 and 1.0")
        
        interval = vision_config.get('capture_interval', 1.0)
        if interval <= 0:
            vision_errors.append("capture_interval must be positive")
        
        if vision_errors:
            errors['vision'] = vision_errors
        
        # Validate automation config
        automation_errors = []
        automation_config = self.get_automation_config()
        
        move_duration = automation_config.get('move_duration', 0.3)
        if move_duration <= 0:
            automation_errors.append("move_duration must be positive")
        
        if automation_errors:
            errors['automation'] = automation_errors
        
        # Validate web config
        web_errors = []
        web_config = self.get_web_config()
        
        port = web_config.get('port', 5000)
        if not 1 <= port <= 65535:
            web_errors.append("port must be between 1 and 65535")
        
        if web_errors:
            errors['web'] = web_errors
        
        return errors
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.defaults.copy()
        self.logger.info("Configuration reset to defaults")
    
    def backup_config(self, backup_file: Optional[str] = None):
        """Create a backup of current configuration"""
        try:
            if backup_file is None:
                timestamp = int(time.time())
                backup_file = f"config_backup_{timestamp}.json"
            
            backup_path = Path(backup_file)
            backup_path.parent.mkdir(exist_ok=True)
            
            with open(backup_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration backed up to {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to backup config: {e}")
            return None
    
    def restore_config(self, backup_file: str):
        """Restore configuration from backup"""
        try:
            backup_path = Path(backup_file)
            
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            with open(backup_path, 'r') as f:
                backup_config = json.load(f)
            
            self.config = backup_config
            self.logger.info(f"Configuration restored from {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to restore config: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.config, indent=2)
