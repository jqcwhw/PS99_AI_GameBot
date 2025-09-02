"""
Command Processor for AI Game Bot
Handles natural language command processing and execution
"""

import logging
import re
from typing import Dict, List, Any, Optional
import time

class CommandProcessor:
    """Processes and executes natural language commands"""
    
    def __init__(self, vision, automation, learning, knowledge, macro):
        self.logger = logging.getLogger(__name__)
        
        # Core system references
        self.vision = vision
        self.automation = automation
        self.learning = learning
        self.knowledge = knowledge
        self.macro = macro
        
        # Command history
        self.command_history = []
        self.last_command = None
        self.last_result = None
        
        # Command patterns and mappings
        self.command_patterns = self._initialize_command_patterns()
        
        self.logger.info("Command processor initialized")
    
    def _initialize_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize command patterns for natural language processing"""
        return {
            # Game action commands
            'open_chests': {
                'patterns': [
                    r'open\s+chests?',
                    r'find\s+and\s+open\s+chests?',
                    r'chest\s+opening?',
                    r'loot\s+chests?'
                ],
                'handler': self._handle_open_chests,
                'description': 'Find and open treasure chests'
            },
            
            'hatch_eggs': {
                'patterns': [
                    r'hatch\s+eggs?',
                    r'incubate\s+eggs?',
                    r'egg\s+hatching?',
                    r'breed\s+pets?'
                ],
                'handler': self._handle_hatch_eggs,
                'description': 'Find and hatch eggs'
            },
            
            'stay_in_breakables': {
                'patterns': [
                    r'stay\s+in\s+breakables?',
                    r'go\s+to\s+breakables?\s+area',
                    r'move\s+to\s+breakables?',
                    r'breakables?\s+area'
                ],
                'handler': self._handle_stay_in_breakables,
                'description': 'Move to and stay in breakables area'
            },
            
            'farm_items': {
                'patterns': [
                    r'farm\s+(\w+)',
                    r'collect\s+(\w+)',
                    r'gather\s+(\w+)',
                    r'mine\s+(\w+)'
                ],
                'handler': self._handle_farm_items,
                'description': 'Farm specific items or resources'
            },
            
            # Vision and analysis commands
            'analyze_screen': {
                'patterns': [
                    r'analyze\s+screen',
                    r'check\s+screen',
                    r'scan\s+screen',
                    r'what\'?s\s+on\s+screen'
                ],
                'handler': self._handle_analyze_screen,
                'description': 'Analyze current screen state'
            },
            
            'find_elements': {
                'patterns': [
                    r'find\s+(\w+)',
                    r'locate\s+(\w+)',
                    r'search\s+for\s+(\w+)',
                    r'look\s+for\s+(\w+)'
                ],
                'handler': self._handle_find_elements,
                'description': 'Find specific game elements'
            },
            
            # Learning and knowledge commands
            'learn_from_file': {
                'patterns': [
                    r'learn\s+from\s+file\s+(.+)',
                    r'read\s+file\s+(.+)',
                    r'process\s+file\s+(.+)'
                ],
                'handler': self._handle_learn_from_file,
                'description': 'Learn from a file'
            },
            
            'learn_from_url': {
                'patterns': [
                    r'learn\s+from\s+url\s+(.+)',
                    r'learn\s+from\s+(https?://\S+)',
                    r'read\s+website\s+(.+)',
                    r'scrape\s+(.+)'
                ],
                'handler': self._handle_learn_from_url,
                'description': 'Learn from a URL'
            },
            
            'update_knowledge': {
                'patterns': [
                    r'update\s+knowledge',
                    r'refresh\s+knowledge',
                    r'check\s+for\s+updates',
                    r'get\s+latest\s+info'
                ],
                'handler': self._handle_update_knowledge,
                'description': 'Update knowledge from developer sources'
            },
            
            'search_knowledge': {
                'patterns': [
                    r'search\s+(.+)',
                    r'find\s+info\s+about\s+(.+)',
                    r'what\s+do\s+you\s+know\s+about\s+(.+)',
                    r'tell\s+me\s+about\s+(.+)'
                ],
                'handler': self._handle_search_knowledge,
                'description': 'Search knowledge base'
            },
            
            # Macro commands
            'record_macro': {
                'patterns': [
                    r'record\s+macro\s+(\w+)',
                    r'start\s+recording\s+(\w+)',
                    r'create\s+macro\s+(\w+)'
                ],
                'handler': self._handle_record_macro,
                'description': 'Start recording a macro'
            },
            
            'stop_recording': {
                'patterns': [
                    r'stop\s+recording',
                    r'end\s+recording',
                    r'finish\s+macro'
                ],
                'handler': self._handle_stop_recording,
                'description': 'Stop macro recording'
            },
            
            'play_macro': {
                'patterns': [
                    r'play\s+macro\s+(\w+)',
                    r'run\s+macro\s+(\w+)',
                    r'execute\s+macro\s+(\w+)'
                ],
                'handler': self._handle_play_macro,
                'description': 'Play a recorded macro'
            },
            
            'list_macros': {
                'patterns': [
                    r'list\s+macros?',
                    r'show\s+macros?',
                    r'available\s+macros?'
                ],
                'handler': self._handle_list_macros,
                'description': 'List available macros'
            },
            
            # System commands
            'status': {
                'patterns': [
                    r'status',
                    r'system\s+status',
                    r'health\s+check',
                    r'how\s+are\s+you'
                ],
                'handler': self._handle_status,
                'description': 'Show system status'
            },
            
            'config': {
                'patterns': [
                    r'config',
                    r'configuration',
                    r'settings',
                    r'show\s+config'
                ],
                'handler': self._handle_config,
                'description': 'Show configuration'
            }
        }
    
    def process_command(self, command: str) -> str:
        """
        Process a natural language command
        
        Args:
            command: Natural language command string
            
        Returns:
            Result message
        """
        try:
            command = command.strip().lower()
            
            if not command:
                return "Please enter a command"
            
            # Store command in history
            self.command_history.append({
                'command': command,
                'timestamp': time.time()
            })
            
            # Keep only last 50 commands
            if len(self.command_history) > 50:
                self.command_history = self.command_history[-50:]
            
            self.last_command = command
            
            # Try to match command patterns
            matched_command, params = self._match_command(command)
            
            if matched_command:
                # Execute the matched command
                handler = self.command_patterns[matched_command]['handler']
                result = handler(params)
                
                # Record the experience for learning
                self.learning.record_experience(
                    action=matched_command,
                    context={'command': command, 'params': params},
                    outcome='success' if 'error' not in result.lower() and 'failed' not in result.lower() else 'failure',
                    details={'result': result}
                )
                
                self.last_result = result
                return result
            else:
                # Try to handle as a general query
                result = self._handle_general_query(command)
                self.last_result = result
                return result
                
        except Exception as e:
            error_msg = f"Command processing failed: {e}"
            self.logger.error(error_msg)
            self.last_result = error_msg
            return error_msg
    
    def _match_command(self, command: str) -> tuple[Optional[str], List[str]]:
        """
        Match command against known patterns
        
        Args:
            command: Command string to match
            
        Returns:
            Tuple of (matched_command_key, extracted_parameters)
        """
        for command_key, command_info in self.command_patterns.items():
            for pattern in command_info['patterns']:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    # Extract parameters from regex groups
                    params = list(match.groups()) if match.groups() else []
                    return command_key, params
        
        return None, []
    
    # Game action handlers
    def _handle_open_chests(self, params: List[str]) -> str:
        """Handle open chests command"""
        try:
            # Start automation engine if not running
            if not self.automation.is_active():
                self.automation.start()
            
            # Capture screen and find chests
            screenshot = self.vision.capture_screen()
            if screenshot is None:
                return "Failed to capture screen"
            
            chest_positions = self.vision.find_chests(screenshot)
            
            if not chest_positions:
                return "No chests found on screen"
            
            # Use automation engine to open chests
            result = self.automation.open_chests(chest_positions)
            return result
            
        except Exception as e:
            return f"Failed to open chests: {e}"
    
    def _handle_hatch_eggs(self, params: List[str]) -> str:
        """Handle hatch eggs command"""
        try:
            if not self.automation.is_active():
                self.automation.start()
            
            screenshot = self.vision.capture_screen()
            if screenshot is None:
                return "Failed to capture screen"
            
            egg_positions = self.vision.find_eggs(screenshot)
            
            if not egg_positions:
                return "No eggs found on screen"
            
            result = self.automation.hatch_eggs(egg_positions)
            return result
            
        except Exception as e:
            return f"Failed to hatch eggs: {e}"
    
    def _handle_stay_in_breakables(self, params: List[str]) -> str:
        """Handle stay in breakables area command"""
        try:
            if not self.automation.is_active():
                self.automation.start()
            
            screenshot = self.vision.capture_screen()
            if screenshot is None:
                return "Failed to capture screen"
            
            breakables_positions = self.vision.find_breakables_area(screenshot)
            
            if not breakables_positions:
                return "No breakables area found on screen"
            
            result = self.automation.stay_in_breakables_area(breakables_positions)
            return result
            
        except Exception as e:
            return f"Failed to move to breakables area: {e}"
    
    def _handle_farm_items(self, params: List[str]) -> str:
        """Handle farm items command"""
        try:
            item_type = params[0] if params else "resources"
            
            # This would be customized based on the specific game
            # For now, we'll provide a general farming strategy
            
            if not self.automation.is_active():
                self.automation.start()
            
            # Look for relevant knowledge about farming this item
            knowledge_results = self.knowledge.get_relevant_knowledge(f"farm {item_type}", limit=3)
            
            if knowledge_results:
                strategy_info = f"Found farming strategies for {item_type}. "
                for result in knowledge_results[:1]:  # Use first strategy
                    strategy_info += result['content'][:100] + "... "
            else:
                strategy_info = f"No specific strategy found for {item_type}. Using general farming approach. "
            
            # Basic farming logic - could be enhanced with specific game knowledge
            screenshot = self.vision.capture_screen()
            if screenshot is None:
                return "Failed to capture screen for farming"
            
            # Look for farmable elements (this would be game-specific)
            analysis = self.vision.analyze_screen()
            farmable_elements = analysis.get('elements_found', {})
            
            return f"{strategy_info}Analyzing screen for {item_type} farming opportunities. Found: {farmable_elements}"
            
        except Exception as e:
            return f"Failed to farm items: {e}"
    
    # Vision and analysis handlers
    def _handle_analyze_screen(self, params: List[str]) -> str:
        """Handle analyze screen command"""
        try:
            analysis = self.vision.analyze_screen()
            
            if 'error' in analysis:
                return f"Screen analysis failed: {analysis['error']}"
            
            result = "Screen Analysis:\n"
            result += f"Screen size: {analysis.get('screen_size', 'unknown')}\n"
            
            elements = analysis.get('elements_found', {})
            for element_type, info in elements.items():
                count = info.get('count', 0)
                result += f"{element_type.title()}: {count} found\n"
            
            regions = analysis.get('regions', {})
            if 'center' in regions:
                center = regions['center']
                result += f"Screen center: ({center[0]}, {center[1]})\n"
            
            color_analysis = analysis.get('color_analysis', {})
            if 'average_brightness' in color_analysis:
                brightness = color_analysis['average_brightness']
                result += f"Average brightness: {brightness:.1f}\n"
            
            return result
            
        except Exception as e:
            return f"Screen analysis failed: {e}"
    
    def _handle_find_elements(self, params: List[str]) -> str:
        """Handle find elements command"""
        try:
            element_type = params[0] if params else "chests"
            
            screenshot = self.vision.capture_screen()
            if screenshot is None:
                return "Failed to capture screen"
            
            positions = self.vision.find_template(element_type, screenshot)
            
            if not positions:
                return f"No {element_type} found on screen"
            
            result = f"Found {len(positions)} {element_type}:\n"
            for i, (x, y, confidence) in enumerate(positions[:5]):  # Show first 5
                result += f"{i+1}. Position: ({x}, {y}), Confidence: {confidence:.2f}\n"
            
            if len(positions) > 5:
                result += f"... and {len(positions) - 5} more"
            
            return result
            
        except Exception as e:
            return f"Failed to find elements: {e}"
    
    # Learning and knowledge handlers
    def _handle_learn_from_file(self, params: List[str]) -> str:
        """Handle learn from file command"""
        try:
            file_path = params[0] if params else ""
            
            if not file_path:
                return "Please specify a file path"
            
            result = self.knowledge.learn_from_file(file_path)
            return result
            
        except Exception as e:
            return f"Failed to learn from file: {e}"
    
    def _handle_learn_from_url(self, params: List[str]) -> str:
        """Handle learn from URL command"""
        try:
            url = params[0] if params else ""
            
            if not url:
                return "Please specify a URL"
            
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            result = self.knowledge.learn_from_url(url)
            return result
            
        except Exception as e:
            return f"Failed to learn from URL: {e}"
    
    def _handle_update_knowledge(self, params: List[str]) -> str:
        """Handle update knowledge command"""
        try:
            result = self.knowledge.update_from_developer_sources()
            return result
            
        except Exception as e:
            return f"Failed to update knowledge: {e}"
    
    def _handle_search_knowledge(self, params: List[str]) -> str:
        """Handle search knowledge command"""
        try:
            query = params[0] if params else ""
            
            if not query:
                return "Please specify what to search for"
            
            result = self.knowledge.search_knowledge(query)
            return result
            
        except Exception as e:
            return f"Failed to search knowledge: {e}"
    
    # Macro handlers
    def _handle_record_macro(self, params: List[str]) -> str:
        """Handle record macro command"""
        try:
            macro_name = params[0] if params else f"macro_{int(time.time())}"
            
            result = self.macro.start_recording(macro_name, f"Recorded via command at {time.ctime()}")
            return result
            
        except Exception as e:
            return f"Failed to start macro recording: {e}"
    
    def _handle_stop_recording(self, params: List[str]) -> str:
        """Handle stop recording command"""
        try:
            result = self.macro.stop_recording()
            return result
            
        except Exception as e:
            return f"Failed to stop recording: {e}"
    
    def _handle_play_macro(self, params: List[str]) -> str:
        """Handle play macro command"""
        try:
            macro_name = params[0] if params else ""
            
            if not macro_name:
                return "Please specify a macro name"
            
            result = self.macro.play_macro(macro_name)
            return result
            
        except Exception as e:
            return f"Failed to play macro: {e}"
    
    def _handle_list_macros(self, params: List[str]) -> str:
        """Handle list macros command"""
        try:
            macros = self.macro.list_macros()
            
            if not macros:
                return "No macros available"
            
            result = f"Available macros ({len(macros)}):\n"
            for macro_name in macros:
                macro_info = self.macro.get_macro_info(macro_name)
                if macro_info:
                    result += f"• {macro_name}: {macro_info['action_count']} actions, {macro_info['duration']:.1f}s\n"
                else:
                    result += f"• {macro_name}\n"
            
            return result
            
        except Exception as e:
            return f"Failed to list macros: {e}"
    
    # System handlers
    def _handle_status(self, params: List[str]) -> str:
        """Handle status command"""
        try:
            status = "System Status:\n"
            
            # Vision system status
            status += f"Vision System: {'Active' if self.vision.is_active() else 'Inactive'}\n"
            
            # Automation system status
            status += f"Automation Engine: {'Active' if self.automation.is_active() else 'Inactive'}\n"
            if self.automation.is_active():
                queue_size = self.automation.get_queue_size()
                status += f"  - Action queue: {queue_size} pending\n"
            
            # Learning system stats
            learning_stats = self.learning.get_stats()
            status += f"Learning System:\n"
            status += f"  - Total experiences: {learning_stats['total_experiences']}\n"
            status += f"  - Success rate: {learning_stats['successful_actions']}/{learning_stats['total_experiences'] if learning_stats['total_experiences'] > 0 else 1}\n"
            status += f"  - Patterns learned: {learning_stats['patterns_learned']}\n"
            
            # Knowledge system stats
            knowledge_count = self.knowledge.get_knowledge_count()
            status += f"Knowledge Base: {knowledge_count} items\n"
            
            # Macro system status
            macro_status = self.macro.get_status()
            status += f"Macro System:\n"
            status += f"  - Total macros: {macro_status['total_macros']}\n"
            status += f"  - Recording: {'Yes' if macro_status['is_recording'] else 'No'}\n"
            status += f"  - Playing: {'Yes' if macro_status['is_playing'] else 'No'}\n"
            
            return status
            
        except Exception as e:
            return f"Failed to get status: {e}"
    
    def _handle_config(self, params: List[str]) -> str:
        """Handle config command"""
        try:
            config = "Configuration:\n"
            config += f"Vision threshold: {self.vision.match_threshold}\n"
            config += f"Automation speed: {self.automation.move_duration}s movement\n"
            config += f"Learning memory: {len(self.learning.experience_memory)} experiences\n"
            config += f"Command history: {len(self.command_history)} commands\n"
            
            return config
            
        except Exception as e:
            return f"Failed to get config: {e}"
    
    def _handle_general_query(self, command: str) -> str:
        """Handle general queries that don't match specific patterns"""
        try:
            # Try to search knowledge base for relevant information
            knowledge_results = self.knowledge.get_relevant_knowledge(command, limit=3)
            
            if knowledge_results:
                result = f"Found relevant information for '{command}':\n\n"
                for i, item in enumerate(knowledge_results, 1):
                    result += f"{i}. {item['content'][:200]}{'...' if len(item['content']) > 200 else ''}\n"
                    result += f"   (Source: {item['source']})\n\n"
                return result
            else:
                return f"I don't understand the command '{command}'. Type 'help' for available commands."
                
        except Exception as e:
            return f"Failed to process query: {e}"
    
    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history"""
        return self.command_history[-limit:]
    
    def get_available_commands(self) -> Dict[str, str]:
        """Get available commands and their descriptions"""
        return {key: info['description'] for key, info in self.command_patterns.items()}
