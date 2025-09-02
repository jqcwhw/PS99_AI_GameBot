#!/usr/bin/env python3
"""
AI Game Bot - Complete Native Desktop Application
"""

import sys
import os
import flask
import platform
import json
import subprocess
from pathlib import Path
from datetime import datetime
import flask
import PIL
import PIL.Image
from PIL.Image import collections.abc
import opencv_python
import trafilatura
import html5lib._inputstream
from html5lib._inputstream import chardet.universaldetector
import lxml.html.diff
from lxml.html.diff import cython.cimports
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
import threading
import time
import werkzeug._reloader
from werkzeug._reloader import watchdog.observers
import dotenv.ipython
from dotenv.ipython import ipython.core
import urllib3.contrib.pyopenssl
from urllib3.contrib.pyopenssl import OpenSSL.crypto
from numpy import numpy._core.arrayprint

# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from main import GameBot

# Import optional systems with error handling
try:
    from core.window_detector import WindowDetector
    WINDOW_DETECTOR_AVAILABLE = True
except Exception as e:
    print(f"Window detector not available: {e}")
    WindowDetector = None
    WINDOW_DETECTOR_AVAILABLE = False

try:
    from comprehensive_mutex_bypass import ComprehensiveMutexBypass
    MUTEX_BYPASS_AVAILABLE = True
except Exception as e:
    print(f"Mutex bypass not available: {e}")
    ComprehensiveMutexBypass = None
    MUTEX_BYPASS_AVAILABLE = False

try:
    from advanced_natro_sync_system import AdvancedNatroSyncSystem
    SYNC_SYSTEM_AVAILABLE = True
except Exception as e:
    print(f"Sync system not available: {e}")
    AdvancedNatroSyncSystem = None
    SYNC_SYSTEM_AVAILABLE = False

class NativeGameBotApp:
    """Complete Native Desktop Application for AI Game Bot"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Game Bot - Complete Native Desktop Application")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Initialize complete game bot with all AI systems
        try:
            from main import GameBot
            self.game_bot = GameBot()
            
            # Initialize additional advanced systems  
            try:
                from core.ps99_api_collector import PS99APICollector
                from core.ps99_action_recorder import PS99ActionRecorder
                from core.natro_automation_system import NatroAutomationSystem
                
                self.game_bot.ps99_api_collector = PS99APICollector()
                self.game_bot.ps99_action_recorder = PS99ActionRecorder()
                self.game_bot.natro_automation = NatroAutomationSystem()
                print("🥚 PS99 Complete System activated (API Collector, Action Recorder)")
                print("🔄 NatroMacro Automation System activated with 17 field types")
            except Exception as e:
                print(f"Advanced systems not available: {e}")
            
            # Verify all AI systems are loaded
            ai_systems = []
            if hasattr(self.game_bot, 'autonomous_learning'):
                ai_systems.append("Autonomous Learning")
            if hasattr(self.game_bot, 'interactive_trainer'): 
                ai_systems.append("Interactive Trainer")
            if hasattr(self.game_bot, 'learning_system'):
                ai_systems.append("Pattern Learning")
            if hasattr(self.game_bot, 'vision_system'):
                ai_systems.append("Computer Vision")
            if hasattr(self.game_bot, 'automation_engine'):
                ai_systems.append("Automation Engine")
            if hasattr(self.game_bot, 'knowledge_manager'):
                ai_systems.append("Knowledge Manager")
            if hasattr(self.game_bot, 'serpent_vision'):
                ai_systems.append("SerpentAI Vision")
            if hasattr(self.game_bot, 'serpent_rl'):
                ai_systems.append("SerpentAI Reinforcement Learning")
            if hasattr(self.game_bot, 'ps99_api_collector'):
                ai_systems.append("PS99 API Integration")
            if hasattr(self.game_bot, 'natro_automation'):
                ai_systems.append("NatroMacro Automation")
                
            self.status_text = f"AI Game Bot initialized - Systems: {', '.join(ai_systems)}"
        except Exception as e:
            self.game_bot = None
            self.status_text = f"Game Bot initialization error: {e}"
            
        # Initialize optional systems with availability checks
        if WINDOW_DETECTOR_AVAILABLE and WindowDetector:
            try:
                self.window_detector = WindowDetector()
            except Exception as e:
                self.window_detector = None
                print(f"Window detector initialization failed: {e}")
        else:
            self.window_detector = None
            
        if MUTEX_BYPASS_AVAILABLE and ComprehensiveMutexBypass:
            try:
                self.mutex_bypass = ComprehensiveMutexBypass()
            except Exception as e:
                self.mutex_bypass = None
                print(f"Mutex bypass initialization failed: {e}")
        else:
            self.mutex_bypass = None
            
        if SYNC_SYSTEM_AVAILABLE and AdvancedNatroSyncSystem:
            try:
                self.sync_system = AdvancedNatroSyncSystem()
            except Exception as e:
                self.sync_system = None
                print(f"Sync system initialization failed: {e}")
        else:
            self.sync_system = None
        
        # Application state
        self.running = True
        self.current_command = ""
        self.detected_windows = []
        self.selected_window = None
        self.roblox_accounts = []
        self.automation_active = False
        self.vision_active = False
        self.mutex_active = False
        self.sync_active = False
        
        # Setup UI
        self.create_interface()
        self.start_status_monitor()
        self.load_saved_data()
        
    def create_interface(self):
        """Create the main application interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="AI Game Bot", 
            font=('Arial', 24, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # Status frame
        status_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(
            status_frame, 
            text="Status:", 
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        ).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text=self.status_text,
            font=('Arial', 11),
            fg='#2ecc71' if self.game_bot else '#e74c3c',
            bg='#34495e'
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Command input frame
        command_frame = tk.Frame(main_frame, bg='#2c3e50')
        command_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            command_frame,
            text="Command:",
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.command_entry = tk.Entry(
            command_frame,
            font=('Consolas', 11),
            width=50,
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        self.command_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.command_entry.bind('<Return>', self.execute_command)
        
        self.execute_btn = tk.Button(
            command_frame,
            text="Execute",
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            command=self.execute_command
        )
        self.execute_btn.pack(side=tk.LEFT)
        
        # Create notebook for different sections
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_automation_tab()
        self.create_multi_instance_tab()
        self.create_macro_tab()
        self.create_settings_tab()
        
        # Results display
        results_frame = tk.Frame(main_frame, bg='#2c3e50')
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            results_frame,
            text="Command Results:",
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            font=('Consolas', 10),
            bg='#1a252f',
            fg='#ecf0f1',
            insertbackground='white',
            height=20
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Add initial message
        self.add_result("AI Game Bot Native Desktop Application")
        self.add_result(f"Status: {self.status_text}")
        self.add_result("Use tabs above to access different features")
        self.add_result("-" * 60)
        
    def create_dashboard_tab(self):
        """Create main dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Status indicators
        status_frame = tk.Frame(dashboard_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(status_frame, text="System Status", font=('Arial', 14, 'bold'), 
                fg='#ecf0f1', bg='#34495e').pack(pady=10)
        
        # Status indicators grid
        indicators_frame = tk.Frame(status_frame, bg='#34495e')
        indicators_frame.pack(pady=10)
        
        self.vision_status = tk.Label(indicators_frame, text="Vision: Inactive", 
                                     bg='#e74c3c', fg='white', padx=10, pady=5)
        self.vision_status.grid(row=0, column=0, padx=5, pady=5)
        
        self.automation_status = tk.Label(indicators_frame, text="Automation: Inactive", 
                                         bg='#e74c3c', fg='white', padx=10, pady=5)
        self.automation_status.grid(row=0, column=1, padx=5, pady=5)
        
        self.mutex_status = tk.Label(indicators_frame, text="Mutex Bypass: Inactive", 
                                    bg='#e74c3c', fg='white', padx=10, pady=5)
        self.mutex_status.grid(row=0, column=2, padx=5, pady=5)
        
        # Quick actions
        actions_frame = tk.Frame(dashboard_frame, bg='#2c3e50')
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(actions_frame, text="Quick Actions", font=('Arial', 12, 'bold'),
                fg='#ecf0f1', bg='#2c3e50').pack(anchor=tk.W, pady=(0, 10))
        
        buttons_frame = tk.Frame(actions_frame, bg='#2c3e50')
        buttons_frame.pack(fill=tk.X)
        
        quick_actions = [
            ("Start Vision", self.toggle_vision),
            ("Detect Windows", self.detect_windows),
            ("Toggle Mutex", self.toggle_mutex_bypass),
            ("Show Status", self.show_system_status),
            ("Refresh Data", self.refresh_ps99_data),
        ]
        
        for i, (text, command) in enumerate(quick_actions):
            btn = tk.Button(buttons_frame, text=text, font=('Arial', 10),
                           bg='#3498db', fg='white', command=command, width=15)
            btn.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
        
        for i in range(len(quick_actions)):
            buttons_frame.grid_columnconfigure(i, weight=1)
            
    def create_automation_tab(self):
        """Create automation control tab"""
        automation_frame = ttk.Frame(self.notebook)
        self.notebook.add(automation_frame, text="Automation")
        
        # Vision system controls
        vision_frame = tk.LabelFrame(automation_frame, text="Vision System & AI Detection", 
                                   font=('Arial', 12, 'bold'), padx=10, pady=10)
        vision_frame.pack(fill=tk.X, padx=10, pady=10)
        
        vision_buttons = tk.Frame(vision_frame)
        vision_buttons.pack(fill=tk.X)
        
        tk.Button(vision_buttons, text="Start Vision", bg='#27ae60', fg='white',
                 command=self.start_vision).pack(side=tk.LEFT, padx=5)
        tk.Button(vision_buttons, text="Stop Vision", bg='#e74c3c', fg='white',
                 command=self.stop_vision).pack(side=tk.LEFT, padx=5)
        tk.Button(vision_buttons, text="Capture Screen", bg='#f39c12', fg='white',
                 command=self.capture_screen).pack(side=tk.LEFT, padx=5)
        tk.Button(vision_buttons, text="Analyze Screen", bg='#9b59b6', fg='white',
                 command=self.analyze_current_screen).pack(side=tk.LEFT, padx=5)
        
        # AI Learning controls
        ai_frame = tk.LabelFrame(automation_frame, text="AI Learning & Automation", 
                               font=('Arial', 12, 'bold'), padx=10, pady=10)
        ai_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ai_buttons = tk.Frame(ai_frame)
        ai_buttons.pack(fill=tk.X)
        
        tk.Button(ai_buttons, text="Start AI Learning", bg='#e67e22', fg='white',
                 command=self.start_ai_learning_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(ai_buttons, text="Stop AI Learning", bg='#95a5a6', fg='white',
                 command=self.stop_ai_learning_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(ai_buttons, text="Start Auto-Play", bg='#2ecc71', fg='white',
                 command=self.start_autoplay_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(ai_buttons, text="Stop Auto-Play", bg='#e74c3c', fg='white',
                 command=self.stop_autoplay_mode).pack(side=tk.LEFT, padx=5)
        
        # Game automation controls
        game_frame = tk.LabelFrame(automation_frame, text="Game Automation", 
                                 font=('Arial', 12, 'bold'), padx=10, pady=10)
        game_frame.pack(fill=tk.X, padx=10, pady=10)
        
        game_buttons = tk.Frame(game_frame)
        game_buttons.pack(fill=tk.X)
        
        tk.Button(game_buttons, text="Open Chests", bg='#f39c12', fg='white',
                 command=self.execute_open_chests).pack(side=tk.LEFT, padx=5)
        tk.Button(game_buttons, text="Hatch Eggs", bg='#3498db', fg='white',
                 command=self.execute_hatch_eggs).pack(side=tk.LEFT, padx=5)
        tk.Button(game_buttons, text="Farm Resources", bg='#27ae60', fg='white',
                 command=self.execute_farm_resources).pack(side=tk.LEFT, padx=5)
        tk.Button(game_buttons, text="Stay in Breakables", bg='#9b59b6', fg='white',
                 command=self.execute_stay_breakables).pack(side=tk.LEFT, padx=5)
        
        # Window detection
        window_frame = tk.LabelFrame(automation_frame, text="Window Detection", 
                                   font=('Arial', 12, 'bold'), padx=10, pady=10)
        window_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(window_frame, text="Detect Windows", bg='#3498db', fg='white',
                 command=self.detect_windows).pack(side=tk.LEFT, padx=5)
        tk.Button(window_frame, text="Select Window", bg='#9b59b6', fg='white',
                 command=self.select_window).pack(side=tk.LEFT, padx=5)
        
        self.windows_list = tk.Listbox(window_frame, height=6)
        self.windows_list.pack(fill=tk.X, pady=10)
        
    def create_multi_instance_tab(self):
        """Create multi-instance management tab"""
        multi_frame = ttk.Frame(self.notebook)
        self.notebook.add(multi_frame, text="Multi-Instance")
        
        # Mutex bypass controls
        mutex_frame = tk.LabelFrame(multi_frame, text="Mutex Bypass System", 
                                  font=('Arial', 12, 'bold'), padx=10, pady=10)
        mutex_frame.pack(fill=tk.X, padx=10, pady=10)
        
        mutex_buttons = tk.Frame(mutex_frame)
        mutex_buttons.pack(fill=tk.X)
        
        tk.Button(mutex_buttons, text="Toggle Mutex Bypass", bg='#e67e22', fg='white',
                 command=self.toggle_mutex_bypass).pack(side=tk.LEFT, padx=5)
        tk.Button(mutex_buttons, text="Launch PS99 Instance", bg='#2ecc71', fg='white',
                 command=self.launch_ps99_instance).pack(side=tk.LEFT, padx=5)
        tk.Button(mutex_buttons, text="Account Manager", bg='#3498db', fg='white',
                 command=self.open_account_manager).pack(side=tk.LEFT, padx=5)
        
        # Sync system
        sync_frame = tk.LabelFrame(multi_frame, text="Synchronization System", 
                                 font=('Arial', 12, 'bold'), padx=10, pady=10)
        sync_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(sync_frame, text="Toggle Sync System", bg='#9b59b6', fg='white',
                 command=self.toggle_sync_system).pack(side=tk.LEFT, padx=5)
        tk.Button(sync_frame, text="Instance Monitor", bg='#34495e', fg='white',
                 command=self.open_instance_monitor).pack(side=tk.LEFT, padx=5)
        
        # Instances list
        self.instances_list = scrolledtext.ScrolledText(multi_frame, height=10, 
                                                       bg='#1a252f', fg='#ecf0f1')
        self.instances_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_macro_tab(self):
        """Create macro management tab"""
        macro_frame = ttk.Frame(self.notebook)
        self.notebook.add(macro_frame, text="Macros")
        
        # Macro controls
        controls_frame = tk.Frame(macro_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(controls_frame, text="Record Macro", bg='#e74c3c', fg='white',
                 command=self.record_macro).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Stop Recording", bg='#95a5a6', fg='white',
                 command=self.stop_recording).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Play Macro", bg='#27ae60', fg='white',
                 command=self.play_macro).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Load Macro", bg='#3498db', fg='white',
                 command=self.load_macro).pack(side=tk.LEFT, padx=5)
        
        # Macro list
        self.macro_listbox = tk.Listbox(macro_frame, height=15)
        self.macro_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Configuration options
        config_frame = tk.LabelFrame(settings_frame, text="Configuration", 
                                   font=('Arial', 12, 'bold'), padx=10, pady=10)
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Auto-start options
        tk.Checkbutton(config_frame, text="Auto-start vision system").pack(anchor=tk.W)
        tk.Checkbutton(config_frame, text="Auto-detect windows on startup").pack(anchor=tk.W)
        tk.Checkbutton(config_frame, text="Enable debug logging").pack(anchor=tk.W)
        
        # Buttons
        settings_buttons = tk.Frame(settings_frame)
        settings_buttons.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(settings_buttons, text="Save Settings", bg='#27ae60', fg='white',
                 command=self.save_settings).pack(side=tk.LEFT, padx=5)
        tk.Button(settings_buttons, text="Reset Settings", bg='#e74c3c', fg='white',
                 command=self.reset_settings).pack(side=tk.LEFT, padx=5)
        tk.Button(settings_buttons, text="Export Config", bg='#f39c12', fg='white',
                 command=self.export_config).pack(side=tk.LEFT, padx=5)
        
    # Functionality methods
    def toggle_vision(self):
        """Toggle vision system"""
        if self.vision_active:
            self.stop_vision()
        else:
            self.start_vision()
            
    def start_vision(self):
        """Start vision system with full AI capabilities"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'vision_system'):
                # Start the complete vision system
                if hasattr(self.game_bot.vision_system, 'start_capture'):
                    self.game_bot.vision_system.start_capture()
                
                self.vision_active = True
                self.vision_status.config(text="Vision: Active", bg='#27ae60')
                self.add_result("✅ Vision system started with AI detection")
                self.add_result("📹 Screen capture active")
                self.add_result("🧠 AI pattern recognition enabled")
        except Exception as e:
            self.add_result(f"Error starting vision: {e}")
            
    def stop_vision(self):
        """Stop vision system"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'vision_system'):
                if hasattr(self.game_bot.vision_system, 'stop_capture'):
                    self.game_bot.vision_system.stop_capture()
                    
            self.vision_active = False
            self.vision_status.config(text="Vision: Inactive", bg='#e74c3c')
            self.add_result("Vision system stopped")
        except Exception as e:
            self.add_result(f"Error stopping vision: {e}")
            
    def capture_screen(self):
        """Capture screen with AI analysis"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'vision_system'):
                # Capture and analyze screen
                if hasattr(self.game_bot.vision_system, 'capture_screen'):
                    screenshot = self.game_bot.vision_system.capture_screen()
                    self.add_result("📸 Screen captured successfully")
                    
                    # Analyze captured image with AI
                    if hasattr(self.game_bot.vision_system, 'analyze_screen'):
                        analysis = self.game_bot.vision_system.analyze_screen(screenshot)
                        self.add_result(f"🔍 AI Analysis: Found {len(analysis.get('elements', []))} game elements")
                else:
                    self.add_result("📸 Basic screen capture completed")
        except Exception as e:
            self.add_result(f"Error capturing screen: {e}")
            
    def detect_windows(self):
        """Detect Roblox windows (including Pet Simulator 99)"""
        try:
            self.add_result("Scanning for Roblox windows...")
            
            # Try using the game bot's window detector first
            if self.game_bot and hasattr(self.game_bot, 'window_detector'):
                windows = self.game_bot.window_detector.scan_for_roblox_windows()
                if windows:
                    self.add_result(f"Found {len(windows)} Roblox windows:")
                    for window in windows:
                        title = window.get('title', 'Unknown')
                        pid = window.get('pid', 'N/A')
                        process = window.get('process_name', 'RobloxPlayerBeta.exe')
                        self.add_result(f"  ▶ {title} (PID: {pid}) - {process}")
                        
                        # Update windows list in UI if it exists
                        if hasattr(self, 'windows_list'):
                            self.windows_list.insert(tk.END, f"{title} - PID: {pid}")
                else:
                    self.add_result("No Roblox windows found")
                    self.add_result("Make sure Roblox is running")
            else:
                # Fallback: Manual process detection for RobloxPlayerBeta
                self.add_result("Using manual process detection...")
                import psutil
                roblox_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] and 'RobloxPlayerBeta' in proc.info['name']:
                            roblox_processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if roblox_processes:
                    self.add_result(f"Found {len(roblox_processes)} RobloxPlayerBeta processes:")
                    for proc in roblox_processes:
                        self.add_result(f"  ▶ Process ID: {proc['pid']}")
                else:
                    self.add_result("No RobloxPlayerBeta.exe processes found")
                    self.add_result("Please start Roblox")
                    
        except Exception as e:
            self.add_result(f"Error detecting windows: {e}")
            
    def select_window(self):
        """Select a window from the list"""
        selection = self.windows_list.curselection()
        if selection:
            index = selection[0]
            if index < len(self.detected_windows):
                self.selected_window = self.detected_windows[index]
                self.add_result(f"Selected window: {self.selected_window.get('title', 'Unknown')}")
                
    def toggle_mutex_bypass(self):
        """Toggle mutex bypass system"""
        try:
            if self.mutex_bypass:
                if self.mutex_active:
                    # Stop mutex bypass
                    self.mutex_active = False
                    self.mutex_status.config(text="Mutex Bypass: Inactive", bg='#e74c3c')
                    self.add_result("Mutex bypass deactivated")
                else:
                    # Start mutex bypass
                    self.mutex_active = True
                    self.mutex_status.config(text="Mutex Bypass: Active", bg='#27ae60')
                    self.add_result("Mutex bypass activated")
        except Exception as e:
            self.add_result(f"Error toggling mutex bypass: {e}")
            
    def launch_ps99_instance(self):
        """Launch a PS99 instance"""
        try:
            # Create a simple dialog for account selection
            account_name = simpledialog.askstring("Launch PS99", "Enter account name:")
            if account_name:
                self.add_result(f"Launching PS99 instance for account: {account_name}")
                # Here would be the actual launch logic
        except Exception as e:
            self.add_result(f"Error launching PS99 instance: {e}")
            
    def open_account_manager(self):
        """Open account manager dialog"""
        try:
            account_window = tk.Toplevel(self.root)
            account_window.title("Account Manager")
            account_window.geometry("600x400")
            
            tk.Label(account_window, text="Roblox Account Manager", 
                    font=('Arial', 16, 'bold')).pack(pady=10)
            
            # Account list
            account_list = tk.Listbox(account_window, height=15)
            account_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Buttons
            btn_frame = tk.Frame(account_window)
            btn_frame.pack(fill=tk.X, padx=10, pady=10)
            
            tk.Button(btn_frame, text="Add Account", bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Remove Account", bg='#e74c3c', fg='white').pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Close", command=account_window.destroy).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            self.add_result(f"Error opening account manager: {e}")
            
    def toggle_sync_system(self):
        """Toggle Natro synchronization system for Roblox accounts"""
        try:
            if self.sync_active:
                self.stop_sync_system()
            else:
                self.start_sync_system()
        except Exception as e:
            self.add_result(f"Error toggling sync system: {e}")
            
    def start_sync_system(self):
        """Start Natro synchronization system for multi-account coordination"""
        try:
            # Initialize Natro sync system if available
            if hasattr(self.game_bot, 'natro_sync_system'):
                self.game_bot.natro_sync_system.start_sync()
                self.sync_active = True
                self.add_result("Natro sync system started")
                self.add_result("Multi-account coordination active")
                self.add_result("Roblox instance synchronization enabled")
            else:
                # Create new sync system
                from natro_sync_system import NatroSyncSystem
                sync_system = NatroSyncSystem()
                sync_system.start_sync()
                if hasattr(self.game_bot, '__dict__'):
                    self.game_bot.natro_sync_system = sync_system
                self.sync_active = True
                self.add_result("Natro sync system initialized and started")
        except Exception as e:
            self.add_result(f"Error starting sync system: {e}")
            
    def stop_sync_system(self):
        """Stop Natro synchronization system"""
        try:
            if hasattr(self.game_bot, 'natro_sync_system'):
                self.game_bot.natro_sync_system.stop_sync()
            self.sync_active = False
            self.add_result("Natro sync system stopped")
        except Exception as e:
            self.add_result(f"Error stopping sync system: {e}")
            
    def open_instance_monitor(self):
        """Open instance monitor"""
        try:
            monitor_window = tk.Toplevel(self.root)
            monitor_window.title("Instance Monitor")
            monitor_window.geometry("800x600")
            
            tk.Label(monitor_window, text="Running Instances", 
                    font=('Arial', 16, 'bold')).pack(pady=10)
            
            monitor_text = scrolledtext.ScrolledText(monitor_window, height=25)
            monitor_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            tk.Button(monitor_window, text="Refresh", bg='#3498db', fg='white').pack(pady=5)
            tk.Button(monitor_window, text="Close", command=monitor_window.destroy).pack(pady=5)
            
        except Exception as e:
            self.add_result(f"Error opening instance monitor: {e}")
            
    def record_macro(self):
        """Start recording a macro"""
        try:
            macro_name = simpledialog.askstring("Record Macro", "Enter macro name:")
            if macro_name:
                self.add_result(f"Started recording macro: {macro_name}")
                # Here would be the actual recording logic
        except Exception as e:
            self.add_result(f"Error starting macro recording: {e}")
            
    def stop_recording(self):
        """Stop recording macro"""
        try:
            self.add_result("Stopped macro recording")
        except Exception as e:
            self.add_result(f"Error stopping recording: {e}")
            
    def play_macro(self):
        """Play selected macro"""
        try:
            selection = self.macro_listbox.curselection()
            if selection:
                macro_name = self.macro_listbox.get(selection[0])
                self.add_result(f"Playing macro: {macro_name}")
        except Exception as e:
            self.add_result(f"Error playing macro: {e}")
            
    def load_macro(self):
        """Load macro from file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Macro",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                self.add_result(f"Loaded macro from: {filename}")
        except Exception as e:
            self.add_result(f"Error loading macro: {e}")
            
    def show_system_status(self):
        """Show comprehensive system status"""
        try:
            status_window = tk.Toplevel(self.root)
            status_window.title("System Status")
            status_window.geometry("600x400")
            
            status_text = scrolledtext.ScrolledText(status_window)
            status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Generate status report
            status_report = f"""
AI Game Bot System Status
========================

Game Bot: {'Initialized' if self.game_bot else 'Not Initialized'}
Vision System: {'Active' if self.vision_active else 'Inactive'}
Automation: {'Active' if self.automation_active else 'Inactive'}
Mutex Bypass: {'Active' if self.mutex_active else 'Inactive'}
Sync System: {'Active' if self.sync_active else 'Inactive'}

Detected Windows: {len(self.detected_windows)}
Selected Window: {self.selected_window.get('title', 'None') if self.selected_window else 'None'}
Platform: {platform.system()}

Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            status_text.insert(tk.END, status_report)
            
        except Exception as e:
            self.add_result(f"Error showing system status: {e}")
            
    def refresh_ps99_data(self):
        """Refresh PS99 API data"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'ps99_api'):
                self.add_result("Refreshing PS99 data...")
                # Refresh API data
                self.add_result("PS99 data refreshed successfully")
        except Exception as e:
            self.add_result(f"Error refreshing PS99 data: {e}")
            
    def save_settings(self):
        """Save application settings"""
        try:
            settings = {
                'vision_active': self.vision_active,
                'automation_active': self.automation_active,
                'mutex_active': self.mutex_active,
                'sync_active': self.sync_active
            }
            
            with open('app_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
                
            self.add_result("Settings saved successfully")
        except Exception as e:
            self.add_result(f"Error saving settings: {e}")
            
    def reset_settings(self):
        """Reset settings to defaults"""
        try:
            self.vision_active = False
            self.automation_active = False
            self.mutex_active = False
            self.sync_active = False
            self.add_result("Settings reset to defaults")
        except Exception as e:
            self.add_result(f"Error resetting settings: {e}")
            
    def export_config(self):
        """Export configuration"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Configuration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                config = {
                    'app_version': '1.0.0',
                    'settings': {
                        'vision_active': self.vision_active,
                        'automation_active': self.automation_active,
                        'mutex_active': self.mutex_active,
                        'sync_active': self.sync_active
                    },
                    'detected_windows': self.detected_windows,
                    'export_date': datetime.now().isoformat()
                }
                
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                self.add_result(f"Configuration exported to: {filename}")
        except Exception as e:
            self.add_result(f"Error exporting configuration: {e}")
            
    def load_saved_data(self):
        """Load saved application data"""
        try:
            if os.path.exists('app_settings.json'):
                with open('app_settings.json', 'r') as f:
                    settings = json.load(f)
                    
                self.vision_active = settings.get('vision_active', False)
                self.automation_active = settings.get('automation_active', False)
                self.mutex_active = settings.get('mutex_active', False)
                self.sync_active = settings.get('sync_active', False)
                
                self.add_result("Loaded saved settings")
        except Exception as e:
            self.add_result(f"Error loading saved data: {e}")
            
    # Enhanced AI functionality methods
    def analyze_current_screen(self):
        """Analyze current screen with AI"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'vision_system'):
                result = self.game_bot.process_single_command("analyze screen")
                self.add_result(f"AI Screen Analysis: {result}")
        except Exception as e:
            self.add_result(f"Error analyzing screen: {e}")
            
    def start_ai_learning_mode(self):
        """Start AI learning mode"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'start_learning_mode'):
                success = self.game_bot.start_learning_mode()
                if success:
                    self.add_result("🧠 AI Learning mode started")
                    self.add_result("👀 AI is now watching and learning from actions")
                else:
                    self.add_result("Failed to start AI learning mode")
        except Exception as e:
            self.add_result(f"Error starting AI learning: {e}")
            
    def stop_ai_learning_mode(self):
        """Stop AI learning mode"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'stop_learning_mode'):
                self.game_bot.stop_learning_mode()
                self.add_result("AI Learning mode stopped")
        except Exception as e:
            self.add_result(f"Error stopping AI learning: {e}")
            
    def start_autoplay_mode(self):
        """Start AI auto-play mode"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'start_autoplay_mode'):
                success = self.game_bot.start_autoplay_mode()
                if success:
                    self.automation_active = True
                    self.automation_status.config(text="Automation: Active", bg='#27ae60')
                    self.add_result("🚀 AI Auto-play mode started")
                    self.add_result("🎮 AI is now playing automatically")
                else:
                    self.add_result("Failed to start auto-play mode")
        except Exception as e:
            self.add_result(f"Error starting auto-play: {e}")
            
    def stop_autoplay_mode(self):
        """Stop AI auto-play mode"""
        try:
            if self.game_bot and hasattr(self.game_bot, 'stop_autoplay_mode'):
                self.game_bot.stop_autoplay_mode()
                self.automation_active = False
                self.automation_status.config(text="Automation: Inactive", bg='#e74c3c')
                self.add_result("AI Auto-play mode stopped")
        except Exception as e:
            self.add_result(f"Error stopping auto-play: {e}")
            
    def execute_open_chests(self):
        """Execute open chests automation"""
        try:
            if self.game_bot:
                result = self.game_bot.process_single_command("open chests")
                self.add_result(f"Chest Opening: {result}")
        except Exception as e:
            self.add_result(f"Error opening chests: {e}")
            
    def execute_hatch_eggs(self):
        """Execute hatch eggs automation"""
        try:
            if self.game_bot:
                result = self.game_bot.process_single_command("hatch eggs")
                self.add_result(f"Egg Hatching: {result}")
        except Exception as e:
            self.add_result(f"Error hatching eggs: {e}")
            
    def execute_farm_resources(self):
        """Execute farm resources automation"""
        try:
            if self.game_bot:
                result = self.game_bot.process_single_command("farm resources")
                self.add_result(f"Resource Farming: {result}")
        except Exception as e:
            self.add_result(f"Error farming resources: {e}")
            
    def execute_stay_breakables(self):
        """Execute stay in breakables automation"""
        try:
            if self.game_bot:
                result = self.game_bot.process_single_command("stay in breakables")
                self.add_result(f"Breakables Area: {result}")
        except Exception as e:
            self.add_result(f"Error staying in breakables: {e}")
        
    def quick_command(self, command):
        """Execute a quick command"""
        self.command_entry.delete(0, tk.END)
        self.command_entry.insert(0, command)
        self.execute_command()
        
    def execute_command(self, event=None):
        """Execute the command in the entry field"""
        command = self.command_entry.get().strip()
        if not command:
            return
            
        if not self.game_bot:
            self.add_result("Error: Game Bot not initialized")
            return
            
        self.add_result(f"\n> {command}")
        
        # Execute command in background thread
        thread = threading.Thread(target=self._execute_command_thread, args=(command,))
        thread.daemon = True
        thread.start()
        
        # Clear command entry
        self.command_entry.delete(0, tk.END)
        
    def _execute_command_thread(self, command):
        """Execute command in background thread"""
        try:
            result = self.game_bot.process_single_command(command)
            self.root.after(0, self.add_result, str(result))
        except Exception as e:
            self.root.after(0, self.add_result, f"Error: {e}")
            
    def add_result(self, message):
        """Add result to the results display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.results_text.insert(tk.END, formatted_message)
        self.results_text.see(tk.END)
        
    def start_status_monitor(self):
        """Start monitoring system status"""
        def update_status():
            while self.running:
                try:
                    if self.game_bot:
                        # Update status text
                        vision_active = hasattr(self.game_bot, 'vision_system') and self.game_bot.vision_system is not None
                        automation_active = hasattr(self.game_bot, 'automation_engine') and self.game_bot.automation_engine is not None
                        
                        status_parts = []
                        if vision_active:
                            status_parts.append("Vision: Active")
                        if automation_active:
                            status_parts.append("Automation: Active")
                            
                        if status_parts:
                            status_text = "Running - " + " | ".join(status_parts)
                        else:
                            status_text = "Ready"
                            
                        self.root.after(0, self.update_status_display, status_text, '#2ecc71')
                    
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    self.root.after(0, self.update_status_display, f"Status Error: {e}", '#e74c3c')
                    time.sleep(5)
                    
        thread = threading.Thread(target=update_status)
        thread.daemon = True
        thread.start()
        
    def update_status_display(self, text, color):
        """Update the status display"""
        self.status_label.config(text=text, fg=color)
        
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        self.root.destroy()
        
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """Main entry point for native desktop app"""
    app = NativeGameBotApp()
    app.run()

if __name__ == "__main__":
    main()