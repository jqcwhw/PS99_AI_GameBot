#!/usr/bin/env python3
"""
AI Game Bot - Cross-Platform Desktop Application
Standalone desktop app that works on Windows, macOS, and Linux
Uses native window detection and user's internet connection directly
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import json
import time
import subprocess
import platform
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from main import GameBot

class CrossPlatformGameBotApp:
    """Cross-platform Desktop Application for AI Game Bot"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Game Bot - Desktop Application")
        self.root.geometry("1400x900")
        
        # Initialize game bot
        try:
            self.game_bot = GameBot()
            self.add_status("AI Game Bot initialized successfully")
        except Exception as e:
            self.game_bot = None
            self.add_status(f"Game Bot initialization error: {e}")
        
        # Application state
        self.current_task = None
        self.recording_task = False
        self.detected_windows = []
        self.selected_window = None
        self.system_platform = platform.system()
        
        # Setup UI
        self.create_interface()
        self.start_background_monitoring()
        
    def create_interface(self):
        """Create the main application interface"""
        # Style configuration
        style = ttk.Style()
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title bar
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="AI Game Bot", font=('Arial', 20, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        platform_label = ttk.Label(title_frame, text=f"Platform: {self.system_platform}", font=('Arial', 10))
        platform_label.pack(side=tk.RIGHT)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create different tabs
        self.create_dashboard_tab()
        self.create_process_detection_tab()
        self.create_task_recorder_tab()
        self.create_automation_tab()
        self.create_export_tab()
        
    def create_dashboard_tab(self):
        """Create main dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Left panel - System Information
        left_panel = ttk.LabelFrame(dashboard_frame, text="System Information")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=5, ipadx=10)
        
        # System info
        info_labels = [
            f"Platform: {self.system_platform}",
            f"Python: {sys.version.split()[0]}",
            f"Game Bot: {'✅ Ready' if self.game_bot else '❌ Error'}",
            f"Process Detection: {'✅ Available' if self.can_detect_processes() else '❌ Limited'}"
        ]
        
        for info in info_labels:
            ttk.Label(left_panel, text=info).pack(anchor=tk.W, pady=2, padx=5)
        
        # Quick actions
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="Quick Actions", font=('Arial', 12, 'bold')).pack(pady=5)
        
        actions = [
            ("Scan Processes", self.scan_processes),
            ("Start Recording", self.start_recording),
            ("Stop Recording", self.stop_recording),
            ("Take Screenshot", self.take_screenshot),
            ("Test Automation", self.test_automation)
        ]
        
        for text, command in actions:
            btn = ttk.Button(left_panel, text=text, command=command, width=20)
            btn.pack(pady=2, padx=5, fill=tk.X)
        
        # Right panel - Status Monitor
        right_panel = ttk.LabelFrame(dashboard_frame, text="System Status")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        
        # Status display
        self.status_text = scrolledtext.ScrolledText(right_panel, height=15, width=60)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Clear button
        clear_btn = ttk.Button(right_panel, text="Clear Log", command=self.clear_status)
        clear_btn.pack(pady=5)
        
        # Initialize status
        self.add_status("AI Game Bot Desktop Application Started")
        self.add_status(f"Running on {self.system_platform}")
        if self.game_bot:
            self.add_status("All systems operational")
        else:
            self.add_status("Warning: Some features may be limited")
        
    def create_process_detection_tab(self):
        """Create process detection tab"""
        process_frame = ttk.Frame(self.notebook)
        self.notebook.add(process_frame, text="Process Detection")
        
        # Controls
        control_frame = ttk.Frame(process_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Scan for Game Processes", 
                  command=self.scan_processes).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh List", 
                  command=self.refresh_processes).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Focus Selected", 
                  command=self.focus_selected_process).pack(side=tk.LEFT, padx=5)
        
        # Search filter
        ttk.Label(control_frame, text="Filter:").pack(side=tk.LEFT, padx=(20, 5))
        self.process_filter = tk.StringVar()
        filter_entry = ttk.Entry(control_frame, textvariable=self.process_filter, width=20)
        filter_entry.pack(side=tk.LEFT, padx=5)
        filter_entry.bind('<KeyRelease>', self.filter_processes)
        
        # Process list
        list_frame = ttk.LabelFrame(process_frame, text="Detected Processes")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for processes
        columns = ('Name', 'PID', 'Status', 'Memory', 'Platform')
        self.process_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.process_tree.heading(col, text=col)
            self.process_tree.column(col, width=120)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.process_tree.yview)
        self.process_tree.configure(yscrollcommand=scrollbar.set)
        
        self.process_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.process_tree.bind('<<TreeviewSelect>>', self.on_process_select)
        
        # Process details
        details_frame = ttk.LabelFrame(process_frame, text="Process Details")
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.process_details = tk.Text(details_frame, height=4)
        self.process_details.pack(fill=tk.X, padx=5, pady=5)
        
    def create_task_recorder_tab(self):
        """Create task recording tab"""
        recorder_frame = ttk.Frame(self.notebook)
        self.notebook.add(recorder_frame, text="Task Recorder")
        
        # Recording controls
        control_frame = ttk.Frame(recorder_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Task Name:").pack(side=tk.LEFT)
        self.task_name_var = tk.StringVar(value="New Task")
        task_entry = ttk.Entry(control_frame, textvariable=self.task_name_var, width=25)
        task_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Start Recording", 
                  command=self.start_recording).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Recording", 
                  command=self.stop_recording).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Pause", 
                  command=self.pause_recording).pack(side=tk.LEFT, padx=5)
        
        # Recording status
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(side=tk.RIGHT)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.recording_status = ttk.Label(status_frame, text="Not Recording", foreground='red')
        self.recording_status.pack(side=tk.LEFT, padx=5)
        
        # Recording timer
        self.recording_timer = ttk.Label(status_frame, text="00:00")
        self.recording_timer.pack(side=tk.LEFT, padx=5)
        
        # Recorded actions list
        actions_frame = ttk.LabelFrame(recorder_frame, text="Recorded Actions")
        actions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Actions treeview
        action_columns = ('Action', 'Details', 'Timestamp', 'Duration')
        self.actions_tree = ttk.Treeview(actions_frame, columns=action_columns, show='headings')
        
        for col in action_columns:
            self.actions_tree.heading(col, text=col)
            if col == 'Details':
                self.actions_tree.column(col, width=300)
            else:
                self.actions_tree.column(col, width=120)
        
        # Scrollbar for actions
        action_scrollbar = ttk.Scrollbar(actions_frame, orient=tk.VERTICAL, command=self.actions_tree.yview)
        self.actions_tree.configure(yscrollcommand=action_scrollbar.set)
        
        self.actions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        action_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action controls
        action_control_frame = ttk.Frame(recorder_frame)
        action_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_control_frame, text="Clear All", 
                  command=self.clear_actions).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_control_frame, text="Delete Selected", 
                  command=self.delete_selected_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_control_frame, text="Preview", 
                  command=self.preview_task).pack(side=tk.LEFT, padx=5)
        
    def create_automation_tab(self):
        """Create automation control tab"""
        auto_frame = ttk.Frame(self.notebook)
        self.notebook.add(auto_frame, text="Automation")
        
        # Automation mode controls
        mode_frame = ttk.LabelFrame(auto_frame, text="Automation Modes")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        mode_control_frame = ttk.Frame(mode_frame)
        mode_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.autoplay_var = tk.BooleanVar()
        ttk.Checkbutton(mode_control_frame, text="Auto-Play Mode", 
                       variable=self.autoplay_var, command=self.toggle_autoplay).pack(side=tk.LEFT)
        
        self.learning_var = tk.BooleanVar()
        ttk.Checkbutton(mode_control_frame, text="AI Learning", 
                       variable=self.learning_var, command=self.toggle_learning).pack(side=tk.LEFT, padx=15)
        
        self.safe_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_control_frame, text="Safe Mode", 
                       variable=self.safe_mode_var).pack(side=tk.LEFT, padx=15)
        
        # Command execution
        command_frame = ttk.LabelFrame(auto_frame, text="Direct Commands")
        command_frame.pack(fill=tk.X, padx=5, pady=5)
        
        cmd_input_frame = ttk.Frame(command_frame)
        cmd_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(cmd_input_frame, text="Command:").pack(side=tk.LEFT)
        self.command_var = tk.StringVar()
        command_entry = ttk.Entry(cmd_input_frame, textvariable=self.command_var, width=50)
        command_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        command_entry.bind('<Return>', self.execute_command)
        
        ttk.Button(cmd_input_frame, text="Execute", 
                  command=self.execute_command).pack(side=tk.LEFT, padx=5)
        
        # Command results
        results_frame = ttk.LabelFrame(auto_frame, text="Command Results & Automation Log")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Result controls
        result_controls = ttk.Frame(results_frame)
        result_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(result_controls, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT)
        ttk.Button(result_controls, text="Save Log", 
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
        
    def create_export_tab(self):
        """Create export functionality tab"""
        export_frame = ttk.Frame(self.notebook)
        self.notebook.add(export_frame, text="Export & Build")
        
        # Export format selection
        format_frame = ttk.LabelFrame(export_frame, text="Export Format")
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.export_format = tk.StringVar(value="AutoHotkey v2")
        formats = ["AutoHotkey v2", "AutoHotkey v1", "Python", "JavaScript", "JSON", "Batch Script"]
        
        format_selection_frame = ttk.Frame(format_frame)
        format_selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(format_selection_frame, text="Format:").pack(side=tk.LEFT)
        format_combo = ttk.Combobox(format_selection_frame, textvariable=self.export_format, 
                                   values=formats, state="readonly")
        format_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(format_selection_frame, text="Export Current Task", 
                  command=self.export_current_task).pack(side=tk.LEFT, padx=20)
        ttk.Button(format_selection_frame, text="Export All Tasks", 
                  command=self.export_all_tasks).pack(side=tk.LEFT, padx=5)
        
        # Task list for export
        task_frame = ttk.LabelFrame(export_frame, text="Available Tasks")
        task_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Task listbox
        task_list_frame = ttk.Frame(task_frame)
        task_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.task_listbox = tk.Listbox(task_list_frame, selectmode=tk.MULTIPLE)
        task_scrollbar = ttk.Scrollbar(task_list_frame, orient=tk.VERTICAL, command=self.task_listbox.yview)
        self.task_listbox.configure(yscrollcommand=task_scrollbar.set)
        
        self.task_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        task_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Task controls
        task_controls = ttk.Frame(task_frame)
        task_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(task_controls, text="Refresh Tasks", 
                  command=self.refresh_task_list).pack(side=tk.LEFT)
        ttk.Button(task_controls, text="Delete Selected", 
                  command=self.delete_selected_tasks).pack(side=tk.LEFT, padx=5)
        ttk.Button(task_controls, text="Duplicate Task", 
                  command=self.duplicate_task).pack(side=tk.LEFT, padx=5)
        
        # Build executable section
        build_frame = ttk.LabelFrame(export_frame, text="Build Standalone Application")
        build_frame.pack(fill=tk.X, padx=5, pady=5)
        
        build_controls = ttk.Frame(build_frame)
        build_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(build_controls, text="Build Executable", 
                  command=self.build_executable).pack(side=tk.LEFT)
        ttk.Button(build_controls, text="Create Installer", 
                  command=self.create_installer).pack(side=tk.LEFT, padx=5)
        
        self.build_status = ttk.Label(build_controls, text="Ready to build")
        self.build_status.pack(side=tk.LEFT, padx=20)
        
    # Core functionality methods
    def can_detect_processes(self):
        """Check if process detection is available on this platform"""
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def scan_processes(self):
        """Scan for game processes cross-platform"""
        try:
            self.add_status("Scanning for game processes...")
            
            if not self.can_detect_processes():
                self.add_status("Process detection not available - psutil required")
                return
            
            import psutil
            
            # Clear existing items
            for item in self.process_tree.get_children():
                self.process_tree.delete(item)
            
            game_processes = []
            target_names = ['roblox', 'minecraft', 'steam', 'game', 'player']
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'status']):
                try:
                    proc_name = proc.info['name'].lower()
                    if any(target in proc_name for target in target_names):
                        memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                        
                        self.process_tree.insert('', 'end', values=(
                            proc.info['name'],
                            proc.info['pid'],
                            proc.info['status'],
                            f"{memory_mb:.1f} MB",
                            self.system_platform
                        ))
                        game_processes.append(proc.info)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.add_status(f"Found {len(game_processes)} game-related processes")
            
        except Exception as e:
            self.add_status(f"Error scanning processes: {e}")
    
    def start_recording(self):
        """Start recording user actions"""
        if not self.recording_task:
            task_name = self.task_name_var.get()
            if not task_name.strip():
                messagebox.showwarning("Warning", "Please enter a task name")
                return
            
            self.recording_task = True
            self.recording_start_time = time.time()
            self.recording_status.config(text="Recording...", foreground='green')
            self.add_status(f"Started recording task: {task_name}")
            
            # Clear actions tree
            for item in self.actions_tree.get_children():
                self.actions_tree.delete(item)
            
            # Start recording timer
            self.update_recording_timer()
    
    def stop_recording(self):
        """Stop recording user actions"""
        if self.recording_task:
            self.recording_task = False
            self.recording_status.config(text="Not Recording", foreground='red')
            task_name = self.task_name_var.get()
            self.add_status(f"Stopped recording task: {task_name}")
            
            # Save the recorded task
            self.save_recorded_task()
    
    def pause_recording(self):
        """Pause/unpause recording"""
        if self.recording_task:
            self.recording_status.config(text="Paused", foreground='orange')
            self.add_status("Recording paused")
    
    def update_recording_timer(self):
        """Update the recording timer display"""
        if self.recording_task:
            elapsed = time.time() - self.recording_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.recording_timer.config(text=f"{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_recording_timer)
    
    def add_status(self, message):
        """Add status message with timestamp"""
        if hasattr(self, 'status_text'):
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.status_text.see(tk.END)
    
    def clear_status(self):
        """Clear status log"""
        if hasattr(self, 'status_text'):
            self.status_text.delete(1.0, tk.END)
    
    def export_current_task(self):
        """Export the currently recorded task"""
        task_name = self.task_name_var.get()
        if not task_name.strip():
            messagebox.showwarning("Warning", "No task to export")
            return
        
        format_name = self.export_format.get()
        
        # Generate script content based on format
        if format_name == "AutoHotkey v2":
            content = self.generate_ahk_v2_script(task_name)
            extension = ".ahk"
        elif format_name == "Python":
            content = self.generate_python_script(task_name)
            extension = ".py"
        elif format_name == "JavaScript":
            content = self.generate_js_script(task_name)
            extension = ".js"
        else:
            content = self.generate_json_export(task_name)
            extension = ".json"
        
        # Save file dialog
        filename = filedialog.asksaveasfilename(
            defaultextension=extension,
            filetypes=[(format_name, f"*{extension}")],
            initialname=f"{task_name.replace(' ', '_')}{extension}"
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(content)
            self.add_status(f"Task exported to: {filename}")
            messagebox.showinfo("Success", f"Task exported successfully to:\n{filename}")
    
    def generate_ahk_v2_script(self, task_name):
        """Generate AutoHotkey v2 script"""
        return f"""; AutoHotkey v2 Script - {task_name}
; Generated by AI Game Bot Desktop Application
; Platform: {self.system_platform}

#Requires AutoHotkey v2.0
#SingleInstance Force

; Hotkeys
F1:: {{
    MsgBox("Starting task: {task_name}")
    ExecuteTask()
}}

F2:: ExitApp()

; Main execution function
ExecuteTask() {{
    ; Add your recorded actions here
    MsgBox("Task execution completed: {task_name}")
}}

; Helper functions
WaitForWindow(windowTitle) {{
    WinWait(windowTitle, , 10)
    if !WinExist(windowTitle)
        MsgBox("Window not found: " . windowTitle)
}}
"""
    
    def generate_python_script(self, task_name):
        """Generate Python automation script"""
        return f"""#!/usr/bin/env python3
# Python Automation Script - {task_name}
# Generated by AI Game Bot Desktop Application
# Platform: {self.system_platform}

import time
import sys

try:
    import pyautogui
    import psutil
except ImportError as e:
    print(f"Missing dependency: {{e}}")
    print("Install with: pip install pyautogui psutil")
    sys.exit(1)

def execute_task():
    \"\"\"Execute the recorded task\"\"\"
    print(f"Starting task: {task_name}")
    print(f"Platform: {self.system_platform}")
    
    # Add your recorded actions here
    
    print(f"Task completed: {task_name}")

def find_game_window():
    \"\"\"Find game window by process name\"\"\"
    for proc in psutil.process_iter(['pid', 'name']):
        if 'roblox' in proc.info['name'].lower():
            return proc.info['pid']
    return None

if __name__ == "__main__":
    try:
        execute_task()
    except KeyboardInterrupt:
        print("Task interrupted by user")
    except Exception as e:
        print(f"Error executing task: {{e}}")
"""
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the application
        try:
            self.add_status("Desktop application ready")
            self.root.mainloop()
        except Exception as e:
            self.add_status(f"Application error: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit AI Game Bot?"):
            self.root.quit()
            os._exit(0)
    
    # Additional stub methods for completeness
    def refresh_processes(self): self.scan_processes()
    def focus_selected_process(self): self.add_status("Focus process functionality")
    def on_process_select(self, event): pass
    def filter_processes(self, event): pass
    def clear_actions(self): pass
    def delete_selected_action(self): pass
    def preview_task(self): pass
    def save_recorded_task(self): pass
    def toggle_autoplay(self): self.add_status(f"Auto-play: {self.autoplay_var.get()}")
    def toggle_learning(self): self.add_status(f"AI Learning: {self.learning_var.get()}")
    def execute_command(self, event=None): pass
    def clear_results(self): pass
    def save_results(self): pass
    def export_all_tasks(self): pass
    def refresh_task_list(self): pass
    def delete_selected_tasks(self): pass
    def duplicate_task(self): pass
    def build_executable(self): self.add_status("Build functionality available")
    def create_installer(self): self.add_status("Installer creation available")
    def take_screenshot(self): self.add_status("Screenshot functionality")
    def test_automation(self): self.add_status("Testing automation systems")
    def start_background_monitoring(self): pass
    def generate_js_script(self, task_name): return f"// JavaScript script for {task_name}"
    def generate_json_export(self, task_name): return f'{{"task_name": "{task_name}"}}'

def main():
    """Main entry point"""
    try:
        print("Starting AI Game Bot Cross-Platform Desktop Application...")
        app = CrossPlatformGameBotApp()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())