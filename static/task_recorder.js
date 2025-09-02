// Task Recorder JavaScript

class TaskRecorder {
    constructor() {
        this.isRecording = false;
        this.recordingStartTime = null;
        this.recordingTimer = null;
        this.selectedTask = null;
        
        this.initializeEventListeners();
        this.loadRecordedTasks();
    }
    
    initializeEventListeners() {
        // Recording controls
        document.getElementById('startTaskRecording').addEventListener('click', () => this.startRecording());
        document.getElementById('stopTaskRecording').addEventListener('click', () => this.stopRecording());
        
        // Export controls
        document.getElementById('taskSelect').addEventListener('change', (e) => this.selectTask(e.target.value));
        document.querySelectorAll('.export-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.previewExport(e.target.closest('.export-btn').dataset.format));
        });
        document.getElementById('downloadFile').addEventListener('click', () => this.downloadFile());
    }
    
    async startRecording() {
        const taskName = document.getElementById('taskName').value.trim();
        const description = document.getElementById('taskDescription').value.trim();
        
        if (!taskName) {
            this.showAlert('Please enter a task name', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/macros/record-task', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    task_name: taskName,
                    description: description
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isRecording = true;
                this.recordingStartTime = Date.now();
                this.updateRecordingUI(true);
                this.startRecordingTimer();
                this.showAlert(result.message, 'success');
            } else {
                this.showAlert(`Failed to start recording: ${result.error}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Error starting recording: ${error.message}`, 'danger');
        }
    }
    
    async stopRecording() {
        try {
            const response = await fetch('/api/macros/stop-task-recording', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isRecording = false;
                this.updateRecordingUI(false);
                this.stopRecordingTimer();
                this.showAlert(result.message, 'success');
                this.loadRecordedTasks();
                this.clearForm();
            } else {
                this.showAlert(`Failed to stop recording: ${result.error}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Error stopping recording: ${error.message}`, 'danger');
        }
    }
    
    updateRecordingUI(recording) {
        const startBtn = document.getElementById('startTaskRecording');
        const stopBtn = document.getElementById('stopTaskRecording');
        const recordingInfo = document.getElementById('recordingInfo');
        const statusDiv = document.getElementById('recordingStatus');
        
        if (recording) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            recordingInfo.style.display = 'block';
            statusDiv.style.display = 'block';
            statusDiv.className = 'alert alert-success';
            document.getElementById('statusMessage').textContent = 'Recording in progress...';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            recordingInfo.style.display = 'none';
            statusDiv.style.display = 'none';
        }
    }
    
    startRecordingTimer() {
        this.recordingTimer = setInterval(() => {
            if (this.recordingStartTime) {
                const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                
                document.getElementById('recordingDuration').textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                
                // Update action count (placeholder - would come from actual recording)
                const currentActions = Math.floor(elapsed / 2); // Simulate action counting
                document.getElementById('actionCount').textContent = currentActions;
                
                // Update progress bar
                const maxDuration = 300; // 5 minutes max
                const progressPercent = Math.min((elapsed / maxDuration) * 100, 100);
                document.getElementById('recordingProgress').style.width = `${progressPercent}%`;
            }
        }, 1000);
    }
    
    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }
    
    async loadRecordedTasks() {
        try {
            // Fetch actual tasks from the API
            const response = await fetch('/api/macros');
            const result = await response.json();
            
            if (result.success) {
                // Filter for user_recorded tasks
                const userTasks = result.macros.filter(macro => macro.type === 'user_recorded');
                this.updateTasksTable(userTasks);
                this.updateTaskSelect(userTasks);
            } else {
                // Fall back to simulated data if API fails
                const tasks = this.getSimulatedTasks();
                this.updateTasksTable(tasks);
                this.updateTaskSelect(tasks);
            }
        } catch (error) {
            console.error('Failed to load tasks:', error);
            // Fall back to simulated data
            const tasks = this.getSimulatedTasks();
            this.updateTasksTable(tasks);
            this.updateTaskSelect(tasks);
        }
    }
    
    getSimulatedTasks() {
        // Simulate recorded tasks - in real implementation, fetch from API
        return [
            {
                name: 'Open Chests Sequence',
                description: 'Opens multiple chests in Pet Simulator 99',
                actions: 15,
                duration: 45.2,
                created: new Date().toISOString()
            },
            {
                name: 'Egg Hatching Loop',
                description: 'Automated egg hatching routine',
                actions: 8,
                duration: 23.1,
                created: new Date(Date.now() - 3600000).toISOString()
            }
        ];
    }
    
    updateTasksTable(tasks) {
        const tbody = document.getElementById('tasksTableBody');
        tbody.innerHTML = '';
        
        tasks.forEach(task => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${task.name}</strong></td>
                <td>${task.description || 'No description'}</td>
                <td><span class="badge bg-info">${task.action_count || task.actions || 0}</span></td>
                <td>${(task.duration || 0).toFixed(1)}s</td>
                <td>${new Date(task.created_at || task.created || Date.now()).toLocaleString()}</td>
                <td>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary" onclick="taskRecorder.exportTask('${task.name}', 'json')">
                            <i class="fas fa-file-code"></i>
                        </button>
                        <button class="btn btn-outline-success" onclick="taskRecorder.exportTask('${task.name}', 'python')">
                            <i class="fab fa-python"></i>
                        </button>
                        <button class="btn btn-outline-warning" onclick="taskRecorder.exportTask('${task.name}', 'javascript')">
                            <i class="fab fa-js-square"></i>
                        </button>
                        <button class="btn btn-outline-info" onclick="taskRecorder.exportTask('${task.name}', 'ahk')">
                            <i class="fas fa-keyboard"></i>
                        </button>
                        <button class="btn btn-outline-secondary" onclick="taskRecorder.exportTask('${task.name}', 'ahk2')">
                            <i class="fas fa-keyboard"></i>
                        </button>
                    </div>
                </td>
            `;
            tbody.appendChild(row);
        });
    }
    
    updateTaskSelect(tasks) {
        const select = document.getElementById('taskSelect');
        select.innerHTML = '<option value="">Choose a recorded task...</option>';
        
        tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task.name;
            option.textContent = task.name;
            select.appendChild(option);
        });
    }
    
    selectTask(taskName) {
        if (taskName) {
            this.selectedTask = taskName;
            document.getElementById('exportFormats').style.display = 'block';
        } else {
            this.selectedTask = null;
            document.getElementById('exportFormats').style.display = 'none';
            document.getElementById('codePreview').style.display = 'none';
        }
    }
    
    async previewExport(format) {
        if (!this.selectedTask) {
            this.showAlert('Please select a task first', 'warning');
            return;
        }
        
        try {
            const response = await fetch(`/api/macros/task-exports/${encodeURIComponent(this.selectedTask)}`);
            const result = await response.json();
            
            if (result.success && result.previews[format]) {
                const preview = result.previews[format];
                document.getElementById('codeContent').textContent = preview;
                document.getElementById('fileName').textContent = `${this.selectedTask}_macro.${this.getFileExtension(format)}`;
                document.getElementById('fileSize').textContent = preview.length;
                document.getElementById('codePreview').style.display = 'block';
                
                // Store format for download
                this.selectedFormat = format;
            } else {
                this.showAlert('Failed to load preview', 'danger');
            }
        } catch (error) {
            this.showAlert(`Error loading preview: ${error.message}`, 'danger');
        }
    }
    
    async downloadFile() {
        if (!this.selectedTask || !this.selectedFormat) {
            this.showAlert('Please select a task and format first', 'warning');
            return;
        }
        
        try {
            const url = `/api/macros/export-task/${encodeURIComponent(this.selectedTask)}/${this.selectedFormat}`;
            const link = document.createElement('a');
            link.href = url;
            link.download = `${this.selectedTask}_macro.${this.getFileExtension(this.selectedFormat)}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showAlert('File download started', 'success');
        } catch (error) {
            this.showAlert(`Error downloading file: ${error.message}`, 'danger');
        }
    }
    
    async exportTask(taskName, format) {
        try {
            const url = `/api/macros/export-task/${encodeURIComponent(taskName)}/${format}`;
            const link = document.createElement('a');
            link.href = url;
            link.download = `${taskName}_macro.${this.getFileExtension(format)}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showAlert(`${taskName} exported as ${format.toUpperCase()}`, 'success');
        } catch (error) {
            this.showAlert(`Error exporting task: ${error.message}`, 'danger');
        }
    }
    
    getFileExtension(format) {
        const extensions = {
            'json': 'json',
            'python': 'py',
            'javascript': 'js',
            'ahk': 'ahk',
            'ahk2': 'ahk'
        };
        return extensions[format] || 'txt';
    }
    
    clearForm() {
        document.getElementById('taskName').value = '';
        document.getElementById('taskDescription').value = '';
    }
    
    showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container-fluid');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize when page loads
let taskRecorder;
document.addEventListener('DOMContentLoaded', () => {
    taskRecorder = new TaskRecorder();
});