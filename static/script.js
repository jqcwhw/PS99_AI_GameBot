/**
 * AI Game Bot Dashboard JavaScript
 * Handles real-time updates and user interactions
 */

class GameBotDashboard {
    constructor() {
        this.updateInterval = 2000; // 2 seconds
        this.isRecording = false;
        this.commandHistory = [];
        this.trainingMode = false;
        this.currentTrainingMode = 'idle';
        this.zoneCorners = [];
        this.spacebarListener = null;
        
        this.initializeEventListeners();
        this.startStatusUpdates();
        this.loadMacros();
    }
    
    initializeEventListeners() {
        // Command execution
        document.getElementById('executeBtn').addEventListener('click', () => {
            this.executeCommand();
        });
        
        document.getElementById('commandInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.executeCommand();
            }
        });
        
        // Quick command buttons
        document.querySelectorAll('.quick-cmd').forEach(btn => {
            btn.addEventListener('click', () => {
                const command = btn.getAttribute('data-cmd');
                document.getElementById('commandInput').value = command;
                this.executeCommand();
            });
        });
        
        // Learning functionality
        document.getElementById('learnBtn').addEventListener('click', () => {
            this.learnFromSource();
        });
        
        document.getElementById('learnInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.learnFromSource();
            }
        });
        
        // Interactive Training controls
        document.getElementById('processCommandBtn').addEventListener('click', () => {
            this.processNaturalCommand();
        });
        
        document.getElementById('trainingCommand').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processNaturalCommand();
            }
        });
        
        document.querySelectorAll('.training-mode').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.getAttribute('data-mode');
                this.startTrainingMode(mode);
            });
        });
        
        document.getElementById('stopTrainingBtn').addEventListener('click', () => {
            this.stopTraining();
        });
        
        document.getElementById('analyzeSimilarBtn').addEventListener('click', () => {
            this.analyzeSimilarities();
        });
        
        document.getElementById('findItemsBtn').addEventListener('click', () => {
            this.findSimilarItems();
        });
        
        // Spacebar listener for item learning
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && this.trainingMode && this.currentTrainingMode === 'item_learning') {
                e.preventDefault();
                this.processSpacebarInput();
            }
        });
        
        // Screen click listener for zone mapping
        document.addEventListener('click', (e) => {
            if (this.trainingMode && this.currentTrainingMode === 'zone_mapping') {
                // Only process clicks on the screen capture area
                const screenImg = document.getElementById('screenCapture');
                if (screenImg && screenImg.contains(e.target)) {
                    this.processZoneClick(e);
                }
            }
        });
        
        // Macro controls
        document.getElementById('recordBtn').addEventListener('click', () => {
            this.toggleRecording();
        });
        
        document.getElementById('stopBtn').addEventListener('click', () => {
            this.stopRecording();
        });
        
        document.getElementById('playMacroBtn').addEventListener('click', () => {
            this.playMacro();
        });
    }
    
    async executeCommand() {
        const commandInput = document.getElementById('commandInput');
        const command = commandInput.value.trim();
        
        if (!command) return;
        
        // Add to history
        this.commandHistory.push({
            command: command,
            timestamp: new Date().toISOString()
        });
        
        // Show loading state
        const executeBtn = document.getElementById('executeBtn');
        const originalHTML = executeBtn.innerHTML;
        executeBtn.innerHTML = '<div class="loading-spinner"></div>';
        executeBtn.disabled = true;
        
        try {
            const response = await fetch('/api/command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ command: command })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayCommandResult(result);
                commandInput.value = ''; // Clear input on success
            } else {
                this.displayError(`Command failed: ${result.error}`);
            }
        } catch (error) {
            this.displayError(`Network error: ${error.message}`);
        } finally {
            executeBtn.innerHTML = originalHTML;
            executeBtn.disabled = false;
        }
    }
    
    async learnFromSource() {
        const learnInput = document.getElementById('learnInput');
        const source = learnInput.value.trim();
        
        if (!source) return;
        
        const learnBtn = document.getElementById('learnBtn');
        const originalHTML = learnBtn.innerHTML;
        learnBtn.innerHTML = '<div class="loading-spinner"></div>';
        learnBtn.disabled = true;
        
        try {
            const response = await fetch('/api/learn', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    source: source,
                    type: 'auto'
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayCommandResult({
                    command: `learn from ${source}`,
                    result: result.result,
                    timestamp: result.timestamp
                });
                learnInput.value = ''; // Clear input on success
            } else {
                this.displayError(`Learning failed: ${result.error}`);
            }
        } catch (error) {
            this.displayError(`Network error: ${error.message}`);
        } finally {
            learnBtn.innerHTML = originalHTML;
            learnBtn.disabled = false;
        }
    }
    
    displayCommandResult(result) {
        const resultsDiv = document.getElementById('commandResults');
        const timestamp = new Date(result.timestamp).toLocaleTimeString();
        
        const resultHTML = `
            <div class="mb-2">
                <span class="command-timestamp">[${timestamp}]</span>
                <strong>Command:</strong> ${this.escapeHtml(result.command)}
            </div>
            <div class="mb-3 result-success">
                <strong>Result:</strong> ${this.escapeHtml(result.result)}
            </div>
        `;
        
        resultsDiv.innerHTML = resultHTML + resultsDiv.innerHTML;
        
        // Keep only last 20 results
        const results = resultsDiv.children;
        while (results.length > 40) { // 40 because each result creates 2 divs
            resultsDiv.removeChild(results[results.length - 1]);
        }
    }
    
    displayError(message) {
        const resultsDiv = document.getElementById('commandResults');
        const timestamp = new Date().toLocaleTimeString();
        
        const errorHTML = `
            <div class="mb-3 result-error">
                <span class="command-timestamp">[${timestamp}]</span>
                <strong>Error:</strong> ${this.escapeHtml(message)}
            </div>
        `;
        
        resultsDiv.innerHTML = errorHTML + resultsDiv.innerHTML;
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            if (response.ok) {
                this.updateStatusDisplay(status);
                this.updateScreenCapture(status.screen_capture);
            } else {
                console.error('Failed to fetch status:', status.error);
                this.updateConnectionStatus(false);
            }
        } catch (error) {
            console.error('Status update failed:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    updateStatusDisplay(status) {
        // Update timestamp
        const timestamp = new Date(status.timestamp).toLocaleTimeString();
        document.getElementById('timestamp').textContent = timestamp;
        
        // Update system status badges
        const visionStatus = document.getElementById('visionStatus');
        const automationStatus = document.getElementById('automationStatus');
        
        visionStatus.textContent = status.vision_active ? 'Active' : 'Inactive';
        visionStatus.className = `badge ${status.vision_active ? 'bg-success' : 'bg-secondary'}`;
        
        automationStatus.textContent = status.automation_active ? 'Active' : 'Inactive';
        automationStatus.className = `badge ${status.automation_active ? 'bg-success' : 'bg-secondary'}`;
        
        // Update counts
        document.getElementById('macroCount').textContent = status.macro_count;
        document.getElementById('knowledgeCount').textContent = status.knowledge_count;
        document.getElementById('knowledgeCount2').textContent = status.knowledge_count;
        
        // Update learning stats
        const learningStats = document.getElementById('learningStats');
        if (status.learning_stats && typeof status.learning_stats === 'object') {
            learningStats.textContent = `Learned: ${status.learning_stats.items_learned || 0}`;
        } else {
            learningStats.textContent = 'No data';
        }
        
        // Update connection status
        this.updateConnectionStatus(true);
    }
    
    updateScreenCapture(screenB64) {
        const screenImg = document.getElementById('screenCapture');
        const noCapture = document.getElementById('noCapture');
        
        if (screenB64) {
            screenImg.src = `data:image/jpeg;base64,${screenB64}`;
            screenImg.style.display = 'block';
            noCapture.style.display = 'none';
        } else {
            screenImg.style.display = 'none';
            noCapture.style.display = 'flex';
        }
    }
    
    updateConnectionStatus(connected) {
        const statusBadge = document.getElementById('status-badge');
        
        if (connected) {
            statusBadge.textContent = 'Connected';
            statusBadge.className = 'badge bg-success me-2';
        } else {
            statusBadge.textContent = 'Disconnected';
            statusBadge.className = 'badge bg-danger me-2';
        }
    }
    
    async loadMacros() {
        try {
            const response = await fetch('/api/macros');
            const data = await response.json();
            
            const macroSelect = document.getElementById('macroSelect');
            
            if (response.ok && data.macros && data.macros.length > 0) {
                macroSelect.innerHTML = data.macros.map(macro => 
                    `<option value="${macro}">${macro}</option>`
                ).join('');
            } else {
                macroSelect.innerHTML = '<option>No macros available</option>';
            }
        } catch (error) {
            console.error('Failed to load macros:', error);
            document.getElementById('macroSelect').innerHTML = '<option>Failed to load macros</option>';
        }
    }
    
    toggleRecording() {
        const recordBtn = document.getElementById('recordBtn');
        
        if (this.isRecording) {
            this.stopRecording();
        } else {
            // Start recording
            this.isRecording = true;
            recordBtn.innerHTML = '<i class="fas fa-stop me-1"></i>Recording...';
            recordBtn.className = 'btn btn-danger btn-sm';
            
            // Execute record command
            this.executeRecordCommand('record macro new_macro');
        }
    }
    
    stopRecording() {
        if (this.isRecording) {
            this.isRecording = false;
            
            const recordBtn = document.getElementById('recordBtn');
            recordBtn.innerHTML = '<i class="fas fa-record-vinyl me-1"></i>Record';
            recordBtn.className = 'btn btn-outline-danger btn-sm';
            
            // Execute stop command
            this.executeRecordCommand('stop recording');
            
            // Refresh macros list
            setTimeout(() => this.loadMacros(), 1000);
        }
    }
    
    async executeRecordCommand(command) {
        try {
            await fetch('/api/command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ command: command })
            });
        } catch (error) {
            console.error('Failed to execute record command:', error);
        }
    }
    
    async playMacro() {
        const macroSelect = document.getElementById('macroSelect');
        const selectedMacro = macroSelect.value;
        
        if (selectedMacro && selectedMacro !== 'No macros available' && selectedMacro !== 'Failed to load macros') {
            const command = `play macro ${selectedMacro}`;
            document.getElementById('commandInput').value = command;
            await this.executeCommand();
        }
    }
    
    startStatusUpdates() {
        // Initial update
        this.updateStatus();
        
        // Set up periodic updates
        setInterval(() => {
            this.updateStatus();
        }, this.updateInterval);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Interactive Training Methods
    async processNaturalCommand() {
        const command = document.getElementById('trainingCommand').value.trim();
        if (!command) return;
        
        try {
            const response = await fetch('/api/training/command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ command: command })
            });
            
            const data = await response.json();
            this.showAlert(data.result || data.error, data.success ? 'success' : 'danger');
            
            // Clear input and update status
            document.getElementById('trainingCommand').value = '';
            this.updateTrainingStatus();
            
        } catch (error) {
            this.showAlert('Failed to process command', 'danger');
        }
    }
    
    async startTrainingMode(mode) {
        try {
            const response = await fetch('/api/training/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ mode: mode })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.trainingMode = true;
                this.currentTrainingMode = mode;
                this.zoneCorners = [];
                
                // Update UI
                this.updateTrainingModeButtons(mode);
                this.showAlert(data.result, 'success');
                
                // Show specific instructions
                this.showTrainingInstructions(mode);
            } else {
                this.showAlert(data.error || 'Failed to start training', 'danger');
            }
            
            this.updateTrainingStatus();
            
        } catch (error) {
            this.showAlert('Failed to start training mode', 'danger');
        }
    }
    
    async stopTraining() {
        try {
            const response = await fetch('/api/training/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            this.trainingMode = false;
            this.currentTrainingMode = 'idle';
            this.zoneCorners = [];
            
            // Reset UI
            this.resetTrainingModeButtons();
            this.showAlert(data.result || 'Training stopped', 'info');
            this.updateTrainingStatus();
            
        } catch (error) {
            this.showAlert('Failed to stop training', 'danger');
        }
    }
    
    async processSpacebarInput() {
        try {
            const itemType = prompt('What type of item is this? (chest, egg, breakable, resource, etc.)');
            if (!itemType) return;
            
            const response = await fetch('/api/training/spacebar', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ item_type: itemType })
            });
            
            const data = await response.json();
            this.showAlert(data.result || data.error, data.success ? 'success' : 'danger');
            this.updateTrainingStatus();
            
        } catch (error) {
            this.showAlert('Failed to process spacebar input', 'danger');
        }
    }
    
    async processZoneClick(event) {
        // Get click coordinates relative to the screen capture
        const rect = event.target.getBoundingClientRect();
        const x = Math.round(event.clientX - rect.left);
        const y = Math.round(event.clientY - rect.top);
        
        try {
            const response = await fetch('/api/training/corner', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ x: x, y: y })
            });
            
            const data = await response.json();
            this.showAlert(data.result || data.error, data.success ? 'success' : 'danger');
            
            // Update zone progress
            this.zoneCorners.push({ x, y });
            document.getElementById('zoneProgress').textContent = `${this.zoneCorners.length}/4 corners`;
            
            // If zone is complete, reset
            if (this.zoneCorners.length >= 4) {
                setTimeout(() => {
                    this.zoneCorners = [];
                    document.getElementById('zoneProgress').textContent = '0/4 corners';
                }, 2000);
            }
            
            this.updateTrainingStatus();
            
        } catch (error) {
            this.showAlert('Failed to process zone click', 'danger');
        }
    }
    
    async analyzeSimilarities() {
        try {
            const response = await fetch('/api/training/similarities');
            const data = await response.json();
            this.showAlert(data.result || data.error, data.success ? 'success' : 'danger');
            
        } catch (error) {
            this.showAlert('Failed to analyze similarities', 'danger');
        }
    }
    
    async findSimilarItems() {
        try {
            const threshold = parseFloat(prompt('Similarity threshold (0.0 - 1.0):') || '0.8');
            const response = await fetch(`/api/training/find-items?threshold=${threshold}`);
            const data = await response.json();
            
            if (data.success && data.matches) {
                this.showAlert(`Found ${data.matches.length} similar items on screen`, 'success');
                this.highlightFoundItems(data.matches);
            } else {
                this.showAlert(data.error || 'No similar items found', 'warning');
            }
            
        } catch (error) {
            this.showAlert('Failed to find similar items', 'danger');
        }
    }
    
    highlightFoundItems(matches) {
        // Clear previous highlights
        const existingHighlights = document.querySelectorAll('.item-highlight');
        existingHighlights.forEach(h => h.remove());
        
        // Add new highlights
        const screenImg = document.getElementById('screenCapture');
        if (!screenImg) return;
        
        matches.forEach(match => {
            const highlight = document.createElement('div');
            highlight.className = 'item-highlight';
            highlight.style.position = 'absolute';
            highlight.style.left = `${match.position[0]}px`;
            highlight.style.top = `${match.position[1]}px`;
            highlight.style.width = '50px';
            highlight.style.height = '50px';
            highlight.style.border = '2px solid #ff0000';
            highlight.style.backgroundColor = 'rgba(255, 0, 0, 0.2)';
            highlight.style.pointerEvents = 'none';
            highlight.style.zIndex = '1000';
            highlight.title = `${match.item_type} (${Math.round(match.confidence * 100)}%)`;
            
            screenImg.parentElement.style.position = 'relative';
            screenImg.parentElement.appendChild(highlight);
        });
        
        // Remove highlights after 5 seconds
        setTimeout(() => {
            const highlights = document.querySelectorAll('.item-highlight');
            highlights.forEach(h => h.remove());
        }, 5000);
    }
    
    updateTrainingModeButtons(activeMode) {
        document.querySelectorAll('.training-mode').forEach(btn => {
            const mode = btn.getAttribute('data-mode');
            if (mode === activeMode) {
                btn.classList.remove('btn-outline-success', 'btn-outline-info', 'btn-outline-warning');
                btn.classList.add('btn-success');
            } else {
                btn.classList.add('btn-outline-success', 'btn-outline-info', 'btn-outline-warning');
                btn.classList.remove('btn-success');
            }
        });
    }
    
    resetTrainingModeButtons() {
        document.querySelectorAll('.training-mode').forEach(btn => {
            btn.classList.add('btn-outline-success', 'btn-outline-info', 'btn-outline-warning');
            btn.classList.remove('btn-success');
        });
    }
    
    showTrainingInstructions(mode) {
        let instructions = '';
        switch (mode) {
            case 'item_learning':
                instructions = 'Hover your mouse over items and press SPACE to learn them. You\'ll be prompted for the item type.';
                break;
            case 'zone_mapping':
                instructions = 'Click 4 corners on the screen to define a game zone boundary.';
                break;
            case 'gameplay_recording':
                instructions = 'The bot will now watch and learn from your gameplay actions.';
                break;
        }
        
        if (instructions) {
            this.showAlert(instructions, 'info');
        }
    }
    
    async updateTrainingStatus() {
        try {
            const response = await fetch('/api/training/status');
            const status = await response.json();
            
            // Update status display
            const statusElement = document.getElementById('trainingStatus');
            if (status.training_mode) {
                statusElement.textContent = `Active: ${status.current_mode}`;
                statusElement.className = 'text-success';
            } else {
                statusElement.textContent = 'Not active';
                statusElement.className = 'text-muted';
            }
            
            // Update counters
            document.getElementById('learnedItemsCount').textContent = status.learned_items_count || 0;
            document.getElementById('gameZonesCount').textContent = status.game_zones_count || 0;
            document.getElementById('zoneProgress').textContent = `${status.zone_corners_collected || 0}/4 corners`;
            
        } catch (error) {
            console.error('Failed to update training status:', error);
        }
    }
    
    showAlert(message, type = 'info') {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.style.position = 'fixed';
        alert.style.top = '20px';
        alert.style.right = '20px';
        alert.style.zIndex = '9999';
        alert.style.maxWidth = '400px';
        alert.innerHTML = `
            ${this.escapeHtml(message)}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new GameBotDashboard();
});

// Missing Multi-Instance Control Functions
function toggleMutexBypass() {
    console.log("Toggling mutex bypass...");
    fetch('/api/mutex/toggle', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Mutex bypass toggled:', data);
        updateMutexStatus(data.active);
    })
    .catch(error => {
        console.error('Error toggling mutex bypass:', error);
    });
}

function updateMutexStatus(active) {
    const badge = document.getElementById('mutex-badge');
    const button = document.getElementById('toggle-mutex');
    
    if (active) {
        badge.className = 'badge bg-success';
        badge.textContent = 'Bypass Active';
        button.textContent = 'Deactivate';
        button.className = 'btn btn-sm btn-outline-danger ms-2';
    } else {
        badge.className = 'badge bg-danger';
        badge.textContent = 'Bypass Inactive';
        button.textContent = 'Activate';
        button.className = 'btn btn-sm btn-outline-primary ms-2';
    }
}

function launchPS99Instance() {
    console.log("Launching PS99 instance...");
    showModal('Launch PS99 Instance', `
        <div class="mb-3">
            <label for="accountName" class="form-label">Account Name</label>
            <input type="text" class="form-control" id="accountName" placeholder="Enter account name">
        </div>
        <div class="mb-3">
            <label for="serverRegion" class="form-label">Server Region</label>
            <select class="form-select" id="serverRegion">
                <option value="us-east">US East</option>
                <option value="us-west">US West</option>
                <option value="eu">Europe</option>
                <option value="asia">Asia</option>
            </select>
        </div>
        <div class="mb-3">
            <div class="form-check">
                <input class="form-check-input" type="checkbox" id="enableSync" checked>
                <label class="form-check-label" for="enableSync">
                    Enable synchronization with other instances
                </label>
            </div>
        </div>
    `, 'Launch Instance', () => {
        const accountName = document.getElementById('accountName').value;
        const serverRegion = document.getElementById('serverRegion').value;
        const enableSync = document.getElementById('enableSync').checked;
        
        fetch('/api/instances/launch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                account_name: accountName,
                server_region: serverRegion,
                enable_sync: enableSync
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Instance launched:', data);
            showNotification('Instance launched successfully', 'success');
        })
        .catch(error => {
            console.error('Error launching instance:', error);
            showNotification('Failed to launch instance', 'error');
        });
    });
}

function toggleSyncSystem() {
    console.log("Toggling sync system...");
    const button = document.getElementById('syncToggleBtn');
    const isActive = button.textContent.includes('Disable');
    
    fetch('/api/sync/toggle', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enable: !isActive })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Sync system toggled:', data);
        updateSyncStatus(data.active);
        
        // Show/hide shadow following panel
        const panel = document.getElementById('shadowFollowingPanel');
        const coordPanel = document.getElementById('coordinatedCommandsPanel');
        if (data.active) {
            panel.style.display = 'block';
            coordPanel.style.display = 'block';
        } else {
            panel.style.display = 'none';
            coordPanel.style.display = 'none';
        }
    })
    .catch(error => {
        console.error('Error toggling sync system:', error);
    });
}

function updateSyncStatus(active) {
    const button = document.getElementById('syncToggleBtn');
    const text = document.getElementById('syncToggleText');
    
    if (active) {
        text.textContent = 'Disable Sync';
        button.className = 'btn btn-outline-danger btn-sm';
    } else {
        text.textContent = 'Enable Sync';
        button.className = 'btn btn-outline-primary btn-sm';
    }
}

function showAccountManager() {
    console.log("Opening account manager...");
    showModal('Roblox Account Manager', `
        <div class="row">
            <div class="col-md-5">
                <h6>Add New Roblox Account</h6>
                <div class="mb-2">
                    <input type="text" class="form-control form-control-sm" id="robloxUsername" placeholder="Roblox Username">
                </div>
                <div class="mb-2">
                    <input type="text" class="form-control form-control-sm" id="robloxDisplayName" placeholder="Display Name (Optional)">
                </div>
                <div class="mb-2">
                    <select class="form-select form-select-sm" id="accountType">
                        <option value="standard">Standard Account</option>
                        <option value="premium">Premium Account</option>
                        <option value="alt">Alt Account</option>
                    </select>
                </div>
                <div class="mb-2">
                    <input type="password" class="form-control form-control-sm" id="authCookie" placeholder="Auth Cookie (Optional)">
                </div>
                <hr>
                <h6>Natro Sync Settings</h6>
                <div class="mb-2">
                    <select class="form-select form-select-sm" id="syncRole">
                        <option value="follower">Follower</option>
                        <option value="leader">Leader</option>
                    </select>
                </div>
                <div class="mb-2" id="leaderSelection" style="display: none;">
                    <select class="form-select form-select-sm" id="leaderAccount">
                        <option value="">Select Leader Account...</option>
                    </select>
                </div>
                <hr>
                <h6>Discord Bot Integration</h6>
                <div class="mb-2">
                    <input type="password" class="form-control form-control-sm" id="discordBotToken" placeholder="Discord Bot Token (Optional)">
                </div>
                <div class="mb-2">
                    <input type="text" class="form-control form-control-sm" id="discordChannelId" placeholder="Discord Channel ID (Optional)">
                </div>
                <div class="mb-3">
                    <button onclick="testDiscordBot()" class="btn btn-outline-secondary btn-sm">Test Discord Bot</button>
                </div>
                <button onclick="addRobloxAccount()" class="btn btn-primary btn-sm w-100">Add Roblox Account</button>
            </div>
            <div class="col-md-7">
                <h6>Current Roblox Accounts</h6>
                <div id="accountsList" class="list-group" style="max-height: 400px; overflow-y: auto;">
                    <div class="text-muted">Loading accounts...</div>
                </div>
                <hr>
                <div class="row">
                    <div class="col-6">
                        <small class="text-muted">
                            <strong>Leaders:</strong> <span id="leadersCount">0</span><br>
                            <strong>Followers:</strong> <span id="followersCount">0</span>
                        </small>
                    </div>
                    <div class="col-6">
                        <small class="text-muted">
                            <strong>Active:</strong> <span id="activeCount">0</span><br>
                            <strong>Discord Configured:</strong> <span id="discordCount">0</span>
                        </small>
                    </div>
                </div>
            </div>
        </div>
    `, 'Close', null, 'xl');
    
    // Setup role selection handler
    document.getElementById('syncRole').addEventListener('change', function() {
        const leaderSelection = document.getElementById('leaderSelection');
        if (this.value === 'follower') {
            leaderSelection.style.display = 'block';
            loadLeaderAccounts();
        } else {
            leaderSelection.style.display = 'none';
        }
    });
    
    // Load current accounts
    loadRobloxAccounts();
}

function addRobloxAccount() {
    const username = document.getElementById('robloxUsername').value.trim();
    const displayName = document.getElementById('robloxDisplayName').value.trim();
    const accountType = document.getElementById('accountType').value;
    const authCookie = document.getElementById('authCookie').value.trim();
    const syncRole = document.getElementById('syncRole').value;
    const leaderAccount = document.getElementById('leaderAccount').value;
    const discordBotToken = document.getElementById('discordBotToken').value.trim();
    const discordChannelId = document.getElementById('discordChannelId').value.trim();
    
    if (!username) {
        showToast('Please enter a Roblox username', 'warning');
        return;
    }
    
    const accountData = {
        username: username,
        display_name: displayName || username,
        account_type: accountType,
        auth_cookie: authCookie,
        sync_role: syncRole,
        leader_account: leaderAccount,
        discord_bot_token: discordBotToken,
        discord_channel_id: discordChannelId
    };
    
    fetch('/api/roblox/accounts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(accountData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast(`Roblox account ${username} added successfully!`, 'success');
            loadRobloxAccounts();
            // Clear form
            document.getElementById('robloxUsername').value = '';
            document.getElementById('robloxDisplayName').value = '';
            document.getElementById('authCookie').value = '';
            document.getElementById('discordBotToken').value = '';
            document.getElementById('discordChannelId').value = '';
        } else {
            showToast(`Failed to add account: ${data.message}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error adding Roblox account:', error);
        showToast('Error adding Roblox account', 'error');
    });
}

function loadRobloxAccounts() {
    fetch('/api/roblox/accounts')
        .then(response => response.json())
        .then(data => {
            const accountsList = document.getElementById('accountsList');
            if (data.success) {
                displayRobloxAccounts(data.data);
            } else {
                accountsList.innerHTML = '<div class="text-danger">Failed to load accounts</div>';
            }
        })
        .catch(error => {
            console.error('Error loading Roblox accounts:', error);
            document.getElementById('accountsList').innerHTML = '<div class="text-danger">Error loading accounts</div>';
        });
}

function displayRobloxAccounts(summary) {
    const accountsList = document.getElementById('accountsList');
    
    // Update summary counts
    document.getElementById('leadersCount').textContent = summary.leaders || 0;
    document.getElementById('followersCount').textContent = summary.followers || 0;
    document.getElementById('activeCount').textContent = summary.active_accounts || 0;
    document.getElementById('discordCount').textContent = summary.discord_configured || 0;
    
    if (!summary.accounts || summary.accounts.length === 0) {
        accountsList.innerHTML = '<div class="text-muted">No Roblox accounts configured</div>';
        return;
    }
    
    let accountsHtml = '';
    summary.accounts.forEach(account => {
        const statusBadge = account.is_active ? 
            '<span class="badge bg-success">Active</span>' : 
            '<span class="badge bg-secondary">Inactive</span>';
        
        const roleBadge = account.role === 'leader' ? 
            '<span class="badge bg-primary">Leader</span>' : 
            '<span class="badge bg-info">Follower</span>';
        
        const discordBadge = account.discord_configured ? 
            '<span class="badge bg-success">Discord ✓</span>' : 
            '<span class="badge bg-warning">No Discord</span>';
        
        accountsHtml += `
            <div class="list-group-item d-flex justify-content-between align-items-start">
                <div class="ms-2 me-auto">
                    <div class="fw-bold">${account.display_name}</div>
                    <small class="text-muted">@${account.username}</small>
                    <div class="mt-1">
                        ${statusBadge} ${roleBadge} ${discordBadge}
                    </div>
                    ${account.ps99_zone ? `<small class="text-muted">Zone: ${account.ps99_zone}</small>` : ''}
                    ${account.automation_enabled ? '<small class="text-success d-block">Automation Enabled</small>' : ''}
                </div>
                <div class="btn-group btn-group-sm" role="group">
                    <button onclick="launchRobloxInstance('${account.username}')" class="btn btn-outline-primary btn-sm" title="Launch Instance">
                        🚀
                    </button>
                    <button onclick="toggleAccountSync('${account.username}')" class="btn btn-outline-info btn-sm" title="Toggle Sync">
                        🔄
                    </button>
                    <button onclick="removeRobloxAccount('${account.username}')" class="btn btn-outline-danger btn-sm" title="Remove Account">
                        🗑️
                    </button>
                </div>
            </div>
        `;
    });
    
    accountsList.innerHTML = accountsHtml;
}

function loadLeaderAccounts() {
    fetch('/api/roblox/accounts/leaders')
        .then(response => response.json())
        .then(data => {
            const leaderSelect = document.getElementById('leaderAccount');
            leaderSelect.innerHTML = '<option value="">Select Leader Account...</option>';
            
            if (data.success && data.leaders) {
                data.leaders.forEach(leader => {
                    leaderSelect.innerHTML += `<option value="${leader}">${leader}</option>`;
                });
            }
        })
        .catch(error => {
            console.error('Error loading leader accounts:', error);
        });
}

function testDiscordBot() {
    const botToken = document.getElementById('discordBotToken').value.trim();
    const channelId = document.getElementById('discordChannelId').value.trim();
    
    if (!botToken || !channelId) {
        showToast('Please enter both Discord bot token and channel ID', 'warning');
        return;
    }
    
    fetch('/api/roblox/test-discord', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            bot_token: botToken,
            channel_id: channelId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Discord bot connection successful!', 'success');
        } else {
            showToast(`Discord bot test failed: ${data.message}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error testing Discord bot:', error);
        showToast('Error testing Discord bot', 'error');
    });
}

function launchRobloxInstance(username) {
    fetch('/api/roblox/launch-instance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            username: username,
            enable_sync: true
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast(`Launching Roblox instance for ${username}`, 'success');
            loadRobloxAccounts(); // Refresh account list
        } else {
            showToast(`Failed to launch instance: ${data.message}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error launching Roblox instance:', error);
        showToast('Error launching Roblox instance', 'error');
    });
}

function toggleAccountSync(username) {
    fetch('/api/roblox/toggle-sync', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            username: username
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast(`Sync toggled for ${username}`, 'success');
            loadRobloxAccounts(); // Refresh account list
        } else {
            showToast(`Failed to toggle sync: ${data.message}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error toggling account sync:', error);
        showToast('Error toggling account sync', 'error');
    });
}

function removeRobloxAccount(username) {
    if (!confirm(`Are you sure you want to remove account "${username}"?`)) {
        return;
    }
    
    fetch(`/api/roblox/accounts/${username}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast(`Account ${username} removed successfully`, 'success');
            loadRobloxAccounts(); // Refresh account list
        } else {
            showToast(`Failed to remove account: ${data.message}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error removing Roblox account:', error);
        showToast('Error removing Roblox account', 'error');
    });
}

function showInstanceMonitor() {
    console.log("Opening instance monitor...");
    showModal('Instance Monitor', `
        <div class="row">
            <div class="col-12 mb-3">
                <div class="d-flex justify-content-between align-items-center">
                    <h6>Active Instances</h6>
                    <button onclick="refreshInstances()" class="btn btn-outline-primary btn-sm">
                        <i class="fas fa-refresh"></i> Refresh
                    </button>
                </div>
            </div>
            <div class="col-12">
                <div id="instancesList">
                    <div class="text-muted">Loading instances...</div>
                </div>
            </div>
        </div>
    `, 'Close', null, 'large');
    
    // Load current instances
    loadInstances();
}

function setupShadowFollowing() {
    const leader = document.getElementById('leaderAccount').value;
    const followers = Array.from(document.getElementById('followerAccounts').selectedOptions).map(option => option.value);
    
    if (!leader || followers.length === 0) {
        showNotification('Please select a leader and at least one follower', 'warning');
        return;
    }
    
    fetch('/api/sync/setup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            leader: leader,
            followers: followers,
            max_follow_time: 900
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Shadow following setup:', data);
        showNotification('Shadow following configured successfully', 'success');
    })
    .catch(error => {
        console.error('Error setting up shadow following:', error);
        showNotification('Failed to setup shadow following', 'error');
    });
}

function showSyncStatus() {
    fetch('/api/sync/status')
        .then(response => response.json())
        .then(data => {
            showModal('Synchronization Status', `
                <div class="row">
                    <div class="col-md-6">
                        <h6>System Status</h6>
                        <ul class="list-unstyled">
                            <li><strong>Sync Active:</strong> ${data.sync_active ? 'Yes' : 'No'}</li>
                            <li><strong>Field Following:</strong> ${data.field_following_enabled ? 'Enabled' : 'Disabled'}</li>
                            <li><strong>Shadow Mode:</strong> ${data.shadow_mode ? 'Active' : 'Inactive'}</li>
                            <li><strong>Max Follow Time:</strong> ${data.max_follow_time}s</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>Account Statistics</h6>
                        <ul class="list-unstyled">
                            <li><strong>Total Accounts:</strong> ${data.total_accounts}</li>
                            <li><strong>Active Accounts:</strong> ${data.active_accounts}</li>
                            <li><strong>Leaders:</strong> ${data.leaders}</li>
                            <li><strong>Followers:</strong> ${data.followers}</li>
                            <li><strong>Action Queue:</strong> ${data.action_queue_size}</li>
                        </ul>
                    </div>
                </div>
            `, 'Close');
        })
        .catch(error => {
            console.error('Error fetching sync status:', error);
        });
}

function sendCoordinatedCommand(command, params = {}) {
    console.log(`Sending coordinated command: ${command}`, params);
    
    fetch('/api/sync/command', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            command: command,
            params: params
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Coordinated command sent:', data);
        showNotification(`Coordinated command "${command}" sent to all instances`, 'info');
    })
    .catch(error => {
        console.error('Error sending coordinated command:', error);
        showNotification('Failed to send coordinated command', 'error');
    });
}

// Helper functions
function addAccount() {
    const name = document.getElementById('newAccountName').value;
    const role = document.getElementById('newAccountRole').value;
    const channelId = document.getElementById('discordChannelId').value;
    
    if (!name) {
        showNotification('Please enter an account name', 'warning');
        return;
    }
    
    fetch('/api/accounts/add', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            name: name,
            role: role,
            discord_channel_id: channelId || null
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Account added:', data);
        showNotification('Account added successfully', 'success');
        loadAccounts();
        // Clear form
        document.getElementById('newAccountName').value = '';
        document.getElementById('discordChannelId').value = '';
    })
    .catch(error => {
        console.error('Error adding account:', error);
        showNotification('Failed to add account', 'error');
    });
}

function loadAccounts() {
    fetch('/api/accounts/list')
        .then(response => response.json())
        .then(data => {
            const accountsList = document.getElementById('accountsList');
            if (data.accounts && data.accounts.length > 0) {
                accountsList.innerHTML = data.accounts.map(account => `
                    <div class="list-group-item d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${account.name}</strong>
                            <small class="text-muted d-block">${account.role}</small>
                        </div>
                        <div>
                            <span class="badge ${account.is_active ? 'bg-success' : 'bg-secondary'}">${account.is_active ? 'Active' : 'Inactive'}</span>
                            <button onclick="removeAccount('${account.id}')" class="btn btn-outline-danger btn-sm ms-2">Remove</button>
                        </div>
                    </div>
                `).join('');
                
                // Update select options
                updateAccountSelects(data.accounts);
            } else {
                accountsList.innerHTML = '<div class="text-muted">No accounts configured</div>';
            }
        })
        .catch(error => {
            console.error('Error loading accounts:', error);
        });
}

function loadInstances() {
    fetch('/api/instances/list')
        .then(response => response.json())
        .then(data => {
            const instancesList = document.getElementById('instancesList');
            if (data.instances && data.instances.length > 0) {
                instancesList.innerHTML = data.instances.map(instance => `
                    <div class="card mb-2">
                        <div class="card-body p-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="card-title mb-1">${instance.account_name}</h6>
                                    <small class="text-muted">PID: ${instance.pid} | HWND: ${instance.hwnd || 'N/A'}</small>
                                </div>
                                <div>
                                    <span class="badge ${instance.status === 'running' ? 'bg-success' : 'bg-warning'}">${instance.status}</span>
                                    <button onclick="terminateInstance('${instance.id}')" class="btn btn-outline-danger btn-sm ms-2">Terminate</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('');
            } else {
                instancesList.innerHTML = '<div class="text-muted">No active instances</div>';
            }
        })
        .catch(error => {
            console.error('Error loading instances:', error);
        });
}

function updateAccountSelects(accounts) {
    const leaderSelect = document.getElementById('leaderAccount');
    const followerSelect = document.getElementById('followerAccounts');
    
    if (leaderSelect) {
        leaderSelect.innerHTML = '<option value="">Select leader...</option>' +
            accounts.map(account => `<option value="${account.id}">${account.name} (${account.role})</option>`).join('');
    }
    
    if (followerSelect) {
        followerSelect.innerHTML = 
            accounts.map(account => `<option value="${account.id}">${account.name} (${account.role})</option>`).join('');
    }
}

function removeAccount(accountId) {
    if (!confirm('Are you sure you want to remove this account?')) return;
    
    fetch(`/api/accounts/remove/${accountId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Account removed:', data);
        showNotification('Account removed successfully', 'success');
        loadAccounts();
    })
    .catch(error => {
        console.error('Error removing account:', error);
        showNotification('Failed to remove account', 'error');
    });
}

function terminateInstance(instanceId) {
    if (!confirm('Are you sure you want to terminate this instance?')) return;
    
    fetch(`/api/instances/terminate/${instanceId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Instance terminated:', data);
        showNotification('Instance terminated successfully', 'success');
        loadInstances();
    })
    .catch(error => {
        console.error('Error terminating instance:', error);
        showNotification('Failed to terminate instance', 'error');
    });
}

function refreshInstances() {
    loadInstances();
}

// Modal and notification utilities
function showModal(title, content, confirmText = 'OK', confirmCallback = null, size = 'default') {
    const sizeClass = size === 'large' ? 'modal-lg' : '';
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog ${sizeClass}">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    ${confirmCallback ? `<button type="button" class="btn btn-primary" onclick="document.getElementById('confirmModalBtn').click()">${confirmText}</button>` : ''}
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    if (confirmCallback) {
        const confirmBtn = document.createElement('button');
        confirmBtn.id = 'confirmModalBtn';
        confirmBtn.style.display = 'none';
        confirmBtn.onclick = () => {
            confirmCallback();
            bootstrap.Modal.getInstance(modal).hide();
        };
        document.body.appendChild(confirmBtn);
    }
    
    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();
    
    modal.addEventListener('hidden.bs.modal', () => {
        document.body.removeChild(modal);
        const confirmBtn = document.getElementById('confirmModalBtn');
        if (confirmBtn) document.body.removeChild(confirmBtn);
    });
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

// Navigation functions
function openMultiInstanceLauncher() {
    window.location.href = '/multi-instance-launcher';
}

function openItemMapper() {
    window.location.href = '/item-mapper';
}

function openPS99EggHatcher() {
    window.location.href = '/ps99-egg-hatcher';
}

function openAccountManager() {
    alert('Account Manager - Coming Soon!\n\nThis will allow you to:\n• Manage multiple Roblox accounts\n• Configure synchronization settings\n• Monitor account status');
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new GameBotDashboard();
});
