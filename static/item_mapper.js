class ItemMapper {
    constructor() {
        this.gameWindow = null;
        this.mappingMode = false;
        this.currentItemType = null;
        this.mappedItems = [];
        this.isLearning = false;
        this.isRecording = false;
        this.isAutoPlaying = false;
        this.screenCapture = null;
        this.updateInterval = 2000; // 2 seconds
        
        this.initializeEventListeners();
        this.startStatusUpdates();
        this.loadMappedItems();
    }

    initializeEventListeners() {
        // Screen detection
        document.getElementById('detectGameBtn').addEventListener('click', () => this.detectGameWindow());
        
        // Item mapping
        document.getElementById('itemTypeSelect').addEventListener('change', (e) => this.selectItemType(e.target.value));
        document.getElementById('startMappingBtn').addEventListener('click', () => this.startMapping());
        document.getElementById('saveItemBtn').addEventListener('click', () => this.saveItem());
        document.getElementById('cancelMappingBtn').addEventListener('click', () => this.cancelMapping());
        
        // AI Learning controls
        document.getElementById('autoPlayBtn').addEventListener('click', () => this.startAutoPlay());
        document.getElementById('stopAutoPlayBtn').addEventListener('click', () => this.stopAutoPlay());
        document.getElementById('learnGameplayBtn').addEventListener('click', () => this.startLearning());
        document.getElementById('stopLearningBtn').addEventListener('click', () => this.stopLearning());
        document.getElementById('recordMacroBtn').addEventListener('click', () => this.startRecording());
        document.getElementById('stopRecordingBtn').addEventListener('click', () => this.stopRecording());
        document.getElementById('watchAndCreateBtn').addEventListener('click', () => this.startWatchAndCreate());
        
        // Screen capture controls
        document.getElementById('captureScreenBtn').addEventListener('click', () => this.captureScreen());
        document.getElementById('liveCapture').addEventListener('change', (e) => this.toggleLiveCapture(e.target.checked));
        
        // Data management
        document.getElementById('refreshItemsBtn').addEventListener('click', () => this.loadMappedItems());
        document.getElementById('clearMemoryBtn').addEventListener('click', () => this.clearAIMemory());
        document.getElementById('exportDataBtn').addEventListener('click', () => this.exportData());
        document.getElementById('importDataBtn').addEventListener('click', () => this.importData());
        
        // Screen click handler for mapping
        document.getElementById('gameScreenCapture').addEventListener('click', (e) => this.handleScreenClick(e));
    }

    async detectGameWindow() {
        this.updateStatus('Detecting game window...', 'info');
        
        try {
            const response = await fetch('/api/screen/detect-window', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success && data.window) {
                this.gameWindow = data.window;
                this.updateDetectionStatus(true, data.window);
                this.updateStatus('Game window detected successfully!', 'success');
                this.startScreenCapture();
            } else {
                this.updateDetectionStatus(false);
                this.updateStatus('No game window found. Make sure the game is running.', 'warning');
            }
        } catch (error) {
            this.updateStatus('Error detecting game window: ' + error.message, 'danger');
            this.updateDetectionStatus(false);
        }
    }

    updateDetectionStatus(found, window = null) {
        const statusEl = document.getElementById('detectionStatus');
        const textEl = document.getElementById('detectionText');
        const windowSizeEl = document.getElementById('windowSize');
        const windowPositionEl = document.getElementById('windowPosition');
        
        if (found && window) {
            statusEl.className = 'alert alert-success';
            textEl.innerHTML = `<i class="fas fa-check-circle me-2"></i>Game window found: ${window.title}`;
            windowSizeEl.textContent = `${window.width}x${window.height}`;
            windowPositionEl.textContent = `(${window.x}, ${window.y})`;
        } else {
            statusEl.className = 'alert alert-danger';
            textEl.innerHTML = `<i class="fas fa-times-circle me-2"></i>Game window not found`;
            windowSizeEl.textContent = 'Unknown';
            windowPositionEl.textContent = 'Unknown';
        }
    }

    selectItemType(itemType) {
        this.currentItemType = itemType;
        const startBtn = document.getElementById('startMappingBtn');
        
        if (itemType && this.gameWindow) {
            startBtn.disabled = false;
            startBtn.classList.remove('disabled');
        } else {
            startBtn.disabled = true;
            startBtn.classList.add('disabled');
        }
        
        if (itemType) {
            this.showMappingInstructions(itemType);
        }
    }

    showMappingInstructions(itemType) {
        const instructionsEl = document.getElementById('mappingInstructions');
        const textEl = document.getElementById('instructionText');
        
        const instructions = {
            chest: 'Click on any chest in the game to map it for automatic opening',
            egg: 'Click on eggs that you want the AI to automatically hatch',
            breakable: 'Click on breakable objects like crystals or rocks',
            path: 'Click to define waypoints for movement paths',
            border: 'Click around the game area to set boundaries',
            inventory: 'Click on the inventory icon to map it',
            auto_hatch: 'Click on the auto hatch button/icon',
            pet: 'Click on pet icons to map different pet types',
            coin: 'Click on coins or currency displays',
            button: 'Click on any game button you want to map',
            menu: 'Click on menu items or interface elements',
            custom: 'Click on any custom item you want to map'
        };
        
        textEl.textContent = instructions[itemType] || 'Click on the item to map it';
        instructionsEl.style.display = 'block';
    }

    startMapping() {
        this.mappingMode = true;
        this.updateMappingUI(true);
        this.updateStatus(`Mapping mode active for ${this.currentItemType}. Click on the game screen.`, 'info');
        
        // Show mapping overlay
        document.getElementById('mappingOverlay').style.display = 'block';
    }

    cancelMapping() {
        this.mappingMode = false;
        this.updateMappingUI(false);
        this.updateStatus('Mapping cancelled', 'info');
        
        // Hide mapping overlay
        document.getElementById('mappingOverlay').style.display = 'none';
    }

    updateMappingUI(mapping) {
        const startBtn = document.getElementById('startMappingBtn');
        const saveBtn = document.getElementById('saveItemBtn');
        const cancelBtn = document.getElementById('cancelMappingBtn');
        const itemSelect = document.getElementById('itemTypeSelect');
        
        if (mapping) {
            startBtn.disabled = true;
            startBtn.textContent = 'Click on screen to map';
            saveBtn.disabled = false;
            cancelBtn.disabled = false;
            itemSelect.disabled = true;
        } else {
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-crosshairs me-2"></i>Start Mapping';
            saveBtn.disabled = true;
            cancelBtn.disabled = true;
            itemSelect.disabled = false;
        }
    }

    async handleScreenClick(event) {
        if (!this.mappingMode) return;
        
        const rect = event.target.getBoundingClientRect();
        const x = Math.round((event.clientX - rect.left) * (event.target.naturalWidth / rect.width));
        const y = Math.round((event.clientY - rect.top) * (event.target.naturalHeight / rect.height));
        
        this.updateStatus(`Mapping ${this.currentItemType} at position (${x}, ${y})`, 'info');
        
        // Capture screenshot area around the click
        try {
            const response = await fetch('/api/screen/capture-area', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    x: x - 25,
                    y: y - 25,
                    width: 50,
                    height: 50,
                    item_type: this.currentItemType
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentMappingData = {
                    type: this.currentItemType,
                    x: x,
                    y: y,
                    screenshot: data.screenshot,
                    features: data.features
                };
                
                this.updateStatus('Item captured! You can now save it or cancel.', 'success');
            }
        } catch (error) {
            this.updateStatus('Error capturing item: ' + error.message, 'danger');
        }
    }

    async saveItem() {
        if (!this.currentMappingData) {
            this.updateStatus('No item data to save', 'warning');
            return;
        }
        
        const itemName = document.getElementById('itemNameInput').value.trim() || 
                        `${this.currentItemType}_${Date.now()}`;
        
        const itemData = {
            ...this.currentMappingData,
            name: itemName,
            created_at: new Date().toISOString()
        };
        
        try {
            const response = await fetch('/api/items/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(itemData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateStatus(`Item '${itemName}' saved successfully!`, 'success');
                this.mappedItems.push(itemData);
                this.updateMappedItemsList();
                this.cancelMapping();
                document.getElementById('itemNameInput').value = '';
            } else {
                this.updateStatus('Error saving item: ' + (data.error || 'Unknown error'), 'danger');
            }
        } catch (error) {
            this.updateStatus('Error saving item: ' + error.message, 'danger');
        }
    }

    async startAutoPlay() {
        this.isAutoPlaying = true;
        this.updateLearningStatus();
        this.updateStatus('Auto play started - AI is now playing the game', 'success');
        
        try {
            const response = await fetch('/api/ai/start-autoplay', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            this.logAIAction('Auto Play', data.message || 'Started successfully');
        } catch (error) {
            this.updateStatus('Error starting auto play: ' + error.message, 'danger');
            this.isAutoPlaying = false;
            this.updateLearningStatus();
        }
    }

    async stopAutoPlay() {
        this.isAutoPlaying = false;
        this.updateLearningStatus();
        this.updateStatus('Auto play stopped', 'info');
        
        try {
            await fetch('/api/ai/stop-autoplay', { method: 'POST' });
            this.logAIAction('Auto Play', 'Stopped by user');
        } catch (error) {
            console.error('Error stopping auto play:', error);
        }
    }

    async startLearning() {
        this.isLearning = true;
        this.updateLearningStatus();
        this.updateStatus('AI learning mode activated - watching your gameplay', 'info');
        
        try {
            const response = await fetch('/api/ai/start-learning', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            this.logAIAction('Learning Mode', data.message || 'Started successfully');
        } catch (error) {
            this.updateStatus('Error starting learning: ' + error.message, 'danger');
            this.isLearning = false;
            this.updateLearningStatus();
        }
    }

    async stopLearning() {
        this.isLearning = false;
        this.updateLearningStatus();
        this.updateStatus('Learning mode stopped', 'info');
        
        try {
            await fetch('/api/ai/stop-learning', { method: 'POST' });
            this.logAIAction('Learning Mode', 'Stopped by user');
        } catch (error) {
            console.error('Error stopping learning:', error);
        }
    }

    async startRecording() {
        this.isRecording = true;
        this.updateLearningStatus();
        this.updateStatus('Macro recording started - perform actions to record', 'info');
        
        try {
            const response = await fetch('/api/macros/start-recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            this.logAIAction('Macro Recording', data.message || 'Started successfully');
        } catch (error) {
            this.updateStatus('Error starting recording: ' + error.message, 'danger');
            this.isRecording = false;
            this.updateLearningStatus();
        }
    }

    async stopRecording() {
        this.isRecording = false;
        this.updateLearningStatus();
        
        try {
            const response = await fetch('/api/macros/stop-recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateStatus(`Macro recording saved: ${data.macro_name}`, 'success');
                this.logAIAction('Macro Recording', `Saved as: ${data.macro_name}`);
            }
        } catch (error) {
            this.updateStatus('Error stopping recording: ' + error.message, 'danger');
        }
    }

    async startWatchAndCreate() {
        this.updateStatus('AI Watch & Create mode activated', 'info');
        
        try {
            const response = await fetch('/api/ai/start-watch-create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            this.logAIAction('Watch & Create', data.message || 'Started watching user actions');
        } catch (error) {
            this.updateStatus('Error starting watch & create: ' + error.message, 'danger');
        }
    }

    async captureScreen() {
        try {
            const response = await fetch('/api/screen/capture');
            const data = await response.json();
            
            if (data.success && data.screenshot) {
                document.getElementById('gameScreenCapture').src = `data:image/png;base64,${data.screenshot}`;
                document.getElementById('noGameScreen').style.display = 'none';
            }
        } catch (error) {
            console.error('Error capturing screen:', error);
        }
    }

    toggleLiveCapture(enabled) {
        if (enabled) {
            this.screenCapture = setInterval(() => this.captureScreen(), this.updateInterval);
        } else {
            if (this.screenCapture) {
                clearInterval(this.screenCapture);
                this.screenCapture = null;
            }
        }
    }

    startScreenCapture() {
        if (document.getElementById('liveCapture').checked) {
            this.toggleLiveCapture(true);
        }
        this.captureScreen();
    }

    async loadMappedItems() {
        try {
            const response = await fetch('/api/items/list');
            const data = await response.json();
            
            if (data.success) {
                this.mappedItems = data.items || [];
                this.updateMappedItemsList();
                this.updateItemsOverlay();
            }
        } catch (error) {
            console.error('Error loading mapped items:', error);
        }
    }

    updateMappedItemsList() {
        const listEl = document.getElementById('mappedItemsList');
        
        if (this.mappedItems.length === 0) {
            listEl.innerHTML = '<div class="text-muted text-center py-3">No items mapped yet</div>';
            return;
        }
        
        listEl.innerHTML = this.mappedItems.map(item => `
            <div class="list-group-item d-flex justify-content-between align-items-center">
                <div>
                    <strong>${item.name}</strong>
                    <small class="text-muted d-block">${item.type} at (${item.x}, ${item.y})</small>
                </div>
                <button class="btn btn-outline-danger btn-sm" onclick="itemMapper.deleteItem('${item.name}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
        
        document.getElementById('itemCount').textContent = this.mappedItems.length;
    }

    updateItemsOverlay() {
        const overlayEl = document.getElementById('itemsOverlay');
        
        overlayEl.innerHTML = this.mappedItems.map(item => `
            <div class="position-absolute" style="left: ${item.x-10}px; top: ${item.y-10}px;">
                <div class="badge bg-primary" style="font-size: 8px;">
                    ${item.type}
                </div>
            </div>
        `).join('');
    }

    async deleteItem(itemName) {
        try {
            const response = await fetch('/api/items/delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name: itemName })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateStatus(`Item '${itemName}' deleted`, 'success');
                this.loadMappedItems();
            }
        } catch (error) {
            this.updateStatus('Error deleting item: ' + error.message, 'danger');
        }
    }

    updateLearningStatus() {
        const learningEl = document.getElementById('learningStatus');
        const autoPlayEl = document.getElementById('autoPlayStatus');
        
        if (this.isLearning) {
            learningEl.className = 'badge bg-success';
            learningEl.textContent = 'Learning';
        } else {
            learningEl.className = 'badge bg-secondary';
            learningEl.textContent = 'Idle';
        }
        
        if (this.isAutoPlaying) {
            autoPlayEl.className = 'badge bg-success';
            autoPlayEl.textContent = 'Running';
        } else {
            autoPlayEl.className = 'badge bg-secondary';
            autoPlayEl.textContent = 'Stopped';
        }
    }

    logAIAction(action, message) {
        const logEl = document.getElementById('aiActionsList');
        const timestamp = new Date().toLocaleTimeString();
        
        const logEntry = document.createElement('div');
        logEntry.innerHTML = `<span class="text-info">[${timestamp}]</span> <span class="text-warning">${action}:</span> ${message}`;
        
        logEl.insertBefore(logEntry, logEl.firstChild);
        
        // Keep only last 100 entries
        while (logEl.children.length > 100) {
            logEl.removeChild(logEl.lastChild);
        }
    }

    startStatusUpdates() {
        this.updateAIStats();
        setInterval(() => {
            this.updateAIStats();
        }, 5000);
    }

    async updateAIStats() {
        try {
            const response = await fetch('/api/ai/stats');
            const data = await response.json();
            
            if (data.success) {
                document.getElementById('macroCount').textContent = data.macro_count || 0;
                document.getElementById('learningHours').textContent = data.learning_hours || 0;
                document.getElementById('accuracyRate').textContent = (data.accuracy || 0) + '%';
            }
        } catch (error) {
            console.error('Error updating AI stats:', error);
        }
    }

    async clearAIMemory() {
        if (confirm('Are you sure you want to clear all AI memory? This cannot be undone.')) {
            try {
                const response = await fetch('/api/ai/clear-memory', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const data = await response.json();
                
                if (data.success) {
                    this.updateStatus('AI memory cleared successfully', 'success');
                    this.loadMappedItems();
                    this.logAIAction('System', 'Memory cleared by user');
                }
            } catch (error) {
                this.updateStatus('Error clearing memory: ' + error.message, 'danger');
            }
        }
    }

    exportData() {
        // TODO: Implement data export
        this.updateStatus('Data export - Coming soon!', 'info');
    }

    importData() {
        // TODO: Implement data import
        this.updateStatus('Data import - Coming soon!', 'info');
    }

    updateStatus(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize ItemMapper when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.itemMapper = new ItemMapper();
});