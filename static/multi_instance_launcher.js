// Multi-Instance Launcher JavaScript
class MultiInstanceLauncher {
    constructor() {
        this.instances = new Map();
        this.mutexActive = false;
        this.init();
    }

    init() {
        this.loadInstanceStatus();
        this.loadMutexStatus();
        this.loadSyncStatus();
        this.startMonitoring();
    }

    async loadInstanceStatus() {
        try {
            const response = await fetch('/api/instances/list');
            const data = await response.json();
            if (data.success) {
                this.updateInstanceDisplay(data.instances || []);
            }
        } catch (error) {
            console.error('Failed to load instance status:', error);
        }
    }

    async loadMutexStatus() {
        try {
            const response = await fetch('/api/mutex/status');
            const data = await response.json();
            if (data.success) {
                this.updateMutexStatus(data);
            }
        } catch (error) {
            console.error('Failed to load mutex status:', error);
        }
    }

    async loadSyncStatus() {
        try {
            const response = await fetch('/api/sync/status');
            const data = await response.json();
            this.updateSyncStatus(data);
        } catch (error) {
            console.error('Failed to load sync status:', error);
        }
    }

    updateInstanceDisplay(instances) {
        const instanceList = document.getElementById('instanceList');
        
        if (instances.length === 0) {
            instanceList.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="fas fa-info-circle fa-2x mb-2"></i>
                    <p>No instances running<br>Launch an instance to begin monitoring</p>
                </div>
            `;
            document.getElementById('instanceCount').textContent = '0';
            return;
        }

        let html = '';
        instances.forEach((instance, index) => {
            const statusClass = instance.status === 'active' ? 'status-active' : 
                               instance.status === 'pending' ? 'status-pending' : 'status-inactive';
            
            html += `
                <div class="d-flex justify-content-between align-items-center p-2 border-bottom border-secondary">
                    <div>
                        <span class="status-indicator ${statusClass}"></span>
                        <strong>${instance.name || `Instance_${index + 1}`}</strong>
                        <br>
                        <small class="text-muted">
                            PID: ${instance.pid || 'N/A'} | 
                            Method: ${instance.method || 'Standard'} | 
                            ${instance.sync_enabled ? 'Sync: ON' : 'Sync: OFF'}
                        </small>
                    </div>
                    <div>
                        <button class="btn btn-sm btn-outline-warning me-1" onclick="launcher.pauseInstance('${instance.name}')">
                            <i class="fas fa-pause"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="launcher.killInstance('${instance.name}')">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            `;
        });

        instanceList.innerHTML = html;
        document.getElementById('instanceCount').textContent = instances.length.toString();
        this.instances.clear();
        instances.forEach(instance => this.instances.set(instance.name, instance));
    }

    updateMutexStatus(data) {
        const statusElement = document.getElementById('mutex-status');
        const btnElement = document.getElementById('mutex-btn-text');
        
        this.mutexActive = data.bypass_active || false;
        
        if (this.mutexActive) {
            statusElement.textContent = 'Active';
            statusElement.className = 'badge bg-success';
            btnElement.textContent = 'Disable Mutex Bypass';
        } else {
            statusElement.textContent = 'Inactive';
            statusElement.className = 'badge bg-danger';
            btnElement.textContent = 'Activate Mutex Bypass';
        }
    }

    updateSyncStatus(data) {
        document.getElementById('syncStatus').textContent = data.sync_active ? 'Active' : 'Inactive';
        document.getElementById('syncStatus').className = data.sync_active ? 'badge bg-success' : 'badge bg-secondary';
        
        document.getElementById('fieldFollowStatus').textContent = data.field_following_enabled ? 'Enabled' : 'Disabled';
        document.getElementById('fieldFollowStatus').className = data.field_following_enabled ? 'badge bg-success' : 'badge bg-secondary';
        
        document.getElementById('shadowModeStatus').textContent = data.shadow_mode ? 'Active' : 'Inactive';
        document.getElementById('shadowModeStatus').className = data.shadow_mode ? 'badge bg-info' : 'badge bg-secondary';
    }

    async toggleMutexBypass() {
        try {
            const response = await fetch('/api/mutex/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            if (data.success) {
                this.updateMutexStatus(data);
                this.showAlert('success', `Mutex bypass ${data.active ? 'activated' : 'deactivated'} successfully`);
            } else {
                this.showAlert('danger', `Failed to toggle mutex bypass: ${data.message || data.error}`);
            }
        } catch (error) {
            this.showAlert('danger', 'Network error while toggling mutex bypass');
            console.error('Mutex toggle error:', error);
        }
    }

    async launchInstance() {
        const instanceName = document.getElementById('instanceName').value || `Instance_${Date.now()}`;
        const launchMethod = document.getElementById('launchMethod').value;
        const serverRegion = document.getElementById('serverRegion').value;
        const enableSync = document.getElementById('enableSync').checked;
        const enableAI = document.getElementById('enableAI').checked;
        const autoLogin = document.getElementById('autoLogin').checked;
        const minimizeStart = document.getElementById('minimizeStart').checked;

        try {
            const response = await fetch('/api/instances/launch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    account_name: instanceName,
                    launch_method: launchMethod,
                    server_region: serverRegion,
                    enable_sync: enableSync,
                    enable_ai: enableAI,
                    auto_login: autoLogin,
                    minimize_start: minimizeStart
                })
            });

            const data = await response.json();
            if (data.success) {
                this.showAlert('success', `Instance "${instanceName}" launched successfully`);
                // Clear the instance name for next launch
                document.getElementById('instanceName').value = '';
                // Refresh the instance list
                setTimeout(() => this.loadInstanceStatus(), 1000);
            } else {
                this.showAlert('danger', `Failed to launch instance: ${data.message}`);
            }
        } catch (error) {
            this.showAlert('danger', 'Network error while launching instance');
            console.error('Launch error:', error);
        }
    }

    async launchBatch() {
        const baseInstanceName = document.getElementById('instanceName').value || 'BatchInstance';
        let successCount = 0;
        
        for (let i = 1; i <= 5; i++) {
            try {
                const instanceName = `${baseInstanceName}_${i}`;
                const response = await fetch('/api/instances/launch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        account_name: instanceName,
                        launch_method: document.getElementById('launchMethod').value,
                        server_region: document.getElementById('serverRegion').value,
                        enable_sync: document.getElementById('enableSync').checked,
                        enable_ai: document.getElementById('enableAI').checked,
                        auto_login: document.getElementById('autoLogin').checked,
                        minimize_start: document.getElementById('minimizeStart').checked
                    })
                });

                const data = await response.json();
                if (data.success) {
                    successCount++;
                }
                
                // Small delay between launches
                await new Promise(resolve => setTimeout(resolve, 500));
            } catch (error) {
                console.error(`Failed to launch instance ${i}:`, error);
            }
        }

        this.showAlert('info', `Batch launch completed: ${successCount}/5 instances launched successfully`);
        setTimeout(() => this.loadInstanceStatus(), 2000);
    }

    async killInstance(instanceName) {
        try {
            const response = await fetch('/api/instances/kill', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ account_name: instanceName })
            });

            const data = await response.json();
            if (data.success) {
                this.showAlert('success', `Instance "${instanceName}" terminated`);
                this.loadInstanceStatus();
            } else {
                this.showAlert('danger', `Failed to terminate instance: ${data.message}`);
            }
        } catch (error) {
            this.showAlert('danger', 'Network error while terminating instance');
            console.error('Kill instance error:', error);
        }
    }

    async pauseInstance(instanceName) {
        try {
            const response = await fetch('/api/instances/pause', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ account_name: instanceName })
            });

            const data = await response.json();
            if (data.success) {
                this.showAlert('info', `Instance "${instanceName}" paused`);
                this.loadInstanceStatus();
            } else {
                this.showAlert('warning', `Failed to pause instance: ${data.message}`);
            }
        } catch (error) {
            this.showAlert('danger', 'Network error while pausing instance');
            console.error('Pause instance error:', error);
        }
    }

    async killAllInstances() {
        if (!confirm('Are you sure you want to terminate ALL running instances?')) {
            return;
        }

        try {
            const response = await fetch('/api/instances/kill-all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (data.success) {
                this.showAlert('success', 'All instances terminated successfully');
                this.loadInstanceStatus();
            } else {
                this.showAlert('danger', `Failed to terminate instances: ${data.message}`);
            }
        } catch (error) {
            this.showAlert('danger', 'Network error while terminating instances');
            console.error('Kill all error:', error);
        }
    }

    async pauseAllInstances() {
        try {
            const response = await fetch('/api/instances/pause-all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (data.success) {
                this.showAlert('info', 'All instances paused');
                this.loadInstanceStatus();
            } else {
                this.showAlert('warning', `Failed to pause instances: ${data.message}`);
            }
        } catch (error) {
            this.showAlert('danger', 'Network error while pausing instances');
            console.error('Pause all error:', error);
        }
    }

    async resumeAllInstances() {
        try {
            const response = await fetch('/api/instances/resume-all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (data.success) {
                this.showAlert('success', 'All instances resumed');
                this.loadInstanceStatus();
            } else {
                this.showAlert('warning', `Failed to resume instances: ${data.message}`);
            }
        } catch (error) {
            this.showAlert('danger', 'Network error while resuming instances');
            console.error('Resume all error:', error);
        }
    }

    async syncAllInstances() {
        try {
            const response = await fetch('/api/sync/force-sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (data.success) {
                this.showAlert('info', 'Synchronization forced for all instances');
                this.loadSyncStatus();
            } else {
                this.showAlert('warning', `Failed to force sync: ${data.message}`);
            }
        } catch (error) {
            this.showAlert('danger', 'Network error while forcing sync');
            console.error('Force sync error:', error);
        }
    }

    showAlert(type, message) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    startMonitoring() {
        // Update system stats and instance list every 5 seconds
        setInterval(() => {
            this.loadInstanceStatus();
            this.loadSyncStatus();
            this.updateSystemStats();
        }, 5000);
    }

    updateSystemStats() {
        // Simulate system stats (in real implementation, these would come from the backend)
        const instanceCount = this.instances.size;
        const memoryUsage = Math.min(20 + (instanceCount * 8), 95);
        const cpuUsage = Math.min(10 + (instanceCount * 3), 85);

        document.getElementById('memoryUsage').textContent = `${memoryUsage}%`;
        document.getElementById('cpuUsage').textContent = `${cpuUsage}%`;
    }
}

// Global functions for onclick handlers
let launcher;

function toggleMutexBypass() {
    launcher.toggleMutexBypass();
}

function launchInstance() {
    launcher.launchInstance();
}

function launchBatch() {
    launcher.launchBatch();
}

function killAllInstances() {
    launcher.killAllInstances();
}

function pauseAllInstances() {
    launcher.pauseAllInstances();
}

function resumeAllInstances() {
    launcher.resumeAllInstances();
}

function syncAllInstances() {
    launcher.syncAllInstances();
}

function refreshStatus() {
    launcher.loadInstanceStatus();
    launcher.loadMutexStatus();
    launcher.loadSyncStatus();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    launcher = new MultiInstanceLauncher();
});

// Global variables for tracking active methods
let activeMutexMethods = new Set();

// Mutex bypass control with multiple methods
function activateMutexMethod(methodId) {
    const methodConfig = {
        stealth_mode: document.getElementById('stealthMode').checked,
        anti_detection: document.getElementById('antiDetection').checked,
        process_randomization: document.getElementById('processRandom').checked
    };

    fetch('/api/mutex/activate-method', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            method_id: methodId,
            ...methodConfig
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            activeMutexMethods.add(methodId);
            updateMethodButton(methodId, true);
            const deactivateBtn = document.getElementById('deactivate-all');
            if (deactivateBtn) deactivateBtn.disabled = false;
            
            const methodNames = {
                1: 'Standard Mutex Capture',
                2: 'Hidden Window Bypass',
                3: 'Event-Based Bypass',
                4: 'Process Hook',
                5: 'Registry Override'
            };
            
            launcher.showAlert('success', `${methodNames[methodId]} activated successfully`);
            
            // Update mutex status
            if (activeMutexMethods.size === 1) {
                const statusEl = document.getElementById('mutex-status');
                if (statusEl) {
                    statusEl.textContent = 'Active';
                    statusEl.className = 'badge bg-success';
                }
            }
        } else {
            launcher.showAlert('danger', `Failed to activate method ${methodId}: ` + (data.error || data.message));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        launcher.showAlert('danger', `Network error while activating method ${methodId}`);
    });
}

function deactivateMutexMethod(methodId) {
    fetch('/api/mutex/deactivate-method', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            method_id: methodId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            activeMutexMethods.delete(methodId);
            updateMethodButton(methodId, false);
            
            if (activeMutexMethods.size === 0) {
                const statusEl = document.getElementById('mutex-status');
                if (statusEl) {
                    statusEl.textContent = 'Ready';
                    statusEl.className = 'badge bg-secondary';
                }
                const deactivateBtn = document.getElementById('deactivate-all');
                if (deactivateBtn) deactivateBtn.disabled = true;
            }
            
            const methodNames = {
                1: 'Standard Mutex Capture',
                2: 'Hidden Window Bypass',
                3: 'Event-Based Bypass',
                4: 'Process Hook',
                5: 'Registry Override'
            };
            
            launcher.showAlert('success', `${methodNames[methodId]} deactivated successfully`);
        } else {
            launcher.showAlert('danger', `Failed to deactivate method ${methodId}: ` + (data.error || data.message));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        launcher.showAlert('danger', `Network error while deactivating method ${methodId}`);
    });
}

function deactivateAllMethods() {
    fetch('/api/mutex/deactivate-all', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Reset all method buttons
            for (let i = 1; i <= 5; i++) {
                updateMethodButton(i, false);
            }
            
            activeMutexMethods.clear();
            const statusEl = document.getElementById('mutex-status');
            if (statusEl) {
                statusEl.textContent = 'Ready';
                statusEl.className = 'badge bg-secondary';
            }
            const deactivateBtn = document.getElementById('deactivate-all');
            if (deactivateBtn) deactivateBtn.disabled = true;
            
            launcher.showAlert('success', 'All mutex bypass methods deactivated');
        } else {
            launcher.showAlert('danger', 'Failed to deactivate all methods: ' + (data.error || data.message));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        launcher.showAlert('danger', 'Network error while deactivating all methods');
    });
}

function updateMethodButton(methodId, isActive) {
    const button = document.getElementById(`method-${methodId}`);
    if (!button) return;
    
    if (isActive) {
        button.classList.remove('btn-primary', 'btn-success', 'btn-warning', 'btn-info', 'btn-secondary');
        button.classList.add('btn-danger');
        button.onclick = () => deactivateMutexMethod(methodId);
        
        // Update button text to show it's active
        const originalText = button.textContent.replace('Stop - ', '');
        button.innerHTML = `<i class="fas fa-stop me-2"></i>Stop - ${originalText}`;
    } else {
        button.classList.remove('btn-danger');
        
        // Restore original button classes and functionality
        const methodClasses = {
            1: 'btn-primary',
            2: 'btn-success', 
            3: 'btn-warning',
            4: 'btn-info',
            5: 'btn-secondary'
        };
        
        button.classList.add(methodClasses[methodId]);
        button.onclick = () => activateMutexMethod(methodId);
        
        // Restore original button text
        const methodTexts = {
            1: '<i class="fas fa-play me-2"></i>Method 1: Standard Mutex Capture',
            2: '<i class="fas fa-eye-slash me-2"></i>Method 2: Hidden Window Bypass',
            3: '<i class="fas fa-calendar-alt me-2"></i>Method 3: Event-Based Bypass',
            4: '<i class="fas fa-link me-2"></i>Method 4: Process Hook',
            5: '<i class="fas fa-cogs me-2"></i>Method 5: Registry Override'
        };
        
        button.innerHTML = methodTexts[methodId];
    }
}