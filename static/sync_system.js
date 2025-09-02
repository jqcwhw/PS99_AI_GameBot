/**
 * Natro Synchronization System JavaScript Interface
 * 
 * Provides UI controls for multi-instance coordination and shadow following
 */

let syncSystemEnabled = false;
let shadowFollowingActive = false;

// Toggle sync system on/off
function toggleSyncSystem() {
    fetch('/api/sync/toggle', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            enable: !syncSystemEnabled
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            syncSystemEnabled = data.enabled;
            updateSyncSystemUI();
            showNotification(
                data.enabled ? 'Sync system enabled - accounts can now follow each other like shadows' : 'Sync system disabled',
                data.enabled ? 'success' : 'info'
            );
            
            // Show/hide sync-related UI components
            const shadowCard = document.getElementById('shadowFollowingCard');
            const commandsCard = document.getElementById('coordinatedCommandsCard');
            
            if (shadowCard && commandsCard) {
                shadowCard.style.display = data.enabled ? 'block' : 'none';
                commandsCard.style.display = data.enabled ? 'block' : 'none';
            }
            
            if (data.enabled) {
                loadAvailableAccounts();
            }
        } else {
            showNotification('Failed to toggle sync system: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error toggling sync system:', error);
        showNotification('Error toggling sync system', 'error');
    });
}

// Update sync system UI elements
function updateSyncSystemUI() {
    const toggleButton = document.getElementById('syncToggleText');
    if (toggleButton) {
        toggleButton.textContent = syncSystemEnabled ? 'Disable Sync' : 'Enable Sync';
    }
    
    const syncButton = toggleButton?.parentElement;
    if (syncButton) {
        syncButton.className = syncSystemEnabled ? 
            'btn btn-success w-100' : 'btn btn-primary w-100';
    }
}

// Load available accounts for shadow following setup
function loadAvailableAccounts() {
    fetch('/api/accounts/list')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const leaderSelect = document.getElementById('leaderAccount');
            const followerSelect = document.getElementById('followerAccounts');
            
            if (leaderSelect && followerSelect) {
                // Clear existing options (except first placeholder)
                leaderSelect.innerHTML = '<option value="">Select leader account...</option>';
                followerSelect.innerHTML = '<option value="">Select follower accounts...</option>';
                
                // Add account options
                data.accounts.forEach(account => {
                    const leaderOption = document.createElement('option');
                    leaderOption.value = account.username;
                    leaderOption.textContent = `${account.display_name} (${account.username})`;
                    leaderSelect.appendChild(leaderOption);
                    
                    const followerOption = document.createElement('option');
                    followerOption.value = account.username;
                    followerOption.textContent = `${account.display_name} (${account.username})`;
                    followerSelect.appendChild(followerOption);
                });
            }
        }
    })
    .catch(error => {
        console.error('Error loading accounts:', error);
    });
}

// Setup shadow following relationships
function setupShadowFollowing() {
    const leaderAccount = document.getElementById('leaderAccount').value;
    const followerAccounts = Array.from(document.getElementById('followerAccounts').selectedOptions)
        .map(option => option.value)
        .filter(value => value !== '');
    
    if (!leaderAccount) {
        showNotification('Please select a leader account', 'warning');
        return;
    }
    
    if (followerAccounts.length === 0) {
        showNotification('Please select at least one follower account', 'warning');
        return;
    }
    
    if (followerAccounts.includes(leaderAccount)) {
        showNotification('Leader account cannot be a follower of itself', 'warning');
        return;
    }
    
    fetch('/api/sync/setup_shadow_following', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            leader_account: leaderAccount,
            follower_accounts: followerAccounts
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            shadowFollowingActive = true;
            showNotification(
                `Shadow following configured: ${followerAccounts.length} accounts will follow ${leaderAccount}`,
                'success'
            );
            showSyncStatus();
        } else {
            showNotification('Failed to setup shadow following: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error setting up shadow following:', error);
        showNotification('Error setting up shadow following', 'error');
    });
}

// Send coordinated command to all synchronized accounts
function sendCoordinatedCommand(commandType, parameters = {}) {
    if (!syncSystemEnabled) {
        showNotification('Sync system must be enabled first', 'warning');
        return;
    }
    
    fetch('/api/sync/send_command', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            command_type: commandType,
            parameters: parameters
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification(`Coordinated command sent: ${commandType}`, 'success');
        } else {
            showNotification('Failed to send coordinated command: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error sending coordinated command:', error);
        showNotification('Error sending coordinated command', 'error');
    });
}

// Show current synchronization status
function showSyncStatus() {
    fetch('/api/sync/status')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const status = data.status;
            let statusHtml = `
                <div class="alert alert-info">
                    <h5><i class="fas fa-info-circle"></i> Synchronization Status</h5>
                    <p><strong>Total Accounts:</strong> ${status.total_accounts}</p>
                    <p><strong>Shadow Mode:</strong> ${status.shadow_mode ? 'Enabled' : 'Disabled'}</p>
                    <p><strong>Command Queue Size:</strong> ${status.command_queue_size}</p>
                    
                    ${status.active_accounts.length > 0 ? `
                        <h6>Active Accounts:</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Account</th>
                                        <th>Instance</th>
                                        <th>State</th>
                                        <th>Field</th>
                                        <th>Task</th>
                                        <th>Role</th>
                                        <th>Last Seen</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${status.active_accounts.map(account => `
                                        <tr>
                                            <td>${account.account_id}</td>
                                            <td>${account.instance_id}</td>
                                            <td><span class="badge bg-secondary">${account.state}</span></td>
                                            <td>${account.field || '-'}</td>
                                            <td>${account.task || '-'}</td>
                                            <td>
                                                ${account.is_leader ? 
                                                    '<span class="badge bg-warning">Leader</span>' : 
                                                    '<span class="badge bg-info">Follower</span>'
                                                }
                                                ${account.follow_target ? 
                                                    `<br><small>Following: ${account.follow_target}</small>` : 
                                                    ''
                                                }
                                            </td>
                                            <td>${account.last_seen}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    ` : '<p>No active accounts</p>'}
                </div>
            `;
            
            // Show in modal or update a status div
            showModal('Synchronization Status', statusHtml);
        }
    })
    .catch(error => {
        console.error('Error loading sync status:', error);
        showNotification('Error loading sync status', 'error');
    });
}

// Emergency stop all synchronized accounts
function emergencyStopAll() {
    if (!confirm('Are you sure you want to send emergency stop to all synchronized accounts?')) {
        return;
    }
    
    sendCoordinatedCommand('emergency_stop', {
        reason: 'Emergency stop triggered by user'
    });
}

// Initialize sync system status on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check if sync system is currently enabled
    fetch('/api/sync/status')
    .then(response => response.json())
    .then(data => {
        if (data.success && data.status.total_accounts > 0) {
            syncSystemEnabled = true;
            updateSyncSystemUI();
            
            // Show sync UI components if system is active
            const shadowCard = document.getElementById('shadowFollowingCard');
            const commandsCard = document.getElementById('coordinatedCommandsCard');
            
            if (shadowCard && commandsCard) {
                shadowCard.style.display = 'block';
                commandsCard.style.display = 'block';
            }
            
            loadAvailableAccounts();
        }
    })
    .catch(error => {
        console.log('Sync system not active');
    });
});