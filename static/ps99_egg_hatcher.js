/**
 * PS99 Egg Hatcher Interface
 */

class PS99EggHatcher {
    constructor() {
        this.isHatching = false;
        this.targetPets = new Set();
        this.eggs = [];
        this.pets = [];
        this.stats = {
            totalHatches: 0,
            petsDetected: 0,
            targetPetsFound: 0,
            successRate: 0
        };
        
        this.initialize();
    }
    
    initialize() {
        this.setupEventListeners();
        this.loadPS99Data();
        this.startStatusUpdates();
    }
    
    setupEventListeners() {
        // Game Controls
        document.getElementById('findGameBtn').addEventListener('click', () => this.findGame());
        document.getElementById('refreshDataBtn').addEventListener('click', () => this.loadPS99Data());
        
        // Hatching Controls
        document.getElementById('hatchOnceBtn').addEventListener('click', () => this.hatchOnce());
        document.getElementById('startAutoHatchBtn').addEventListener('click', () => this.startAutoHatch());
        document.getElementById('stopAutoHatchBtn').addEventListener('click', () => this.stopAutoHatch());
        
        // Target Pet Management
        document.getElementById('addTargetBtn').addEventListener('click', () => this.addTargetPet());
        document.getElementById('targetPetInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addTargetPet();
        });
        
        // Log Management
        document.getElementById('clearLogBtn').addEventListener('click', () => this.clearLog());
    }
    
    async loadPS99Data() {
        this.showStatus('Loading PS99 data...', 'info');
        
        try {
            // Load eggs data
            const eggsResponse = await fetch('/api/ps99/eggs');
            if (eggsResponse.ok) {
                const eggsData = await eggsResponse.json();
                this.eggs = eggsData.eggs || [];
                this.populateEggSelect();
            }
            
            // Load pets data
            const petsResponse = await fetch('/api/ps99/pets');
            if (petsResponse.ok) {
                const petsData = await petsResponse.json();
                this.pets = petsData.pets || [];
            }
            
            this.showStatus('PS99 data loaded successfully', 'success');
        } catch (error) {
            console.error('Error loading PS99 data:', error);
            this.showStatus('Failed to load PS99 data', 'danger');
        }
    }
    
    populateEggSelect() {
        const select = document.getElementById('eggSelect');
        select.innerHTML = '<option value="">Any Available Egg</option>';
        
        // Group eggs by category
        const eggsByCategory = {};
        this.eggs.forEach(egg => {
            const category = egg.category || 'Other';
            if (!eggsByCategory[category]) {
                eggsByCategory[category] = [];
            }
            eggsByCategory[category].push(egg);
        });
        
        // Add eggs to select
        Object.entries(eggsByCategory).forEach(([category, eggs]) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = category;
            
            eggs.forEach(egg => {
                const option = document.createElement('option');
                option.value = egg.configName;
                option.textContent = egg.configData?.name || egg.configName;
                optgroup.appendChild(option);
            });
            
            select.appendChild(optgroup);
        });
    }
    
    async findGame() {
        this.showStatus('Searching for Pet Simulator 99 window...', 'info');
        
        try {
            const response = await fetch('/api/ps99/find-game', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.updateGameStatus('Connected', 'success');
                this.showStatus('Game window found and connected', 'success');
            } else {
                this.updateGameStatus('Not Found', 'warning');
                this.showStatus(data.message || 'Game window not found', 'warning');
            }
        } catch (error) {
            console.error('Error finding game:', error);
            this.updateGameStatus('Error', 'danger');
            this.showStatus('Error searching for game window', 'danger');
        }
    }
    
    async hatchOnce() {
        if (this.isHatching) {
            this.showStatus('Already hatching, please wait...', 'warning');
            return;
        }
        
        const selectedEgg = document.getElementById('eggSelect').value;
        this.updateHatchStatus('Hatching...', 'info');
        
        try {
            const response = await fetch('/api/ps99/hatch-egg', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ egg_name: selectedEgg || null })
            });
            
            const data = await response.json();
            this.handleHatchResult(data);
            
        } catch (error) {
            console.error('Error hatching egg:', error);
            this.updateHatchStatus('Error', 'danger');
            this.showStatus('Error during egg hatching', 'danger');
        }
    }
    
    async startAutoHatch() {
        if (this.isHatching) return;
        
        this.isHatching = true;
        this.updateUI();
        
        const selectedEgg = document.getElementById('eggSelect').value;
        const targetPets = Array.from(this.targetPets);
        
        try {
            const response = await fetch('/api/ps99/start-auto-hatch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    egg_types: selectedEgg ? [selectedEgg] : null,
                    target_pets: targetPets
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateHatchStatus('Auto Hatching', 'success');
                this.showStatus('Auto-hatch started', 'success');
            } else {
                this.isHatching = false;
                this.updateUI();
                this.showStatus(data.message || 'Failed to start auto-hatch', 'danger');
            }
        } catch (error) {
            console.error('Error starting auto-hatch:', error);
            this.isHatching = false;
            this.updateUI();
            this.showStatus('Error starting auto-hatch', 'danger');
        }
    }
    
    async stopAutoHatch() {
        try {
            const response = await fetch('/api/ps99/stop-auto-hatch', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.isHatching = false;
                this.updateUI();
                this.updateHatchStatus('Idle', 'secondary');
                this.showStatus('Auto-hatch stopped', 'info');
            } else {
                this.showStatus(data.message || 'Failed to stop auto-hatch', 'danger');
            }
        } catch (error) {
            console.error('Error stopping auto-hatch:', error);
            this.showStatus('Error stopping auto-hatch', 'danger');
        }
    }
    
    addTargetPet() {
        const input = document.getElementById('targetPetInput');
        const petName = input.value.trim();
        
        if (!petName) return;
        
        if (this.targetPets.has(petName)) {
            this.showStatus('Pet already in target list', 'warning');
            return;
        }
        
        this.targetPets.add(petName);
        this.updateTargetPetsList();
        input.value = '';
        
        // Add to server
        fetch('/api/ps99/add-target-pet', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pet_name: petName })
        }).catch(error => console.error('Error adding target pet:', error));
    }
    
    removeTargetPet(petName) {
        this.targetPets.delete(petName);
        this.updateTargetPetsList();
        
        // Remove from server
        fetch('/api/ps99/remove-target-pet', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pet_name: petName })
        }).catch(error => console.error('Error removing target pet:', error));
    }
    
    updateTargetPetsList() {
        const container = document.getElementById('targetPetsList');
        container.innerHTML = '';
        
        if (this.targetPets.size === 0) {
            container.innerHTML = '<small class="text-muted">No target pets set</small>';
            return;
        }
        
        this.targetPets.forEach(petName => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-primary target-pet-badge me-1 mb-1';
            badge.innerHTML = `
                ${petName}
                <button type="button" class="btn-close btn-close-white ms-1" 
                        style="font-size: 0.6em;" onclick="eggHatcher.removeTargetPet('${petName}')"></button>
            `;
            container.appendChild(badge);
        });
    }
    
    handleHatchResult(data) {
        if (data.success) {
            this.updateHatchStatus('Idle', 'secondary');
            this.stats.totalHatches++;
            
            const petsDetected = data.pets_detected || [];
            this.stats.petsDetected += petsDetected.length;
            
            // Check for target pets
            const targetPetsFound = petsDetected.filter(pet => this.targetPets.has(pet.name));
            this.stats.targetPetsFound += targetPetsFound.length;
            
            // Calculate success rate
            this.stats.successRate = this.stats.totalHatches > 0 
                ? (this.stats.petsDetected / this.stats.totalHatches * 100).toFixed(1)
                : 0;
            
            this.updateStats();
            this.addToLog(data);
            
            // Show target pet alert
            if (targetPetsFound.length > 0) {
                targetPetsFound.forEach(pet => this.showTargetPetAlert(pet.name));
            }
            
            this.showStatus(`Hatched ${data.egg_hatched}, detected ${petsDetected.length} pets`, 'success');
        } else {
            this.updateHatchStatus('Error', 'danger');
            this.showStatus(data.error || 'Hatching failed', 'danger');
        }
    }
    
    showTargetPetAlert(petName) {
        // Play notification sound (if allowed)
        this.playNotificationSound();
        
        // Show modal
        document.getElementById('targetPetName').textContent = petName;
        const modal = new bootstrap.Modal(document.getElementById('targetPetModal'));
        modal.show();
        
        // Browser notification (if allowed)
        if (Notification.permission === 'granted') {
            new Notification('Target Pet Hatched!', {
                body: `You hatched a ${petName}!`,
                icon: '/static/favicon.ico'
            });
        }
    }
    
    playNotificationSound() {
        try {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+D2u2AfBStcnM1fIUYVW9iWGVWRFaLTdKtxG0ktPNVJvXLLJaKLxnJaF7oEFMn3HHU2fRDbdKtvGz8rU9RDVOFoVXAb3b4HAL3n1aGnXZpLEOORvIc0T6uN2jPAG1H5+LKnqX4pD26tnz+SZgk=');
            audio.play().catch(() => {});
        } catch (error) {
            console.log('Could not play notification sound:', error);
        }
    }
    
    updateStats() {
        document.getElementById('totalHatches').textContent = this.stats.totalHatches;
        document.getElementById('petsDetected').textContent = this.stats.petsDetected;
        document.getElementById('targetPetsFound').textContent = this.stats.targetPetsFound;
        document.getElementById('successRate').textContent = this.stats.successRate + '%';
    }
    
    addToLog(result) {
        const log = document.getElementById('hatchingLog');
        
        // Remove "no results" message
        if (log.children.length === 1 && log.children[0].classList.contains('text-muted')) {
            log.innerHTML = '';
        }
        
        const logEntry = document.createElement('div');
        logEntry.className = 'border-bottom pb-2 mb-2';
        
        const timestamp = new Date().toLocaleTimeString();
        const petsDetected = result.pets_detected || [];
        const targetPets = petsDetected.filter(pet => this.targetPets.has(pet.name));
        
        logEntry.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <strong>${result.egg_hatched}</strong>
                    <small class="text-muted d-block">${timestamp}</small>
                </div>
                <div class="text-end">
                    ${petsDetected.length > 0 
                        ? `<span class="badge bg-success">${petsDetected.length} pets</span>`
                        : '<span class="badge bg-secondary">No pets</span>'
                    }
                    ${targetPets.length > 0 
                        ? `<span class="badge bg-warning ms-1">${targetPets.length} targets!</span>`
                        : ''
                    }
                </div>
            </div>
            ${petsDetected.length > 0 
                ? `<div class="mt-1">
                     <small class="text-muted">Pets: </small>
                     ${petsDetected.map(pet => 
                         `<span class="badge ${this.targetPets.has(pet.name) ? 'bg-warning' : 'bg-info'} me-1">${pet.name}</span>`
                     ).join('')}
                   </div>`
                : ''
            }
        `;
        
        log.insertBefore(logEntry, log.firstChild);
        
        // Limit log to 50 entries
        while (log.children.length > 50) {
            log.removeChild(log.lastChild);
        }
    }
    
    clearLog() {
        const log = document.getElementById('hatchingLog');
        log.innerHTML = '<div class="text-muted text-center"><i class="fas fa-clock me-2"></i>No hatching results yet</div>';
    }
    
    updateGameStatus(status, type) {
        const element = document.getElementById('gameStatus');
        element.textContent = status;
        element.className = `mb-1 text-${type}`;
    }
    
    updateHatchStatus(status, type) {
        const element = document.getElementById('hatchStatus');
        element.textContent = status;
        element.className = `mb-1 text-${type}`;
    }
    
    updateUI() {
        document.getElementById('startAutoHatchBtn').disabled = this.isHatching;
        document.getElementById('stopAutoHatchBtn').disabled = !this.isHatching;
        document.getElementById('hatchOnceBtn').disabled = this.isHatching;
    }
    
    startStatusUpdates() {
        // Update screenshot every 5 seconds
        setInterval(() => this.updateScreenshot(), 5000);
        
        // Update stats every 10 seconds
        setInterval(() => this.updateStatsFromServer(), 10000);
        
        // Update game status every 30 seconds
        setInterval(() => this.updateGameStatusFromServer(), 30000);
    }
    
    async updateScreenshot() {
        try {
            const response = await fetch('/api/ps99/screenshot');
            const data = await response.json();
            
            if (data.success && data.screenshot) {
                document.getElementById('gameScreenshot').src = `data:image/png;base64,${data.screenshot}`;
                document.getElementById('screenshotTime').textContent = new Date().toLocaleTimeString();
            }
        } catch (error) {
            console.error('Error updating screenshot:', error);
        }
    }
    
    async updateStatsFromServer() {
        try {
            const response = await fetch('/api/ps99/stats');
            const data = await response.json();
            
            if (data.success) {
                this.stats = { ...this.stats, ...data.stats };
                this.updateStats();
            }
        } catch (error) {
            console.error('Error updating stats:', error);
        }
    }
    
    async updateGameStatusFromServer() {
        try {
            const response = await fetch('/api/ps99/game-status');
            const data = await response.json();
            
            if (data.success) {
                this.updateGameStatus(data.game_connected ? 'Connected' : 'Not Connected', 
                                    data.game_connected ? 'success' : 'warning');
                this.updateHatchStatus(data.hatch_status, data.hatch_status_type);
                this.isHatching = data.is_hatching;
                this.updateUI();
            }
        } catch (error) {
            console.error('Error updating game status:', error);
        }
    }
    
    showStatus(message, type) {
        // Create or update status toast
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        // Add to toast container or create one
        let container = document.getElementById('toastContainer');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toastContainer';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        
        container.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
}

// Initialize egg hatcher when page loads
let eggHatcher;
document.addEventListener('DOMContentLoaded', function() {
    eggHatcher = new PS99EggHatcher();
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});