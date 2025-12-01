/**
 * NeuralVis.js - Cortex Neural Network Visualization
 * ===================================================
 * 
 * Bio-Cyberpunk styled 3D visualization of the P2P network
 * using Three.js and 3d-force-graph.
 * 
 * [STYLE]
 * - Deep black background
 * - Neon-colored nodes by role
 * - Particle animations for data flow
 * - Volumetric glow for topic clusters
 */

class NeuralVis {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        if (!this.container) {
            console.error(`[NeuralVis] Container #${containerId} not found`);
            return;
        }
        
        // Configuration
        // Determine WebSocket URL - use current location for iframe, or fallback to common ports
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host || 'localhost:8080';
        const defaultWsUrl = `${protocol}//${host}/ws/vis`;
        
        this.config = {
            wsUrl: options.wsUrl || defaultWsUrl,
            nodeSize: options.nodeSize || 8,
            linkOpacity: options.linkOpacity || 0.15,
            particleSpeed: options.particleSpeed || 0.02,
            breathingSpeed: options.breathingSpeed || 0.002,
            backgroundColor: options.backgroundColor || '#000000',
            ...options
        };
        
        // Role colors (Bio-Cyberpunk palette)
        this.roleColors = {
            'me': '#FFFFFF',           // White - My node
            'scout': '#00FFFF',        // Cyan - Scout/Разведчик
            'analyst': '#9D4EDD',      // Purple - Analyst/Аналитик
            'librarian': '#FFD700',    // Gold - Librarian/Хранитель
            'unknown': '#4A5568',      // Gray - Unknown role
            'council': '#FF6B6B',      // Coral - Council participant
            'bounty': '#00FF88',       // Green - Bounty hunter
        };
        
        // State
        this.graph = null;
        this.ws = null;
        this.nodes = new Map();
        this.links = new Map();
        this.particles = [];
        this.topicClusters = new Map();
        this.selectedNode = null;
        this.animationId = null;
        this.time = 0;
        
        // Initialize
        this._init();
    }
    
    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    
    _init() {
        // Create graph
        this.graph = ForceGraph3D({ controlType: 'orbit' })(this.container)
            .backgroundColor(this.config.backgroundColor)
            .showNavInfo(false)
            .nodeRelSize(this.config.nodeSize)
            .nodeVal(node => node.val || 10)
            .nodeColor(node => this._getNodeColor(node))
            .nodeOpacity(0.9)
            .nodeResolution(16)
            .linkWidth(link => link.active ? 2 : 0.5)
            .linkOpacity(this.config.linkOpacity)
            .linkColor(link => link.active ? '#00FFFF' : '#1a1a2e')
            .linkDirectionalParticles(link => link.particles || 0)
            .linkDirectionalParticleWidth(3)
            .linkDirectionalParticleSpeed(this.config.particleSpeed)
            .linkDirectionalParticleColor(() => '#00FFFF')
            .onNodeClick(node => this._onNodeClick(node))
            .onLinkClick(link => this._onLinkClick(link))
            .onNodeHover(node => this._onNodeHover(node))
            .d3AlphaDecay(0.05)
            .d3VelocityDecay(0.5)
            .warmupTicks(200)
            .cooldownTicks(0);
        
        // Compact forces - use internal d3 reference
        const d3 = this.graph.d3Force('link');
        if (d3) {
            this.graph.d3Force('link').distance(30).strength(1);
        }
        this.graph.d3Force('charge').strength(-20);
        
        // Custom node rendering with glow
        this.graph.nodeThreeObject(node => this._createNodeObject(node));
        
        // Add ambient effects
        this._addAmbientEffects();
        
        // Start breathing animation
        this._startBreathingAnimation();
        
        // Connect WebSocket
        this._connectWebSocket();
        
        // Handle resize
        window.addEventListener('resize', () => this._onResize());
        
        console.log('[NeuralVis] Initialized');
    }
    
    _createNodeObject(node) {
        const THREE = window.THREE;
        const group = new THREE.Group();
        
        // Core sphere
        const geometry = new THREE.SphereGeometry(1, 32, 32);
        const color = this._getNodeColor(node);
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.9,
        });
        
        const sphere = new THREE.Mesh(geometry, material);
        group.add(sphere);
        
        // Outer glow (larger, more transparent)
        const glowGeometry = new THREE.SphereGeometry(1.5, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.15,
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        group.add(glow);
        
        // Add ring for "me" node
        if (node.role === 'me' || node.id === 'me') {
            const ringGeometry = new THREE.RingGeometry(1.8, 2.2, 32);
            const ringMaterial = new THREE.MeshBasicMaterial({
                color: '#FFFFFF',
                transparent: true,
                opacity: 0.6,
                side: THREE.DoubleSide,
            });
            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.rotation.x = Math.PI / 2;
            group.add(ring);
        }
        
        // Store reference for animations
        node.__threeObj = group;
        node.__sphere = sphere;
        node.__glow = glow;
        
        return group;
    }
    
    _addAmbientEffects() {
        const THREE = window.THREE;
        const scene = this.graph.scene();
        
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);
        
        // Point lights for atmosphere
        const colors = [0x00FFFF, 0x9D4EDD, 0xFFD700];
        colors.forEach((color, i) => {
            const light = new THREE.PointLight(color, 0.3, 500);
            const angle = (i / colors.length) * Math.PI * 2;
            light.position.set(
                Math.cos(angle) * 200,
                100,
                Math.sin(angle) * 200
            );
            scene.add(light);
        });
        
        // Star field background
        this._createStarField(scene);
    }
    
    _createStarField(scene) {
        const THREE = window.THREE;
        const starCount = 2000;
        const positions = new Float32Array(starCount * 3);
        const colors = new Float32Array(starCount * 3);
        
        for (let i = 0; i < starCount; i++) {
            const i3 = i * 3;
            
            // Random position in sphere
            const radius = 800 + Math.random() * 400;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            
            positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i3 + 2] = radius * Math.cos(phi);
            
            // Random blue/purple/white color
            const colorChoice = Math.random();
            if (colorChoice < 0.33) {
                colors[i3] = 0.5; colors[i3 + 1] = 0.8; colors[i3 + 2] = 1.0;
            } else if (colorChoice < 0.66) {
                colors[i3] = 0.8; colors[i3 + 1] = 0.5; colors[i3 + 2] = 1.0;
            } else {
                colors[i3] = 1.0; colors[i3 + 1] = 1.0; colors[i3 + 2] = 1.0;
            }
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 1.5,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
        });
        
        const stars = new THREE.Points(geometry, material);
        scene.add(stars);
        this._stars = stars;
    }
    
    // =========================================================================
    // ANIMATIONS
    // =========================================================================
    
    _startBreathingAnimation() {
        const animate = () => {
            this.time += this.config.breathingSpeed;
            
            // Breathing effect for all nodes
            const breathScale = 1 + Math.sin(this.time * 2) * 0.1;
            const glowScale = 1.5 + Math.sin(this.time * 2) * 0.2;
            
            this.nodes.forEach((node, id) => {
                if (node.__sphere) {
                    // Pulse "me" node faster
                    const isMe = node.role === 'me' || node.id === 'me';
                    const nodeBreath = isMe 
                        ? 1 + Math.sin(this.time * 4) * 0.15
                        : breathScale;
                    
                    node.__sphere.scale.setScalar(nodeBreath);
                    
                    if (node.__glow) {
                        node.__glow.scale.setScalar(glowScale);
                        node.__glow.material.opacity = 0.1 + Math.sin(this.time * 2) * 0.05;
                    }
                }
            });
            
            // Rotate star field slowly
            if (this._stars) {
                this._stars.rotation.y += 0.0001;
            }
            
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    // =========================================================================
    // WEBSOCKET
    // =========================================================================
    
    _connectWebSocket() {
        try {
            this.ws = new WebSocket(this.config.wsUrl);
            
            this.ws.onopen = () => {
                console.log('[NeuralVis] WebSocket connected');
                this._requestInitialData();
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this._handleMessage(data);
                } catch (e) {
                    console.error('[NeuralVis] Parse error:', e);
                }
            };
            
            this.ws.onclose = () => {
                console.log('[NeuralVis] WebSocket closed, reconnecting...');
                setTimeout(() => this._connectWebSocket(), 3000);
            };
            
            this.ws.onerror = (e) => {
                console.error('[NeuralVis] WebSocket error:', e);
                // Try polling API as fallback
                if (this.nodes.size === 0) {
                    console.log('[NeuralVis] Trying API polling fallback');
                    this._startPolling();
                }
            };
        } catch (e) {
            console.warn('[NeuralVis] WebSocket not available, using polling/demo mode');
            this._startPolling();
        }
    }
    
    _startPolling() {
        // Try to fetch from API, fallback to demo data
        const apiUrl = `/api/vis/graph`;
        
        fetch(apiUrl)
            .then(response => {
                if (response.ok) return response.json();
                throw new Error('API not available');
            })
            .then(data => {
                console.log('[NeuralVis] Loaded data from API');
                if (data.nodes && data.links) {
                    this._updateGraph(data.nodes, data.links);
                    // Start polling interval
                    this._pollInterval = setInterval(() => this._pollApi(), 5000);
                } else {
                    this._loadDemoData();
                }
            })
            .catch(e => {
                console.log('[NeuralVis] API not available, loading demo data');
                this._loadDemoData();
            });
    }
    
    _pollApi() {
        fetch('/api/vis/graph')
            .then(response => response.json())
            .then(data => {
                if (data.nodes && data.links) {
                    this._updateGraph(data.nodes, data.links);
                }
            })
            .catch(e => {
                console.debug('[NeuralVis] Poll error:', e);
            });
    }
    
    _requestInitialData() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'get_graph' }));
        }
    }
    
    _handleMessage(data) {
        switch (data.type) {
            case 'graph':
                this._updateGraph(data.nodes, data.links);
                break;
                
            case 'pulse':
                this._emitParticle(data.from, data.to, data.value);
                break;
                
            case 'node_update':
                this._updateNode(data.node);
                break;
                
            case 'node_add':
                this._addNode(data.node);
                break;
                
            case 'node_remove':
                this._removeNode(data.nodeId);
                break;
                
            case 'link_add':
                this._addLink(data.link);
                break;
                
            case 'cluster':
                this._updateCluster(data.topic, data.nodes);
                break;
                
            default:
                console.log('[NeuralVis] Unknown message type:', data.type);
        }
    }
    
    // =========================================================================
    // GRAPH OPERATIONS
    // =========================================================================
    
    _updateGraph(nodes, links) {
        // Update internal state
        this.nodes.clear();
        this.links.clear();
        
        nodes.forEach(node => {
            this.nodes.set(node.id, node);
        });
        
        links.forEach(link => {
            const key = `${link.source}-${link.target}`;
            this.links.set(key, link);
        });
        
        // Update graph
        this.graph.graphData({ nodes, links });
        
        console.log(`[NeuralVis] Graph updated: ${nodes.length} nodes, ${links.length} links`);
    }
    
    _addNode(node) {
        this.nodes.set(node.id, node);
        
        const { nodes, links } = this.graph.graphData();
        nodes.push(node);
        this.graph.graphData({ nodes, links });
    }
    
    _removeNode(nodeId) {
        this.nodes.delete(nodeId);
        
        const { nodes, links } = this.graph.graphData();
        const filteredNodes = nodes.filter(n => n.id !== nodeId);
        const filteredLinks = links.filter(l => 
            l.source.id !== nodeId && l.target.id !== nodeId
        );
        this.graph.graphData({ nodes: filteredNodes, links: filteredLinks });
    }
    
    _updateNode(nodeData) {
        const node = this.nodes.get(nodeData.id);
        if (node) {
            Object.assign(node, nodeData);
            this.graph.nodeColor(this.graph.nodeColor()); // Force re-render
        }
    }
    
    _addLink(link) {
        const key = `${link.source}-${link.target}`;
        this.links.set(key, link);
        
        const { nodes, links } = this.graph.graphData();
        links.push(link);
        this.graph.graphData({ nodes, links });
    }
    
    _emitParticle(fromId, toId, value = 1) {
        const { links } = this.graph.graphData();
        
        // Find the link
        const link = links.find(l => 
            (l.source.id === fromId && l.target.id === toId) ||
            (l.source.id === toId && l.target.id === fromId)
        );
        
        if (link) {
            // Activate link
            link.active = true;
            link.particles = Math.min(value, 10);
            
            // Update visualization
            this.graph
                .linkWidth(l => l.active ? 2 : 0.5)
                .linkOpacity(l => l.active ? 0.8 : this.config.linkOpacity)
                .linkDirectionalParticles(l => l.particles || 0);
            
            // Deactivate after animation
            setTimeout(() => {
                link.active = false;
                link.particles = 0;
                this.graph
                    .linkWidth(l => l.active ? 2 : 0.5)
                    .linkOpacity(l => l.active ? 0.8 : this.config.linkOpacity)
                    .linkDirectionalParticles(l => l.particles || 0);
            }, 2000);
        }
    }
    
    _updateCluster(topic, nodeIds) {
        this.topicClusters.set(topic, nodeIds);
        
        // TODO: Add volumetric glow around cluster
        // This would require custom Three.js shaders
    }
    
    // =========================================================================
    // INTERACTION
    // =========================================================================
    
    _onNodeClick(node) {
        if (!node) return;
        
        this.selectedNode = node;
        
        // Show info card
        this._showNodeCard(node);
        
        // Zoom to node
        this.graph.cameraPosition(
            { x: node.x + 50, y: node.y + 50, z: node.z + 50 },
            { x: node.x, y: node.y, z: node.z },
            1000
        );
    }
    
    _onLinkClick(link) {
        if (!link) return;
        
        // Show transaction flow
        this._showLinkCard(link);
    }
    
    _onNodeHover(node) {
        this.container.style.cursor = node ? 'pointer' : 'default';
        
        // Highlight connected nodes
        if (node) {
            this.graph.nodeColor(n => {
                if (n.id === node.id) return '#FFFFFF';
                return this._getNodeColor(n);
            });
        } else {
            this.graph.nodeColor(n => this._getNodeColor(n));
        }
    }
    
    _showNodeCard(node) {
        // Remove existing card
        const existingCard = document.getElementById('neural-vis-card');
        if (existingCard) existingCard.remove();
        
        const card = document.createElement('div');
        card.id = 'neural-vis-card';
        card.className = 'neural-vis-card';
        card.innerHTML = `
            <div class="card-header">
                <span class="role-badge" style="background: ${this._getNodeColor(node)}">
                    ${(node.role || 'unknown').toUpperCase()}
                </span>
                <button class="close-btn" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="card-body">
                <div class="card-row">
                    <span class="label">ID:</span>
                    <span class="value mono">${node.id.substring(0, 16)}...</span>
                </div>
                <div class="card-row">
                    <span class="label">Trust Score:</span>
                    <span class="value">${(node.trustScore || 0.5).toFixed(3)}</span>
                </div>
                <div class="card-row">
                    <span class="label">Current Task:</span>
                    <span class="value">${node.currentTask || 'Idle'}</span>
                </div>
                ${node.topic ? `
                <div class="card-row">
                    <span class="label">Topic:</span>
                    <span class="value">${node.topic}</span>
                </div>
                ` : ''}
            </div>
        `;
        
        this.container.appendChild(card);
    }
    
    _showLinkCard(link) {
        const existingCard = document.getElementById('neural-vis-card');
        if (existingCard) existingCard.remove();
        
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        
        const card = document.createElement('div');
        card.id = 'neural-vis-card';
        card.className = 'neural-vis-card';
        card.innerHTML = `
            <div class="card-header">
                <span class="role-badge" style="background: #00FFFF">CONNECTION</span>
                <button class="close-btn" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="card-body">
                <div class="card-row">
                    <span class="label">From:</span>
                    <span class="value mono">${sourceId.substring(0, 12)}...</span>
                </div>
                <div class="card-row">
                    <span class="label">To:</span>
                    <span class="value mono">${targetId.substring(0, 12)}...</span>
                </div>
                <div class="card-row">
                    <span class="label">Messages:</span>
                    <span class="value">${link.messageCount || 0}</span>
                </div>
                <div class="card-row">
                    <span class="label">Balance:</span>
                    <span class="value">${link.balance || 0}</span>
                </div>
            </div>
        `;
        
        this.container.appendChild(card);
    }
    
    // =========================================================================
    // HELPERS
    // =========================================================================
    
    _getNodeColor(node) {
        if (node.id === 'me' || node.role === 'me') {
            return this.roleColors.me;
        }
        return this.roleColors[node.role] || this.roleColors.unknown;
    }
    
    _onResize() {
        this.graph.width(this.container.clientWidth);
        this.graph.height(this.container.clientHeight);
    }
    
    // =========================================================================
    // DEMO DATA
    // =========================================================================
    
    _loadDemoData() {
        console.log('[NeuralVis] Loading demo data');
        
        const nodes = [
            { id: 'me', role: 'me', val: 25, trustScore: 1.0, currentTask: 'Coordinating', x: 0, y: 0, z: 0 },
            { id: 'scout_1', role: 'scout', val: 15, trustScore: 0.85, currentTask: 'Searching: quantum computing', x: 50, y: 20, z: -30 },
            { id: 'scout_2', role: 'scout', val: 12, trustScore: 0.72, currentTask: 'Idle', x: -40, y: 30, z: 20 },
            { id: 'analyst_1', role: 'analyst', val: 18, trustScore: 0.91, currentTask: 'Analyzing text', x: 30, y: -40, z: 50 },
            { id: 'analyst_2', role: 'analyst', val: 16, trustScore: 0.88, currentTask: 'Council deliberation', x: -50, y: -20, z: -40 },
            { id: 'analyst_3', role: 'analyst', val: 14, trustScore: 0.79, currentTask: 'Idle', x: 60, y: 10, z: 30 },
            { id: 'librarian_1', role: 'librarian', val: 20, trustScore: 0.95, currentTask: 'Indexing: AI safety', x: -30, y: 50, z: -20 },
            { id: 'librarian_2', role: 'librarian', val: 17, trustScore: 0.82, currentTask: 'DHT sync', x: 20, y: -50, z: -50 },
            { id: 'peer_1', role: 'unknown', val: 10, trustScore: 0.5, currentTask: 'Unknown', x: -60, y: -30, z: 40 },
            { id: 'peer_2', role: 'unknown', val: 8, trustScore: 0.45, currentTask: 'Unknown', x: 40, y: 40, z: 60 },
        ];
        
        const links = [
            { source: 'me', target: 'scout_1' },
            { source: 'me', target: 'scout_2' },
            { source: 'me', target: 'analyst_1' },
            { source: 'me', target: 'librarian_1' },
            { source: 'scout_1', target: 'analyst_1' },
            { source: 'scout_1', target: 'analyst_2' },
            { source: 'scout_2', target: 'analyst_3' },
            { source: 'analyst_1', target: 'librarian_1' },
            { source: 'analyst_2', target: 'librarian_1' },
            { source: 'analyst_3', target: 'librarian_2' },
            { source: 'librarian_1', target: 'librarian_2' },
            { source: 'peer_1', target: 'scout_1' },
            { source: 'peer_2', target: 'analyst_2' },
        ];
        
        this._updateGraph(nodes, links);
        
        // Set camera closer to see the nodes
        setTimeout(() => {
            this.graph.cameraPosition(
                { x: 0, y: 50, z: 150 },
                { x: 0, y: 0, z: 0 },
                1000
            );
        }, 500);
        
        // Demo: emit particles periodically
        setInterval(() => {
            const linkIdx = Math.floor(Math.random() * links.length);
            const link = links[linkIdx];
            this._emitParticle(link.source, link.target, Math.ceil(Math.random() * 5));
        }, 2000);
    }
    
    // =========================================================================
    // PUBLIC API
    // =========================================================================
    
    /**
     * Add a node to the visualization
     */
    addNode(node) {
        this._addNode(node);
    }
    
    /**
     * Remove a node from the visualization
     */
    removeNode(nodeId) {
        this._removeNode(nodeId);
    }
    
    /**
     * Emit a particle along a link (data transfer visualization)
     */
    pulse(fromId, toId, value = 1) {
        this._emitParticle(fromId, toId, value);
    }
    
    /**
     * Focus camera on a specific node
     */
    focusNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (node) {
            this._onNodeClick(node);
        }
    }
    
    /**
     * Update node data
     */
    updateNode(nodeId, data) {
        this._updateNode({ id: nodeId, ...data });
    }
    
    /**
     * Set graph data directly
     */
    setData(nodes, links) {
        this._updateGraph(nodes, links);
    }
    
    /**
     * Destroy the visualization
     */
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.ws) {
            this.ws.close();
        }
        if (this.graph) {
            this.graph._destructor();
        }
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NeuralVis;
}

// Global export
window.NeuralVis = NeuralVis;

