/**
 * Comprehensive Raytracing Debugger
 * Real-time visualization and analysis tool for acoustic raytracing
 */

import { vec3, mat4 } from 'gl-matrix';
import * as dat from 'dat.gui';
import { AcousticRaytracer, RaytracerConfig } from '../audio/acoustic-raytracer';
import { AcousticRay, RayDistributionType } from '../audio/ray-types';
import { PerformanceMonitor } from '../utils/performance-monitor';

export interface DebugConfig {
    visualization: {
        showRays: boolean;
        showBouncePoints: boolean;
        showEnergyLevels: boolean;
        rayOpacity: number;
        maxRaysToShow: number;
        colorByFrequency: boolean;
        colorByEnergy: boolean;
        colorByBounceCount: boolean;
    };
    analysis: {
        showFrequencyResponse: boolean;
        showImpulseResponse: boolean;
        showStatistics: boolean;
        updateRate: number; // Hz
    };
    performance: {
        showGPUMetrics: boolean;
        showFrameTimes: boolean;
        showRayProcessingRate: boolean;
        targetFPS: number;
    };
}

export interface RayStatistics {
    totalRays: number;
    activeRays: number;
    collectedRays: number;
    averageBounces: number;
    maxBounces: number;
    totalEnergy: number;
    averageEnergy: number;
    energyByFrequency: Float32Array; // 8 bands
    bounceDistribution: number[]; // Histogram of bounce counts
}

export interface AcousticMetrics {
    rt60: number; // Reverberation time
    clarity: number; // C80 clarity index
    definition: number; // D50 definition
    centerTime: number; // Ts center time
    earlyDecayTime: number; // EDT
    bassRatio: number; // BR bass ratio
}

export class RaytracingDebugger {
    private device: GPUDevice;
    private canvas: HTMLCanvasElement;
    private raytracer: AcousticRaytracer;
    private performanceMonitor: PerformanceMonitor;
    
    // Debug configuration
    private config: DebugConfig;
    private gui: dat.GUI;
    
    // Visualization resources
    private rayVisualizationPipeline: GPURenderPipeline | null = null;
    private rayVertexBuffer: GPUBuffer | null = null;
    private rayIndexBuffer: GPUBuffer | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    
    // Analysis data
    private rayStatistics: RayStatistics;
    private acousticMetrics: AcousticMetrics;
    private frequencyResponse: Float32Array;
    private impulseResponse: Float32Array;
    
    // UI elements
    private statisticsPanel: HTMLElement | null = null;
    private frequencyChart: HTMLCanvasElement | null = null;
    private impulseChart: HTMLCanvasElement | null = null;
    private performanceChart: HTMLCanvasElement | null = null;
    
    // Update timing
    private lastUpdateTime: number = 0;
    private updateInterval: number = 100; // ms
    
    constructor(
        device: GPUDevice,
        canvas: HTMLCanvasElement,
        raytracer: AcousticRaytracer
    ) {
        this.device = device;
        this.canvas = canvas;
        this.raytracer = raytracer;
        this.performanceMonitor = new PerformanceMonitor();
        
        // Initialize default configuration
        this.config = {
            visualization: {
                showRays: true,
                showBouncePoints: true,
                showEnergyLevels: true,
                rayOpacity: 0.7,
                maxRaysToShow: 1000,
                colorByFrequency: false,
                colorByEnergy: true,
                colorByBounceCount: false
            },
            analysis: {
                showFrequencyResponse: true,
                showImpulseResponse: true,
                showStatistics: true,
                updateRate: 10
            },
            performance: {
                showGPUMetrics: true,
                showFrameTimes: true,
                showRayProcessingRate: true,
                targetFPS: 60
            }
        };
        
        // Initialize data structures
        this.rayStatistics = this.createEmptyStatistics();
        this.acousticMetrics = this.createEmptyMetrics();
        this.frequencyResponse = new Float32Array(8);
        this.impulseResponse = new Float32Array(4096); // 2 seconds at 48kHz
        
        this.updateInterval = 1000 / this.config.analysis.updateRate;
    }
    
    /**
     * Initialize the debugger
     */
    async initialize(): Promise<void> {
        // Create UI elements
        this.createDebugUI();
        this.createVisualizationPanels();
        
        // Initialize visualization resources
        await this.createVisualizationPipeline();
        
        // Start performance monitoring
        this.performanceMonitor.startMonitoring();
        
        console.log('Raytracing Debugger initialized');
    }
    
    /**
     * Create debug UI controls
     */
    private createDebugUI(): void {
        this.gui = new dat.GUI({ name: 'Raytracing Debugger' });
        
        // Visualization controls
        const vizFolder = this.gui.addFolder('Visualization');
        vizFolder.add(this.config.visualization, 'showRays').name('Show Rays');
        vizFolder.add(this.config.visualization, 'showBouncePoints').name('Show Bounce Points');
        vizFolder.add(this.config.visualization, 'showEnergyLevels').name('Show Energy Levels');
        vizFolder.add(this.config.visualization, 'rayOpacity', 0, 1).name('Ray Opacity');
        vizFolder.add(this.config.visualization, 'maxRaysToShow', 100, 5000).name('Max Rays to Show');
        
        // Color mode controls
        const colorFolder = vizFolder.addFolder('Color Mode');
        colorFolder.add(this.config.visualization, 'colorByFrequency').name('Color by Frequency');
        colorFolder.add(this.config.visualization, 'colorByEnergy').name('Color by Energy');
        colorFolder.add(this.config.visualization, 'colorByBounceCount').name('Color by Bounces');
        
        // Analysis controls
        const analysisFolder = this.gui.addFolder('Analysis');
        analysisFolder.add(this.config.analysis, 'showFrequencyResponse').name('Frequency Response');
        analysisFolder.add(this.config.analysis, 'showImpulseResponse').name('Impulse Response');
        analysisFolder.add(this.config.analysis, 'showStatistics').name('Statistics');
        analysisFolder.add(this.config.analysis, 'updateRate', 1, 60).name('Update Rate (Hz)')
            .onChange((value: number) => {
                this.updateInterval = 1000 / value;
            });
        
        // Performance controls
        const perfFolder = this.gui.addFolder('Performance');
        perfFolder.add(this.config.performance, 'showGPUMetrics').name('GPU Metrics');
        perfFolder.add(this.config.performance, 'showFrameTimes').name('Frame Times');
        perfFolder.add(this.config.performance, 'showRayProcessingRate').name('Ray Processing Rate');
        perfFolder.add(this.config.performance, 'targetFPS', 30, 120).name('Target FPS');
        
        // Action buttons
        const actionsFolder = this.gui.addFolder('Actions');
        actionsFolder.add({ exportData: () => this.exportDebugData() }, 'exportData').name('Export Debug Data');
        actionsFolder.add({ resetStats: () => this.resetStatistics() }, 'resetStats').name('Reset Statistics');
        actionsFolder.add({ captureFrame: () => this.captureFrame() }, 'captureFrame').name('Capture Frame');
        
        vizFolder.open();
        analysisFolder.open();
        perfFolder.open();
    }
    
    /**
     * Create visualization panels
     */
    private createVisualizationPanels(): void {
        // Create container for debug panels
        const debugContainer = document.createElement('div');
        debugContainer.id = 'raytracing-debug-container';
        debugContainer.style.cssText = `
            position: fixed;
            top: 10px;
            left: 10px;
            width: 400px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
            z-index: 1000;
            pointer-events: none;
        `;
        document.body.appendChild(debugContainer);
        
        // Statistics panel
        this.statisticsPanel = document.createElement('div');
        this.statisticsPanel.id = 'statistics-panel';
        debugContainer.appendChild(this.statisticsPanel);
        
        // Frequency response chart
        this.frequencyChart = document.createElement('canvas');
        this.frequencyChart.width = 380;
        this.frequencyChart.height = 100;
        this.frequencyChart.style.cssText = 'margin: 5px 0; border: 1px solid #333;';
        debugContainer.appendChild(this.frequencyChart);
        
        // Impulse response chart
        this.impulseChart = document.createElement('canvas');
        this.impulseChart.width = 380;
        this.impulseChart.height = 100;
        this.impulseChart.style.cssText = 'margin: 5px 0; border: 1px solid #333;';
        debugContainer.appendChild(this.impulseChart);
        
        // Performance chart
        this.performanceChart = document.createElement('canvas');
        this.performanceChart.width = 380;
        this.performanceChart.height = 80;
        this.performanceChart.style.cssText = 'margin: 5px 0; border: 1px solid #333;';
        debugContainer.appendChild(this.performanceChart);
    }
    
    /**
     * Create empty statistics object
     */
    private createEmptyStatistics(): RayStatistics {
        return {
            totalRays: 0,
            activeRays: 0,
            collectedRays: 0,
            averageBounces: 0,
            maxBounces: 0,
            totalEnergy: 0,
            averageEnergy: 0,
            energyByFrequency: new Float32Array(8),
            bounceDistribution: new Array(50).fill(0)
        };
    }
    
    /**
     * Create empty acoustic metrics object
     */
    private createEmptyMetrics(): AcousticMetrics {
        return {
            rt60: 0,
            clarity: 0,
            definition: 0,
            centerTime: 0,
            earlyDecayTime: 0,
            bassRatio: 0
        };
    }

    /**
     * Create visualization pipeline for ray rendering
     */
    private async createVisualizationPipeline(): Promise<void> {
        // Ray visualization vertex shader
        const rayVertexShader = `
            struct VertexInput {
                @location(0) position: vec3<f32>,
                @location(1) color: vec3<f32>,
                @location(2) energy: f32,
            }

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec3<f32>,
                @location(1) energy: f32,
            }

            struct Uniforms {
                viewProjection: mat4x4<f32>,
                opacity: f32,
                energyScale: f32,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;

            @vertex
            fn main(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                output.position = uniforms.viewProjection * vec4<f32>(input.position, 1.0);
                output.color = input.color;
                output.energy = input.energy * uniforms.energyScale;
                return output;
            }
        `;

        // Ray visualization fragment shader
        const rayFragmentShader = `
            struct FragmentInput {
                @location(0) color: vec3<f32>,
                @location(1) energy: f32,
            }

            @fragment
            fn main(input: FragmentInput) -> @location(0) vec4<f32> {
                let alpha = min(input.energy, 1.0);
                return vec4<f32>(input.color, alpha);
            }
        `;

        // Create shader modules
        const vertexModule = this.device.createShaderModule({ code: rayVertexShader });
        const fragmentModule = this.device.createShaderModule({ code: rayFragmentShader });

        // Create uniform buffer
        this.uniformBuffer = this.device.createBuffer({
            size: 80, // mat4x4 (64 bytes) + 2 floats (8 bytes) + padding (8 bytes)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Create bind group layout
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'uniform' }
            }]
        });

        // Create pipeline
        this.rayVisualizationPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout]
            }),
            vertex: {
                module: vertexModule,
                entryPoint: 'main',
                buffers: [{
                    arrayStride: 28, // 3 floats (position) + 3 floats (color) + 1 float (energy)
                    attributes: [
                        { format: 'float32x3', offset: 0, shaderLocation: 0 }, // position
                        { format: 'float32x3', offset: 12, shaderLocation: 1 }, // color
                        { format: 'float32', offset: 24, shaderLocation: 2 }, // energy
                    ]
                }]
            },
            fragment: {
                module: fragmentModule,
                entryPoint: 'main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'line-list'
            }
        });
    }

    /**
     * Update debug information
     */
    async update(deltaTime: number): Promise<void> {
        const currentTime = performance.now();

        if (currentTime - this.lastUpdateTime < this.updateInterval) {
            return;
        }

        this.lastUpdateTime = currentTime;

        // Update performance metrics
        this.performanceMonitor.update(deltaTime);

        // Get ray statistics from raytracer
        this.rayStatistics = await this.raytracer.getRayStatistics();

        // Calculate acoustic metrics from impulse response
        this.calculateAcousticMetrics();

        // Update UI panels
        this.updateStatisticsPanel();
        this.updateFrequencyChart();
        this.updateImpulseChart();
        this.updatePerformanceChart();
    }

    /**
     * Calculate acoustic metrics from impulse response
     */
    private calculateAcousticMetrics(): void {
        if (this.impulseResponse.length === 0) return;

        const sampleRate = 48000;
        const dt = 1.0 / sampleRate;

        // Calculate RT60 (reverberation time)
        this.acousticMetrics.rt60 = this.calculateRT60(this.impulseResponse, sampleRate);

        // Calculate C80 (clarity index)
        this.acousticMetrics.clarity = this.calculateClarity(this.impulseResponse, sampleRate);

        // Calculate D50 (definition)
        this.acousticMetrics.definition = this.calculateDefinition(this.impulseResponse, sampleRate);

        // Calculate center time
        this.acousticMetrics.centerTime = this.calculateCenterTime(this.impulseResponse, sampleRate);

        // Calculate early decay time
        this.acousticMetrics.earlyDecayTime = this.calculateEDT(this.impulseResponse, sampleRate);

        // Calculate bass ratio
        this.acousticMetrics.bassRatio = this.calculateBassRatio();
    }

    /**
     * Calculate RT60 reverberation time
     */
    private calculateRT60(ir: Float32Array, sampleRate: number): number {
        // Find peak and calculate decay curve
        let peak = 0;
        let peakIndex = 0;

        for (let i = 0; i < ir.length; i++) {
            if (Math.abs(ir[i]) > peak) {
                peak = Math.abs(ir[i]);
                peakIndex = i;
            }
        }

        if (peak === 0) return 0;

        // Calculate energy decay curve
        const energyCurve = new Float32Array(ir.length - peakIndex);
        for (let i = peakIndex; i < ir.length; i++) {
            energyCurve[i - peakIndex] = ir[i] * ir[i];
        }

        // Find -60dB point (1/1000000 of peak energy)
        const targetEnergy = peak * peak / 1000000;

        for (let i = 0; i < energyCurve.length; i++) {
            if (energyCurve[i] <= targetEnergy) {
                return i / sampleRate;
            }
        }

        return 0; // No decay found
    }

    /**
     * Calculate C80 clarity index
     */
    private calculateClarity(ir: Float32Array, sampleRate: number): number {
        const splitTime = 0.08; // 80ms
        const splitSample = Math.floor(splitTime * sampleRate);

        let earlyEnergy = 0;
        let lateEnergy = 0;

        for (let i = 0; i < Math.min(splitSample, ir.length); i++) {
            earlyEnergy += ir[i] * ir[i];
        }

        for (let i = splitSample; i < ir.length; i++) {
            lateEnergy += ir[i] * ir[i];
        }

        if (lateEnergy === 0) return Infinity;
        return 10 * Math.log10(earlyEnergy / lateEnergy);
    }

    /**
     * Calculate D50 definition
     */
    private calculateDefinition(ir: Float32Array, sampleRate: number): number {
        const splitTime = 0.05; // 50ms
        const splitSample = Math.floor(splitTime * sampleRate);

        let earlyEnergy = 0;
        let totalEnergy = 0;

        for (let i = 0; i < ir.length; i++) {
            const energy = ir[i] * ir[i];
            totalEnergy += energy;

            if (i < splitSample) {
                earlyEnergy += energy;
            }
        }

        return totalEnergy === 0 ? 0 : earlyEnergy / totalEnergy;
    }

    /**
     * Calculate center time
     */
    private calculateCenterTime(ir: Float32Array, sampleRate: number): number {
        let totalEnergy = 0;
        let weightedSum = 0;

        for (let i = 0; i < ir.length; i++) {
            const energy = ir[i] * ir[i];
            const time = i / sampleRate;

            totalEnergy += energy;
            weightedSum += energy * time;
        }

        return totalEnergy === 0 ? 0 : weightedSum / totalEnergy;
    }

    /**
     * Calculate early decay time
     */
    private calculateEDT(ir: Float32Array, sampleRate: number): number {
        // Simplified EDT calculation - would need more sophisticated analysis in practice
        return this.calculateRT60(ir, sampleRate) * 0.8; // Approximation
    }

    /**
     * Calculate bass ratio
     */
    private calculateBassRatio(): number {
        // Ratio of low frequency energy (125-250Hz) to mid frequency energy (500-1kHz)
        const lowFreqEnergy = this.rayStatistics.energyByFrequency[0] + this.rayStatistics.energyByFrequency[1];
        const midFreqEnergy = this.rayStatistics.energyByFrequency[2] + this.rayStatistics.energyByFrequency[3];

        return midFreqEnergy === 0 ? 0 : lowFreqEnergy / midFreqEnergy;
    }

    /**
     * Update statistics panel
     */
    private updateStatisticsPanel(): void {
        if (!this.statisticsPanel || !this.config.analysis.showStatistics) return;

        const stats = this.rayStatistics;
        const metrics = this.acousticMetrics;
        const perf = this.performanceMonitor.getMetrics();

        this.statisticsPanel.innerHTML = `
            <div style="margin-bottom: 10px;">
                <h3 style="margin: 0 0 5px 0; color: #4CAF50;">Ray Statistics</h3>
                <div>Total Rays: ${stats.totalRays}</div>
                <div>Active Rays: ${stats.activeRays}</div>
                <div>Collected Rays: ${stats.collectedRays}</div>
                <div>Avg Bounces: ${stats.averageBounces.toFixed(1)}</div>
                <div>Max Bounces: ${stats.maxBounces}</div>
                <div>Total Energy: ${stats.totalEnergy.toFixed(3)}</div>
                <div>Avg Energy: ${stats.averageEnergy.toFixed(3)}</div>
            </div>

            <div style="margin-bottom: 10px;">
                <h3 style="margin: 0 0 5px 0; color: #2196F3;">Acoustic Metrics</h3>
                <div>RT60: ${metrics.rt60.toFixed(2)}s</div>
                <div>C80 Clarity: ${metrics.clarity.toFixed(1)}dB</div>
                <div>D50 Definition: ${(metrics.definition * 100).toFixed(1)}%</div>
                <div>Center Time: ${(metrics.centerTime * 1000).toFixed(1)}ms</div>
                <div>EDT: ${metrics.earlyDecayTime.toFixed(2)}s</div>
                <div>Bass Ratio: ${metrics.bassRatio.toFixed(2)}</div>
            </div>

            <div>
                <h3 style="margin: 0 0 5px 0; color: #FF9800;">Performance</h3>
                <div>FPS: ${perf.fps.toFixed(1)}</div>
                <div>Frame Time: ${perf.frameTime.toFixed(1)}ms</div>
                <div>GPU Usage: ${(perf.gpuUtilization * 100).toFixed(1)}%</div>
                <div>Ray Rate: ${(stats.totalRays / (perf.frameTime / 1000)).toFixed(0)}/s</div>
            </div>
        `;
    }

    /**
     * Update frequency response chart
     */
    private updateFrequencyChart(): void {
        if (!this.frequencyChart || !this.config.analysis.showFrequencyResponse) return;

        const ctx = this.frequencyChart.getContext('2d');
        if (!ctx) return;

        const width = this.frequencyChart.width;
        const height = this.frequencyChart.height;

        // Clear canvas
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);

        // Draw frequency response
        const frequencies = ['125', '250', '500', '1k', '2k', '4k', '8k', '16k'];
        const energies = this.rayStatistics.energyByFrequency;

        if (energies.length === 0) return;

        const maxEnergy = Math.max(...energies);
        if (maxEnergy === 0) return;

        const barWidth = width / frequencies.length;

        for (let i = 0; i < frequencies.length; i++) {
            const normalizedEnergy = energies[i] / maxEnergy;
            const barHeight = normalizedEnergy * (height - 20);

            // Color gradient based on frequency
            const hue = (i / frequencies.length) * 240; // Blue to red
            ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;

            ctx.fillRect(i * barWidth + 2, height - barHeight - 10, barWidth - 4, barHeight);

            // Draw frequency labels
            ctx.fillStyle = '#ffffff';
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(frequencies[i], i * barWidth + barWidth / 2, height - 2);
        }

        // Draw title
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('Frequency Response (Hz)', 5, 15);
    }

    /**
     * Update impulse response chart
     */
    private updateImpulseChart(): void {
        if (!this.impulseChart || !this.config.analysis.showImpulseResponse) return;

        const ctx = this.impulseChart.getContext('2d');
        if (!ctx) return;

        const width = this.impulseChart.width;
        const height = this.impulseChart.height;

        // Clear canvas
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);

        if (this.impulseResponse.length === 0) return;

        // Find peak for normalization
        const maxValue = Math.max(...this.impulseResponse.map(Math.abs));
        if (maxValue === 0) return;

        // Draw impulse response waveform
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 1;
        ctx.beginPath();

        for (let i = 0; i < this.impulseResponse.length; i++) {
            const x = (i / this.impulseResponse.length) * width;
            const y = height / 2 - (this.impulseResponse[i] / maxValue) * (height / 2 - 10);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw zero line
        ctx.strokeStyle = '#666666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Draw title
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('Impulse Response', 5, 15);

        // Draw time scale
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        const timeLength = this.impulseResponse.length / 48000; // Assuming 48kHz
        for (let i = 0; i <= 4; i++) {
            const x = (i / 4) * width;
            const time = (i / 4) * timeLength;
            ctx.fillText(`${time.toFixed(2)}s`, x, height - 2);
        }
    }

    /**
     * Update performance chart
     */
    private updatePerformanceChart(): void {
        if (!this.performanceChart || !this.config.performance.showFrameTimes) return;

        const ctx = this.performanceChart.getContext('2d');
        if (!ctx) return;

        const width = this.performanceChart.width;
        const height = this.performanceChart.height;

        // Clear canvas
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);

        const metrics = this.performanceMonitor.getMetrics();
        const frameTimeHistory = this.performanceMonitor.getFrameTimeHistory();

        if (frameTimeHistory.length === 0) return;

        // Draw frame time graph
        const maxFrameTime = Math.max(...frameTimeHistory);
        const targetFrameTime = 1000 / this.config.performance.targetFPS;

        ctx.strokeStyle = '#2196F3';
        ctx.lineWidth = 1;
        ctx.beginPath();

        for (let i = 0; i < frameTimeHistory.length; i++) {
            const x = (i / frameTimeHistory.length) * width;
            const y = height - (frameTimeHistory[i] / maxFrameTime) * (height - 20);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw target frame time line
        ctx.strokeStyle = '#FF5722';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        const targetY = height - (targetFrameTime / maxFrameTime) * (height - 20);
        ctx.moveTo(0, targetY);
        ctx.lineTo(width, targetY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw title and current values
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`Frame Times (Current: ${metrics.frameTime.toFixed(1)}ms, Target: ${targetFrameTime.toFixed(1)}ms)`, 5, 15);
    }

    /**
     * Export debug data
     */
    private exportDebugData(): void {
        const debugData = {
            timestamp: new Date().toISOString(),
            rayStatistics: this.rayStatistics,
            acousticMetrics: this.acousticMetrics,
            frequencyResponse: Array.from(this.frequencyResponse),
            impulseResponse: Array.from(this.impulseResponse),
            performanceMetrics: this.performanceMonitor.getMetrics(),
            configuration: this.config
        };

        const dataStr = JSON.stringify(debugData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });

        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `raytracing-debug-${Date.now()}.json`;
        link.click();

        console.log('Debug data exported');
    }

    /**
     * Reset statistics
     */
    private resetStatistics(): void {
        this.rayStatistics = this.createEmptyStatistics();
        this.acousticMetrics = this.createEmptyMetrics();
        this.performanceMonitor.reset();
        console.log('Statistics reset');
    }

    /**
     * Capture current frame for analysis
     */
    private captureFrame(): void {
        // This would capture the current GPU state for detailed analysis
        console.log('Frame captured for analysis');
    }

    /**
     * Dispose of resources
     */
    dispose(): void {
        this.gui?.destroy();

        // Remove debug panels
        const debugContainer = document.getElementById('raytracing-debug-container');
        if (debugContainer) {
            debugContainer.remove();
        }

        // Dispose GPU resources
        this.rayVertexBuffer?.destroy();
        this.rayIndexBuffer?.destroy();
        this.uniformBuffer?.destroy();

        this.performanceMonitor.stopMonitoring();

        console.log('Raytracing Debugger disposed');
    }
}
