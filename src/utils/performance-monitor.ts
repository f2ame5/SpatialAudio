/**
 * Performance Monitor - Tracks frame times and performance metrics
 */

export interface PerformanceMetrics {
    frameTime: number;
    fps: number;
    averageFrameTime: number;
    averageFps: number;
    minFrameTime: number;
    maxFrameTime: number;
    gpuTime?: number;
    audioLatency?: number;
}

export class PerformanceMonitor {
    private frameTimes: number[] = [];
    private lastTime: number = performance.now();
    private maxSamples: number = 120; // Store last 2 seconds at 60fps
    private gpuTimer: GPUQuerySet | null = null;
    private gpuBuffer: GPUBuffer | null = null;
    private metrics: PerformanceMetrics = {
        frameTime: 0,
        fps: 0,
        averageFrameTime: 0,
        averageFps: 0,
        minFrameTime: Infinity,
        maxFrameTime: 0
    };

    /**
     * Initialize GPU timing (if supported)
     */
    async initializeGPUTiming(device: GPUDevice): Promise<boolean> {
        // Check if timestamp queries are supported
        if (!device.features.has('timestamp-query')) {
            console.warn('GPU timestamp queries not supported');
            return false;
        }

        try {
            // Create query set for GPU timing
            this.gpuTimer = device.createQuerySet({
                type: 'timestamp',
                count: 2
            });

            // Create buffer for reading back results
            this.gpuBuffer = device.createBuffer({
                size: 16, // 2 timestamps * 8 bytes each
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
            });

            return true;
        } catch (error) {
            console.error('Failed to initialize GPU timing:', error);
            return false;
        }
    }

    /**
     * Start frame measurement
     */
    startFrame(): void {
        this.lastTime = performance.now();
    }

    /**
     * End frame measurement and update metrics
     */
    endFrame(): void {
        const currentTime = performance.now();
        const frameTime = currentTime - this.lastTime;

        // Add to frame times array
        this.frameTimes.push(frameTime);
        if (this.frameTimes.length > this.maxSamples) {
            this.frameTimes.shift();
        }

        // Update current metrics
        this.metrics.frameTime = frameTime;
        this.metrics.fps = frameTime > 0 ? 1000 / frameTime : 0;

        // Update min/max
        this.metrics.minFrameTime = Math.min(this.metrics.minFrameTime, frameTime);
        this.metrics.maxFrameTime = Math.max(this.metrics.maxFrameTime, frameTime);

        // Calculate averages
        if (this.frameTimes.length > 0) {
            const sum = this.frameTimes.reduce((a, b) => a + b, 0);
            this.metrics.averageFrameTime = sum / this.frameTimes.length;
            this.metrics.averageFps = 1000 / this.metrics.averageFrameTime;
        }
    }

    /**
     * Mark GPU command start
     */
    markGPUStart(encoder: GPUCommandEncoder): void {
        if (this.gpuTimer) {
            // GPU timestamp queries require special handling
            // For now, we'll skip this functionality
            // encoder.writeTimestamp(this.gpuTimer, 0);
        }
    }

    /**
     * Mark GPU command end
     */
    markGPUEnd(encoder: GPUCommandEncoder): void {
        if (this.gpuTimer) {
            // GPU timestamp queries require special handling
            // For now, we'll skip this functionality
            // encoder.writeTimestamp(this.gpuTimer, 1);
        }
    }

    /**
     * Resolve GPU timing results
     */
    async resolveGPUTiming(device: GPUDevice): Promise<number | null> {
        if (!this.gpuTimer || !this.gpuBuffer) {
            return null;
        }

        // TODO: Implement GPU timing resolution
        // This requires reading back the query results
        return null;
    }

    /**
     * Update audio latency measurement
     */
    updateAudioLatency(latency: number): void {
        this.metrics.audioLatency = latency;
    }

    /**
     * Get current metrics
     */
    getMetrics(): PerformanceMetrics {
        return { ...this.metrics };
    }

    /**
     * Get metrics as formatted string
     */
    getMetricsString(): string {
        const m = this.metrics;
        return `FPS: ${m.fps.toFixed(1)} (avg: ${m.averageFps.toFixed(1)}) | ` +
               `Frame: ${m.frameTime.toFixed(2)}ms (avg: ${m.averageFrameTime.toFixed(2)}ms) | ` +
               `Range: ${m.minFrameTime.toFixed(2)}-${m.maxFrameTime.toFixed(2)}ms` +
               (m.gpuTime ? ` | GPU: ${m.gpuTime.toFixed(2)}ms` : '') +
               (m.audioLatency ? ` | Audio: ${m.audioLatency.toFixed(2)}ms` : '');
    }

    /**
     * Reset all metrics
     */
    reset(): void {
        this.frameTimes = [];
        this.metrics = {
            frameTime: 0,
            fps: 0,
            averageFrameTime: 0,
            averageFps: 0,
            minFrameTime: Infinity,
            maxFrameTime: 0,
            gpuTime: this.metrics.gpuTime,
            audioLatency: this.metrics.audioLatency
        };
    }

    /**
     * Set maximum number of samples to store
     */
    setMaxSamples(samples: number): void {
        this.maxSamples = Math.max(1, samples);
        while (this.frameTimes.length > this.maxSamples) {
            this.frameTimes.shift();
        }
    }

    /**
     * Get frame time history
     */
    getFrameTimeHistory(): number[] {
        return [...this.frameTimes];
    }

    /**
     * Check if performance is below target
     */
    isBelowTargetFPS(targetFPS: number): boolean {
        return this.metrics.averageFps < targetFPS;
    }

    /**
     * Get performance rating (0-1)
     */
    getPerformanceRating(targetFPS: number = 60): number {
        return Math.min(1, this.metrics.averageFps / targetFPS);
    }
}

// Export singleton instance
export const performanceMonitor = new PerformanceMonitor();
