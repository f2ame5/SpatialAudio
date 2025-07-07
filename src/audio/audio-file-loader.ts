/**
 * AudioFileLoader - Advanced audio file loading and management
 */

import { getSupportedAudioFormats } from './audio-utils';

export interface AudioFileInfo {
    url: string;
    format: string;
    size?: number;
    duration?: number;
    buffer?: AudioBuffer;
    loadProgress: number;
    isLoaded: boolean;
    error?: string;
}

export interface LoadOptions {
    priority?: 'high' | 'normal' | 'low';
    onProgress?: (progress: number) => void;
    onError?: (error: Error) => void;
}

export class AudioFileLoader {
    private context: AudioContext;
    private files: Map<string, AudioFileInfo> = new Map();
    private loadingQueue: Array<{ id: string; options: LoadOptions }> = [];
    private isLoading: boolean = false;
    private maxConcurrentLoads: number = 3;
    private currentLoads: number = 0;
    private supportedFormats: ReturnType<typeof getSupportedAudioFormats>;

    constructor(context: AudioContext) {
        this.context = context;
        this.supportedFormats = getSupportedAudioFormats();
    }

    /**
     * Add an audio file to the loader
     */
    addFile(id: string, url: string): AudioFileInfo {
        if (this.files.has(id)) {
            console.warn(`Audio file '${id}' already exists`);
            return this.files.get(id)!;
        }

        // Detect format from URL
        const format = this.getFormatFromUrl(url);
        
        // Check if format is supported
        if (!this.isFormatSupported(format)) {
            console.warn(`Audio format '${format}' may not be supported`);
        }

        const fileInfo: AudioFileInfo = {
            url,
            format,
            loadProgress: 0,
            isLoaded: false
        };

        this.files.set(id, fileInfo);
        return fileInfo;
    }

    /**
     * Load an audio file
     */
    async loadFile(id: string, options: LoadOptions = {}): Promise<AudioBuffer> {
        const fileInfo = this.files.get(id);
        if (!fileInfo) {
            throw new Error(`Audio file '${id}' not found`);
        }

        // Return cached buffer if already loaded
        if (fileInfo.isLoaded && fileInfo.buffer) {
            return fileInfo.buffer;
        }

        // Add to queue based on priority
        return new Promise((resolve, reject) => {
            const queueItem = {
                id,
                options: {
                    ...options,
                    onProgress: (progress: number) => {
                        fileInfo.loadProgress = progress;
                        options.onProgress?.(progress);
                    },
                    onError: (error: Error) => {
                        fileInfo.error = error.message;
                        options.onError?.(error);
                        reject(error);
                    }
                }
            };

            // Handle priority
            if (options.priority === 'high') {
                this.loadingQueue.unshift(queueItem);
            } else if (options.priority === 'low') {
                this.loadingQueue.push(queueItem);
            } else {
                // Normal priority - add to middle
                const midIndex = Math.floor(this.loadingQueue.length / 2);
                this.loadingQueue.splice(midIndex, 0, queueItem);
            }

            // Process queue
            this.processQueue();

            // Store resolve callback
            (queueItem as any).resolve = resolve;
        });
    }

    /**
     * Load multiple files
     */
    async loadFiles(
        ids: string[], 
        options: LoadOptions = {}
    ): Promise<Map<string, AudioBuffer>> {
        const buffers = new Map<string, AudioBuffer>();
        const totalFiles = ids.length;
        let loadedFiles = 0;

        const onProgress = options.onProgress;
        options.onProgress = (progress: number) => {
            const totalProgress = (loadedFiles + progress) / totalFiles;
            onProgress?.(totalProgress);
        };

        for (const id of ids) {
            try {
                const buffer = await this.loadFile(id, options);
                buffers.set(id, buffer);
                loadedFiles++;
            } catch (error) {
                console.error(`Failed to load file '${id}':`, error);
                if (options.onError) {
                    options.onError(error as Error);
                }
            }
        }

        return buffers;
    }

    /**
     * Preload all added files
     */
    async preloadAll(options: LoadOptions = {}): Promise<void> {
        const ids = Array.from(this.files.keys());
        await this.loadFiles(ids, options);
    }

    /**
     * Process the loading queue
     */
    private async processQueue(): Promise<void> {
        if (this.isLoading || this.loadingQueue.length === 0) {
            return;
        }

        if (this.currentLoads >= this.maxConcurrentLoads) {
            return;
        }

        this.isLoading = true;
        
        while (this.loadingQueue.length > 0 && this.currentLoads < this.maxConcurrentLoads) {
            const item = this.loadingQueue.shift()!;
            this.currentLoads++;
            
            this.loadFileInternal(item.id, item.options)
                .then((buffer) => {
                    (item as any).resolve?.(buffer);
                    this.currentLoads--;
                    this.processQueue();
                })
                .catch((error) => {
                    this.currentLoads--;
                    this.processQueue();
                });
        }
        
        this.isLoading = false;
    }

    /**
     * Internal file loading implementation
     */
    private async loadFileInternal(
        id: string, 
        options: LoadOptions
    ): Promise<AudioBuffer> {
        const fileInfo = this.files.get(id);
        if (!fileInfo) {
            throw new Error(`Audio file '${id}' not found`);
        }

        try {
            // Fetch with progress tracking
            const response = await this.fetchWithProgress(
                fileInfo.url, 
                options.onProgress
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Get file size
            const contentLength = response.headers.get('content-length');
            if (contentLength) {
                fileInfo.size = parseInt(contentLength, 10);
            }

            // Get array buffer
            const arrayBuffer = await response.arrayBuffer();
            
            // Decode audio data
            const audioBuffer = await this.context.decodeAudioData(arrayBuffer);
            
            // Update file info
            fileInfo.buffer = audioBuffer;
            fileInfo.duration = audioBuffer.duration;
            fileInfo.isLoaded = true;
            fileInfo.loadProgress = 1;
            
            console.log(`Loaded audio file '${id}': ${fileInfo.duration.toFixed(2)}s, ${audioBuffer.sampleRate}Hz`);
            
            return audioBuffer;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            fileInfo.error = errorMessage;
            throw new Error(`Failed to load audio file '${id}': ${errorMessage}`);
        }
    }

    /**
     * Fetch with progress tracking
     */
    private async fetchWithProgress(
        url: string, 
        onProgress?: (progress: number) => void
    ): Promise<Response> {
        const response = await fetch(url);
        
        if (!onProgress || !response.body) {
            return response;
        }

        // Clone response for progress tracking
        const contentLength = response.headers.get('content-length');
        if (!contentLength) {
            return response;
        }

        const total = parseInt(contentLength, 10);
        let loaded = 0;

        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];

        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            chunks.push(value);
            loaded += value.length;
            
            const progress = loaded / total;
            onProgress(progress);
        }

        // Combine chunks
        const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
        const combined = new Uint8Array(totalLength);
        let position = 0;
        
        for (const chunk of chunks) {
            combined.set(chunk, position);
            position += chunk.length;
        }

        // Create new response with combined data
        return new Response(combined, {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers
        });
    }

    /**
     * Get format from URL
     */
    private getFormatFromUrl(url: string): string {
        const extension = url.split('.').pop()?.toLowerCase() || '';
        const queryIndex = extension.indexOf('?');
        
        if (queryIndex !== -1) {
            return extension.substring(0, queryIndex);
        }
        
        return extension;
    }

    /**
     * Check if format is supported
     */
    private isFormatSupported(format: string): boolean {
        switch (format) {
            case 'mp3':
                return this.supportedFormats.mp3;
            case 'wav':
                return this.supportedFormats.wav;
            case 'ogg':
            case 'oga':
                return this.supportedFormats.ogg;
            case 'aac':
            case 'm4a':
                return this.supportedFormats.aac;
            case 'flac':
                return this.supportedFormats.flac;
            default:
                return false;
        }
    }

    /**
     * Get file info
     */
    getFileInfo(id: string): AudioFileInfo | undefined {
        return this.files.get(id);
    }

    /**
     * Get all file IDs
     */
    getFileIds(): string[] {
        return Array.from(this.files.keys());
    }

    /**
     * Remove a file
     */
    removeFile(id: string): void {
        this.files.delete(id);
        
        // Remove from loading queue
        this.loadingQueue = this.loadingQueue.filter(item => item.id !== id);
    }

    /**
     * Clear all files
     */
    clear(): void {
        this.files.clear();
        this.loadingQueue = [];
    }

    /**
     * Get loading progress for all files
     */
    getTotalProgress(): number {
        if (this.files.size === 0) return 0;
        
        let totalProgress = 0;
        for (const file of this.files.values()) {
            totalProgress += file.loadProgress;
        }
        
        return totalProgress / this.files.size;
    }

    /**
     * Get loaded buffers
     */
    getLoadedBuffers(): Map<string, AudioBuffer> {
        const buffers = new Map<string, AudioBuffer>();
        
        for (const [id, file] of this.files) {
            if (file.isLoaded && file.buffer) {
                buffers.set(id, file.buffer);
            }
        }
        
        return buffers;
    }

    /**
     * Set max concurrent loads
     */
    setMaxConcurrentLoads(max: number): void {
        this.maxConcurrentLoads = Math.max(1, max);
    }
}
