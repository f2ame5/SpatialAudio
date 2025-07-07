/**
 * Ray Visualization - Handles ray rendering and visualization
 */

import { mat4 } from "gl-matrix";

export type RayVisualizationMode = 'initial' | 'bounced' | 'full-path';

export interface RayData {
  start: number[];
  end: number[];
  energy: number;
  bounceCount: number;
  direction?: number[];
  active?: number;
}

export class RayVisualization {
  private device: GPUDevice;
  private enabled: boolean = true; // Enable by default for testing
  private mode: RayVisualizationMode = 'full-path';
  
  // WebGPU resources
  private vertexBuffer: GPUBuffer | null = null;
  private renderPipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  
  // Ray data
  private currentRayData: RayData[] = [];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Initialize ray visualization pipeline
   */
  public async initialize(): Promise<void> {
    if (this.renderPipeline) return; // Already initialized

    // Ray visualization vertex shader
    const rayVertexShader = `
      struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) color: vec3<f32>,
      }

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec3<f32>,
      }

      struct Uniforms {
        viewProjection: mat4x4<f32>,
        opacity: f32,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      @vertex
      fn main(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.viewProjection * vec4<f32>(input.position, 1.0);
        output.color = input.color;
        return output;
      }
    `;

    // Ray visualization fragment shader
    const rayFragmentShader = `
      struct FragmentInput {
        @location(0) color: vec3<f32>,
      }

      @fragment
      fn main(input: FragmentInput) -> @location(0) vec4<f32> {
        return vec4<f32>(input.color, 0.7); // Semi-transparent rays
      }
    `;

    // Create shader modules
    const vertexModule = this.device.createShaderModule({ code: rayVertexShader });
    const fragmentModule = this.device.createShaderModule({ code: rayFragmentShader });

    // Create uniform buffer
    this.uniformBuffer = this.device.createBuffer({
      size: 80, // mat4x4 (64 bytes) + float (4 bytes) + padding (12 bytes)
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

    // Create bind group
    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.uniformBuffer }
      }]
    });

    // Create pipeline
    this.renderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      vertex: {
        module: vertexModule,
        entryPoint: 'main',
        buffers: [{
          arrayStride: 24, // 3 floats (position) + 3 floats (color)
          attributes: [
            { format: 'float32x3', offset: 0, shaderLocation: 0 }, // position
            { format: 'float32x3', offset: 12, shaderLocation: 1 }, // color
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
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus'
      }
    });

    console.log('Ray visualization pipeline initialized');
  }

  /**
   * Set ray data for visualization
   */
  public setRayData(rayData: RayData[]): void {
    this.currentRayData = rayData;
    this.updateVertexBuffer();
  }

  /**
   * Update vertex buffer with current ray data
   */
  private updateVertexBuffer(): void {
    if (!this.enabled || this.currentRayData.length === 0) return;

    // Create vertex data for ray lines
    const vertices: number[] = [];

    for (const rayData of this.currentRayData) {
      const start = rayData.start;
      const end = rayData.end;

      // Color based on visualization mode and data
      let color = [1.0, 1.0, 1.0]; // Default white

      if (this.mode === 'full-path') {
        // Color progression for full path: green -> yellow -> red
        const bounceProgress = Math.min(rayData.bounceCount / 10.0, 1.0);
        color = [
          0.2 + bounceProgress * 0.8,  // Red increases with bounces
          1.0 - bounceProgress * 0.5,  // Green decreases slower
          0.1                          // Blue low for visibility
        ];
      } else {
        // Original coloring for other modes
        const energyNormalized = Math.min(rayData.energy || 1.0, 1.0);
        const bounceNormalized = Math.min(rayData.bounceCount / 5.0, 1.0);

        color = [
          0.2 + bounceNormalized * 0.8, // Red increases with bounces
          energyNormalized,              // Green based on energy
          0.3                            // Blue constant
        ];
      }

      // Add line vertices (start and end)
      vertices.push(
        start[0], start[1], start[2], color[0], color[1], color[2],
        end[0], end[1], end[2], color[0], color[1], color[2]
      );
    }

    // Create or update vertex buffer
    if (this.vertexBuffer) {
      this.vertexBuffer.destroy();
    }

    this.vertexBuffer = this.device.createBuffer({
      size: vertices.length * 4, // 4 bytes per float
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });

    this.device.queue.writeBuffer(this.vertexBuffer, 0, new Float32Array(vertices));
  }

  /**
   * Render rays in the scene
   */
  public render(renderPass: GPURenderPassEncoder, viewProjection: mat4): void {
    if (!this.enabled || !this.renderPipeline || !this.vertexBuffer || 
        !this.bindGroup || !this.uniformBuffer || this.currentRayData.length === 0) {
      return;
    }

    // Update uniform buffer with view-projection matrix
    const uniformData = new Float32Array(20); // 80 bytes / 4 bytes per float
    uniformData.set(viewProjection, 0); // mat4x4 (16 floats)
    uniformData[16] = 0.7; // opacity

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Set pipeline and render rays
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.bindGroup);
    renderPass.setVertexBuffer(0, this.vertexBuffer);

    // Draw lines (2 vertices per ray)
    const vertexCount = this.currentRayData.length * 2;
    renderPass.draw(vertexCount);
  }

  /**
   * Toggle ray visualization
   */
  public toggle(): void {
    this.enabled = !this.enabled;
    if (this.enabled) {
      this.updateVertexBuffer();
    }
  }

  /**
   * Set visualization mode
   */
  public setMode(mode: RayVisualizationMode): void {
    this.mode = mode;
    this.updateVertexBuffer();
  }

  /**
   * Check if visualization is enabled
   */
  public isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Get current mode
   */
  public getMode(): RayVisualizationMode {
    return this.mode;
  }

  /**
   * Clean up resources
   */
  public dispose(): void {
    if (this.vertexBuffer) {
      this.vertexBuffer.destroy();
    }
    if (this.uniformBuffer) {
      this.uniformBuffer.destroy();
    }
  }
}
