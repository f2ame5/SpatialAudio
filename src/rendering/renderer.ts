/**
 * Renderer - Handles WebGPU rendering operations
 */

import { mat4 } from "gl-matrix";
import { Room } from "../room/room";
import { Camera } from "../camera/camera";
import { Sphere } from "../objects/sphere";
import { SphereRenderer } from "../objects/sphere-renderer";
import { RayVisualization } from "../visualization/ray-visualization";

export class Renderer {
  private canvas: HTMLCanvasElement;
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private depthTexture: GPUTexture;
  private sphereRenderer: SphereRenderer;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
    this.canvas = canvas;
    this.device = device;
    this.context = canvas.getContext("webgpu") as GPUCanvasContext;
    this.sphereRenderer = new SphereRenderer(device);

    // Configure the canvas context
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "premultiplied",
    });

    // Create initial depth texture
    this.createDepthTexture();
  }

  /**
   * Create depth texture for 3D rendering
   */
  private createDepthTexture(): void {
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  /**
   * Main render method
   */
  public render(room: Room, camera: Camera, sphere: Sphere, rayVisualization?: RayVisualization): void {
    // Update room's view projection with camera
    const aspect = this.canvas.width / this.canvas.height;
    const viewProjection = camera.getViewProjection(aspect);
    
    this.device.queue.writeBuffer(
      room.getUniformBuffer(),
      0,
      new Float32Array(viewProjection)
    );

    // Begin render pass
    const commandEncoder = this.device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    // Render room
    room.render(renderPass);

    // Render sphere
    this.sphereRenderer.render(
      renderPass,
      viewProjection as Float32Array,
      sphere.getPosition(),
      sphere.getRadius()
    );

    // Render rays if visualization is provided
    if (rayVisualization) {
      rayVisualization.render(renderPass, viewProjection as Float32Array);
    }

    // End render pass and submit
    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Handle canvas resize
   */
  public resize(): void {
    if (
      this.canvas.width !== this.canvas.clientWidth ||
      this.canvas.height !== this.canvas.clientHeight
    ) {
      this.canvas.width = this.canvas.clientWidth;
      this.canvas.height = this.canvas.clientHeight;

      // Recreate depth texture with new size
      this.depthTexture.destroy();
      this.createDepthTexture();
    }
  }

  /**
   * Get canvas dimensions
   */
  public getCanvasSize(): { width: number; height: number } {
    return {
      width: this.canvas.width,
      height: this.canvas.height
    };
  }

  /**
   * Get canvas aspect ratio
   */
  public getAspectRatio(): number {
    return this.canvas.width / this.canvas.height;
  }

  /**
   * Get the WebGPU device
   */
  public getDevice(): GPUDevice {
    return this.device;
  }

  /**
   * Get the canvas context
   */
  public getContext(): GPUCanvasContext {
    return this.context;
  }

  /**
   * Clean up resources
   */
  public dispose(): void {
    this.depthTexture.destroy();
  }
}
