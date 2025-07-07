import { Room, RoomConfig } from "./room/room";
import { Camera } from "./camera/camera";
import { vec3, mat4 } from "gl-matrix";
import * as dat from "dat.gui";
import { Sphere } from "./objects/sphere";
import { SphereRenderer } from "./objects/sphere-renderer";
import { AcousticRaytracer } from "./audio/acoustic-raytracer";
import { RayDistributionType } from "./audio/ray-types";
import { SpatialAudioController } from "./audio/spatial-audio-controller";

export class Main {
  private canvas: HTMLCanvasElement;
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private room: Room;
  private camera: Camera;
  private depthTexture!: GPUTexture;
  private keys: { [key: string]: boolean } = {};
  private roomConfig: RoomConfig;
  private gui!: dat.GUI;
  private sphere: Sphere;
  private sphereRenderer: SphereRenderer;
  private sourceControllers!: {
    x: dat.GUIController;
    y: dat.GUIController;
    z: dat.GUIController;
  };
  private raytracer: AcousticRaytracer | null = null;

  // Ray visualization
  private rayVisualizationEnabled: boolean = false;
  private rayVertexBuffer: GPUBuffer | null = null;
  private rayRenderPipeline: GPURenderPipeline | null = null;
  private rayUniformBuffer: GPUBuffer | null = null;
  private rayBindGroup: GPUBindGroup | null = null;
  private currentRayData: any[] = [];
  private rayVisualizationMode: 'initial' | 'bounced' | 'full-path' = 'full-path';
  private spatialAudioController!: SpatialAudioController;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice, private adapter: GPUAdapter) {
    this.canvas = canvas;
    this.device = device;
    this.context = canvas.getContext("webgpu") as GPUCanvasContext;

    // Configure the canvas context
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "premultiplied",
    });

    // Initialize room config
    this.roomConfig = {
      dimensions: { width: 8, height: 3, depth: 5 },
      materials: {
        walls: {
          color: [0.8, 0.8, 0.8], // Light gray walls
        },
        ceiling: {
          color: [0.9, 0.9, 0.9], // White ceiling
        },
        floor: {
          color: [0.6, 0.4, 0.2], // Brown floor
        },
      },
    };

    // Initialize room
    this.room = new Room(device, this.roomConfig);

    // Initialize camera near the back wall of the room
    this.camera = new Camera(
      vec3.fromValues(0, 1.7, 3), // Start near back wall (Z=3 is inside Z=4 boundary)
      -90, // Looking toward the center
      0 // Level view
    );

    // Initialize sphere in the middle of the room
    this.sphere = new Sphere(
      vec3.fromValues(0, 1.7, 0), // Center of room, eye level
      0.2 // Smaller radius for sound source
    );

    // Initialize sphere renderer
    this.sphereRenderer = new SphereRenderer(device);

    // Setup debug UI
    this.setupDebugUI();

    // Initialize spatial audio controller
    this.spatialAudioController = new SpatialAudioController(
      device,
      adapter,
      this.room,
      this.camera,
      this.sphere,
      this.gui
    );

    // Setup input handlers
    this.setupInputHandlers();

    // Create depth texture
    this.createDepthTexture();

    // Initialize raytracing (async)
    this.initializeRaytracing();
  }



  private setupDebugUI(): void {
    this.gui = new dat.GUI();

    const roomFolder = this.gui.addFolder("Room Dimensions");
    roomFolder
      .add(this.roomConfig.dimensions, "width", 2, 20)
      .onChange(() => this.updateRoom());
    roomFolder
      .add(this.roomConfig.dimensions, "height", 2, 10)
      .onChange(() => this.updateRoom());
    roomFolder
      .add(this.roomConfig.dimensions, "depth", 2, 20)
      .onChange(() => this.updateRoom());

    // Create a data object for the sphere position
    const spherePosition = {
      x: 0,
      y: 1.7,
      z: 0,
    };

    // Update sphere controls to allow movement
    const sphereFolder = this.gui.addFolder("Sphere Position");

    // Store controller references for updating ranges
    this.sourceControllers = {
      x: sphereFolder
        .add(
          spherePosition,
          "x",
          -this.roomConfig.dimensions.width / 2,
          this.roomConfig.dimensions.width / 2
        )
        .onChange((value: number) => {
          const pos = this.sphere.getPosition();
          pos[0] = value;
          this.sphere.setPosition(pos);
        }),
      y: sphereFolder
        .add(spherePosition, "y", 0, this.roomConfig.dimensions.height)
        .onChange((value: number) => {
          const pos = this.sphere.getPosition();
          pos[1] = value;
          this.sphere.setPosition(pos);
        }),
      z: sphereFolder
        .add(
          spherePosition,
          "z",
          -this.roomConfig.dimensions.depth / 2,
          this.roomConfig.dimensions.depth / 2
        )
        .onChange((value: number) => {
          const pos = this.sphere.getPosition();
          pos[2] = value;
          this.sphere.setPosition(pos);
        }),
    };

    roomFolder.open();
    sphereFolder.open();
  }

  private updateRoom(): void {
    // Recreate room with new dimensions
    this.room = new Room(this.device, this.roomConfig);

    // Update spatial audio controller
    this.spatialAudioController.updateRoom(this.room);

    // Update raytracer room bounds
    this.updateRaytracerRoomBounds();

    // Ensure camera stays within room bounds
    this.constrainCamera();

    // Update source position slider ranges
    this.sourceControllers.x
      .min(-this.roomConfig.dimensions.width / 2)
      .max(this.roomConfig.dimensions.width / 2);
    this.sourceControllers.y.min(0).max(this.roomConfig.dimensions.height);
    this.sourceControllers.z
      .min(-this.roomConfig.dimensions.depth / 2)
      .max(this.roomConfig.dimensions.depth / 2);

    // Keep sphere at current position unless it's outside new bounds
    const currentPos = this.sphere.getPosition();
    const validPos = this.room.getClosestValidPosition([
      currentPos[0],
      currentPos[1],
      currentPos[2],
    ]); // Convert vec3 to tuple
    this.sphere.setPosition(validPos);
  }

  private constrainCamera(): void {
    const pos = this.camera.getPosition();
    const { width, height, depth } = this.roomConfig.dimensions;
    const halfWidth = width / 2;
    const halfDepth = depth / 2;
    const margin = 0.5; // Keep camera slightly away from walls

    // Constrain position
    const newPos = vec3.fromValues(
      Math.max(-halfWidth + margin, Math.min(halfWidth - margin, pos[0])),
      Math.max(margin, Math.min(height - margin, pos[1])),
      Math.max(-halfDepth + margin, Math.min(halfDepth - margin, pos[2]))
    );

    this.camera.setPosition(newPos);
  }

  private setupInputHandlers(): void {
    // Keyboard controls
    window.addEventListener(
      "keydown",
      (e) => (this.keys[e.key.toLowerCase()] = true)
    );
    window.addEventListener(
      "keyup",
      (e) => (this.keys[e.key.toLowerCase()] = false)
    );

    // New keyboard rotation controls
    window.addEventListener("keydown", (event) => {
      switch (event.key) {
        case "ArrowLeft":
          this.camera.rotateWithKeyboard("left");
          break;
        case "ArrowRight":
          this.camera.rotateWithKeyboard("right");
          break;
        case "ArrowUp":
          this.camera.rotateWithKeyboard("up");
          break;
        case "ArrowDown":
          this.camera.rotateWithKeyboard("down");
          break;
      }
    });
  }

  private createDepthTexture(): void {
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private handleInput(deltaTime: number): void {
    // Apply movement
    if (this.keys["w"]) this.camera.moveForward(deltaTime);
    if (this.keys["s"]) this.camera.moveForward(-deltaTime);
    if (this.keys["a"]) this.camera.moveRight(-deltaTime);
    if (this.keys["d"]) this.camera.moveRight(deltaTime);
    if (this.keys[" "]) this.camera.moveUp(deltaTime);
    if (this.keys["shift"]) this.camera.moveUp(-deltaTime);
  }

  public render(deltaTime: number): void {
    // Handle input
    this.handleInput(deltaTime);

    // Update spatial audio
    this.spatialAudioController.update(deltaTime);

    // Update room's view projection with camera
    const aspect = this.canvas.width / this.canvas.height;
    const viewProjection = this.camera.getViewProjection(aspect);
    this.device.queue.writeBuffer(
      this.room.getUniformBuffer(), // Use public method
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
    this.room.render(renderPass);

    // Render sphere
    this.sphereRenderer.render(
      renderPass,
      viewProjection as Float32Array,
      this.sphere.getPosition(),
      this.sphere.getRadius()
    );

    // Render rays if visualization is enabled
    if (this.rayVisualizationEnabled && this.rayRenderPipeline && this.rayVertexBuffer && this.currentRayData.length > 0) {
      this.renderRays(renderPass, viewProjection);
    }

    // End render pass and submit
    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

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
   * Initialize raytracing system
   */
  private async initializeRaytracing(): Promise<void> {
    try {
      this.raytracer = new AcousticRaytracer(this.device, this.adapter, {
        maxRays: 1024,
        maxBounces: 10,
        minEnergy: 0.001
      });

      await this.raytracer.initialize();

      // Set room bounds
      this.updateRaytracerRoomBounds();
      console.log('Raytracing system initialized');

      // Add raytracing test buttons to GUI
      const raytracingFolder = this.gui.addFolder('Raytracing Test');
      raytracingFolder.add({
        testRays: () => this.testRayGeneration()
      }, 'testRays').name('Test Ray Generation');
      raytracingFolder.add({
        testBouncing: () => this.testRayBouncing()
      }, 'testBouncing').name('Test Ray Bouncing');
      raytracingFolder.add({
        testPipeline: () => this.testCompletePipeline()
      }, 'testPipeline').name('Test Complete Pipeline');
      raytracingFolder.add({
        visualizeRays: () => this.toggleRayVisualization()
      }, 'visualizeRays').name('Toggle Ray Visualization');

      // Add visualization mode control
      const vizSettings = { mode: 'full-path' };
      raytracingFolder.add(vizSettings, 'mode', ['initial', 'bounced', 'full-path']).name('Visualization Mode')
        .onChange((value: string) => {
          this.rayVisualizationMode = value as 'initial' | 'bounced' | 'full-path';
          if (this.rayVisualizationEnabled) {
            this.updateRayVisualizationMode();
          }
        });
      raytracingFolder.open();

    } catch (error) {
      console.error('Failed to initialize raytracing:', error);
    }
  }

  /**
   * Test ray generation
   */
  private async testRayGeneration(): Promise<void> {
    if (!this.raytracer) {
      console.error('Raytracer not initialized');
      return;
    }

    try {
      const sourcePos = this.sphere.getPosition();
      console.log('Testing ray generation from position:', sourcePos);

      const rays = await this.raytracer.testRayGeneration(
        sourcePos,
        RayDistributionType.UNIFORM_SPHERE
      );

      console.log(`Generated ${rays.length} rays`);

      // Count active rays
      const activeRays = rays.filter(ray => ray.active > 0.5).length;
      console.log(`Active rays: ${activeRays}`);

      // Check energy distribution
      const totalEnergy = rays.reduce((sum, ray) => sum + ray.energy, 0);
      console.log(`Total energy: ${totalEnergy.toFixed(3)}`);

      // Analyze first few rays
      console.log('First 3 rays:');
      for (let i = 0; i < Math.min(3, rays.length); i++) {
        const ray = rays[i];
        console.log(`Ray ${i}:`, {
          origin: [ray.origin[0].toFixed(3), ray.origin[1].toFixed(3), ray.origin[2].toFixed(3)],
          direction: [ray.direction[0].toFixed(3), ray.direction[1].toFixed(3), ray.direction[2].toFixed(3)],
          energy: ray.energy.toFixed(3),
          phase: ray.phase.toFixed(3),
          active: ray.active,
          bounceCount: ray.bounceCount,
          frequencyEnergy: Array.from(ray.frequencyEnergy).map(e => e.toFixed(3))
        });
      }

      // Check direction distribution
      const directions = rays.slice(0, 10).map(ray =>
        Math.sqrt(ray.direction[0]**2 + ray.direction[1]**2 + ray.direction[2]**2)
      );
      console.log('Direction magnitudes (should be ~1.0):', directions.map(d => d.toFixed(3)));

    } catch (error) {
      console.error('Ray generation test failed:', error);
    }
  }

  /**
   * Test ray bouncing physics
   */
  private async testRayBouncing(): Promise<void> {
    if (!this.raytracer) {
      console.error('Raytracer not initialized');
      return;
    }

    try {
      const sourcePos = this.sphere.getPosition();
      console.log('Testing ray bouncing from position:', sourcePos);

      // First generate rays
      let rays = await this.raytracer.testRayGeneration(
        sourcePos,
        RayDistributionType.UNIFORM_SPHERE
      );

      console.log(`Initial: ${rays.length} rays generated`);

      // Now test bouncing for several iterations
      for (let bounce = 0; bounce < 5; bounce++) {
        console.log(`\n--- Bounce Iteration ${bounce + 1} ---`);

        // Run bouncing shader (this would need to be implemented in raytracer)
        // For now, let's simulate what should happen
        const activeRaysBefore = rays.filter(ray => ray.active > 0.5).length;
        console.log(`Active rays before bounce: ${activeRaysBefore}`);

        if (activeRaysBefore === 0) {
          console.log('No active rays remaining');
          break;
        }

        // Simulate bouncing (in real implementation, this would call GPU shader)
        rays = this.simulateRayBouncing(rays);

        const activeRaysAfter = rays.filter(ray => ray.active > 0.5).length;
        const avgBounces = rays.reduce((sum, ray) => sum + ray.bounceCount, 0) / rays.length;
        const totalEnergy = rays.reduce((sum, ray) => sum + ray.energy, 0);

        console.log(`Active rays after bounce: ${activeRaysAfter}`);
        console.log(`Average bounce count: ${avgBounces.toFixed(2)}`);
        console.log(`Total energy remaining: ${totalEnergy.toFixed(3)}`);

        // Show some example rays that bounced
        const bouncedRays = rays.filter(ray => ray.bounceCount > bounce).slice(0, 3);
        console.log(`Sample bounced rays:`, bouncedRays.map(ray => ({
          bounces: ray.bounceCount,
          energy: ray.energy.toFixed(3),
          position: Array.from(ray.origin).map(x => x.toFixed(2)),
          direction: Array.from(ray.direction).map(x => x.toFixed(2))
        })));
      }

    } catch (error) {
      console.error('Ray bouncing test failed:', error);
    }
  }

  /**
   * Get room bounds based on current room configuration
   */
  private getRoomBounds(): { min: number[]; max: number[] } {
    const { width, height, depth } = this.roomConfig.dimensions;
    return {
      min: [-width / 2, 0, -depth / 2],
      max: [width / 2, height, depth / 2]
    };
  }

  /**
   * Update raytracer with current room bounds
   */
  private updateRaytracerRoomBounds(): void {
    if (!this.raytracer) return;

    const bounds = this.getRoomBounds();
    this.raytracer.setRoomBounds({
      min: vec3.fromValues(bounds.min[0], bounds.min[1], bounds.min[2]),
      max: vec3.fromValues(bounds.max[0], bounds.max[1], bounds.max[2])
    });
  }

  /**
   * Calculate appropriate ray length based on room size
   */
  private calculateRayLength(): number {
    const { width, height, depth } = this.roomConfig.dimensions;
    const roomDiagonal = Math.sqrt(width * width + height * height + depth * depth);
    // Ray length should be about 1/4 to 1/3 of room diagonal for good visualization
    return Math.max(0.5, roomDiagonal * 0.25);
  }

  /**
   * Simulate ray bouncing (placeholder for actual GPU shader)
   */
  private simulateRayBouncing(rays: any[]): any[] {
    const roomBounds = this.getRoomBounds();

    return rays.map(ray => {
      if (ray.active < 0.5) return ray;

      // Simple ray-box intersection simulation
      const origin = ray.origin;
      const direction = ray.direction;

      // Find intersection with room bounds
      let tMin = Infinity;
      let hitNormal = [0, 0, 0];

      // Check each axis
      for (let axis = 0; axis < 3; axis++) {
        if (Math.abs(direction[axis]) > 0.001) {
          // Check positive face
          const tPos = (roomBounds.max[axis] - origin[axis]) / direction[axis];
          if (tPos > 0.001 && tPos < tMin) {
            const hitPoint = [
              origin[0] + direction[0] * tPos,
              origin[1] + direction[1] * tPos,
              origin[2] + direction[2] * tPos
            ];

            // Check if hit point is within face bounds
            let valid = true;
            for (let checkAxis = 0; checkAxis < 3; checkAxis++) {
              if (checkAxis !== axis) {
                if (hitPoint[checkAxis] < roomBounds.min[checkAxis] ||
                    hitPoint[checkAxis] > roomBounds.max[checkAxis]) {
                  valid = false;
                  break;
                }
              }
            }

            if (valid) {
              tMin = tPos;
              hitNormal = [0, 0, 0];
              hitNormal[axis] = -1; // Inward normal
            }
          }

          // Check negative face
          const tNeg = (roomBounds.min[axis] - origin[axis]) / direction[axis];
          if (tNeg > 0.001 && tNeg < tMin) {
            const hitPoint = [
              origin[0] + direction[0] * tNeg,
              origin[1] + direction[1] * tNeg,
              origin[2] + direction[2] * tNeg
            ];

            // Check if hit point is within face bounds
            let valid = true;
            for (let checkAxis = 0; checkAxis < 3; checkAxis++) {
              if (checkAxis !== axis) {
                if (hitPoint[checkAxis] < roomBounds.min[checkAxis] ||
                    hitPoint[checkAxis] > roomBounds.max[checkAxis]) {
                  valid = false;
                  break;
                }
              }
            }

            if (valid) {
              tMin = tNeg;
              hitNormal = [0, 0, 0];
              hitNormal[axis] = 1; // Inward normal
            }
          }
        }
      }

      if (tMin < Infinity) {
        // Ray hit a wall - calculate reflection
        const hitPoint = [
          origin[0] + direction[0] * tMin,
          origin[1] + direction[1] * tMin,
          origin[2] + direction[2] * tMin
        ];

        // Reflect direction: r = d - 2(d·n)n
        const dotProduct = direction[0] * hitNormal[0] +
                          direction[1] * hitNormal[1] +
                          direction[2] * hitNormal[2];

        const newDirection = [
          direction[0] - 2 * dotProduct * hitNormal[0],
          direction[1] - 2 * dotProduct * hitNormal[1],
          direction[2] - 2 * dotProduct * hitNormal[2]
        ];

        // Apply energy absorption (10% loss per bounce)
        const energyLoss = 0.1;
        const newEnergy = ray.energy * (1 - energyLoss);

        // Update ray
        return {
          ...ray,
          origin: hitPoint,
          direction: newDirection,
          energy: newEnergy,
          bounceCount: ray.bounceCount + 1,
          active: newEnergy > 0.01 ? 1 : 0 // Deactivate if energy too low
        };
      } else {
        // Ray didn't hit anything - deactivate
        return {
          ...ray,
          active: 0
        };
      }
    });
  }

  /**
   * Test complete raytracing pipeline
   */
  private async testCompletePipeline(): Promise<void> {
    if (!this.raytracer) {
      console.error('Raytracer not initialized');
      return;
    }

    try {
      console.log('=== COMPLETE RAYTRACING PIPELINE TEST ===');

      const sourcePos = this.sphere.getPosition();
      const listenerPos = this.camera.getPosition();

      console.log('Source position:', sourcePos);
      console.log('Listener position:', listenerPos);

      // Step 1: Generate rays
      console.log('\n1. RAY GENERATION');
      let rays = await this.raytracer.testRayGeneration(
        sourcePos,
        RayDistributionType.UNIFORM_SPHERE
      );

      const initialActiveRays = rays.filter(ray => ray.active > 0.5).length;
      const initialEnergy = rays.reduce((sum, ray) => sum + ray.energy, 0);
      console.log(`✓ Generated ${rays.length} rays (${initialActiveRays} active)`);
      console.log(`✓ Initial total energy: ${initialEnergy.toFixed(3)}`);

      // Step 2: Simulate bouncing
      console.log('\n2. RAY BOUNCING SIMULATION');
      const maxBounces = 10;
      const rayHistory = []; // Store ray states for visualization

      for (let bounce = 0; bounce < maxBounces; bounce++) {
        const activeRays = rays.filter(ray => ray.active > 0.5).length;
        if (activeRays === 0) break;

        rays = this.simulateRayBouncing(rays);

        const currentEnergy = rays.reduce((sum, ray) => sum + ray.energy, 0);
        const avgBounces = rays.reduce((sum, ray) => sum + ray.bounceCount, 0) / rays.length;

        console.log(`  Bounce ${bounce + 1}: ${activeRays} active rays, energy: ${currentEnergy.toFixed(3)}, avg bounces: ${avgBounces.toFixed(1)}`);

        // Store snapshot for visualization
        rayHistory.push({
          bounce: bounce + 1,
          rays: rays.filter(ray => ray.active > 0.5).slice(0, 100), // Store first 100 active rays
          totalEnergy: currentEnergy,
          activeCount: activeRays
        });
      }

      // Step 3: Simulate ray collection
      console.log('\n3. RAY COLLECTION SIMULATION');
      const collectedRays = this.simulateRayCollection(rays, listenerPos);

      console.log(`✓ Collected ${collectedRays.length} rays at listener`);

      if (collectedRays.length > 0) {
        const avgArrivalTime = collectedRays.reduce((sum, ray) => sum + ray.arrivalTime, 0) / collectedRays.length;
        const totalCollectedEnergy = collectedRays.reduce((sum, ray) => sum + ray.energy, 0);
        const avgBounces = collectedRays.reduce((sum, ray) => sum + ray.bounceCount, 0) / collectedRays.length;

        console.log(`✓ Average arrival time: ${(avgArrivalTime * 1000).toFixed(1)}ms`);
        console.log(`✓ Total collected energy: ${totalCollectedEnergy.toFixed(3)}`);
        console.log(`✓ Average bounces: ${avgBounces.toFixed(1)}`);

        // Analyze frequency response
        const frequencyEnergy = new Array(8).fill(0);
        collectedRays.forEach(ray => {
          for (let i = 0; i < 8; i++) {
            frequencyEnergy[i] += ray.frequencyEnergy[i];
          }
        });

        console.log('✓ Frequency energy distribution:',
          frequencyEnergy.map((e, i) => `${['125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz', '8kHz', '16kHz'][i]}: ${e.toFixed(3)}`));

        // Step 4: Generate impulse response
        console.log('\n4. IMPULSE RESPONSE GENERATION');
        const impulseResponse = this.generateImpulseResponse(collectedRays);
        console.log(`✓ Generated impulse response with ${impulseResponse.length} samples`);

        const maxAmplitude = Math.max(...impulseResponse.map(Math.abs));
        const rt60 = this.estimateRT60(impulseResponse);

        console.log(`✓ Peak amplitude: ${maxAmplitude.toFixed(3)}`);
        console.log(`✓ Estimated RT60: ${rt60.toFixed(2)}s`);

        // Store results for visualization
        (window as any).raytracingResults = {
          rayHistory,
          collectedRays,
          impulseResponse,
          metrics: {
            initialEnergy,
            collectedEnergy: totalCollectedEnergy,
            rt60,
            avgArrivalTime,
            frequencyEnergy
          }
        };

        console.log('\n✓ Pipeline test complete! Results stored in window.raytracingResults');

      } else {
        console.log('⚠ No rays reached the listener position');
      }

    } catch (error) {
      console.error('Complete pipeline test failed:', error);
    }
  }

  /**
   * Simulate ray collection at listener position
   */
  private simulateRayCollection(rays: any[], listenerPos: Float32Array | vec3): any[] {
    const listenerRadius = 0.2; // 20cm sphere
    const collectedRays = [];

    for (const ray of rays) {
      if (ray.active < 0.5) continue;

      // Check if ray passes near listener
      const rayToListener = [
        listenerPos[0] - ray.origin[0],
        listenerPos[1] - ray.origin[1],
        listenerPos[2] - ray.origin[2]
      ];

      const projectionLength = rayToListener[0] * ray.direction[0] +
                              rayToListener[1] * ray.direction[1] +
                              rayToListener[2] * ray.direction[2];

      if (projectionLength > 0) {
        const closestPoint = [
          ray.origin[0] + ray.direction[0] * projectionLength,
          ray.origin[1] + ray.direction[1] * projectionLength,
          ray.origin[2] + ray.direction[2] * projectionLength
        ];

        const distance = Math.sqrt(
          Math.pow(listenerPos[0] - closestPoint[0], 2) +
          Math.pow(listenerPos[1] - closestPoint[1], 2) +
          Math.pow(listenerPos[2] - closestPoint[2], 2)
        );

        if (distance <= listenerRadius) {
          // Calculate arrival time
          const pathLength = ray.pathLength || 0;
          const additionalDistance = projectionLength;
          const totalPath = pathLength + additionalDistance;
          const arrivalTime = totalPath / 343; // Speed of sound

          collectedRays.push({
            ...ray,
            arrivalTime,
            distanceToListener: distance
          });
        }
      }
    }

    return collectedRays;
  }

  /**
   * Generate impulse response from collected rays
   */
  private generateImpulseResponse(collectedRays: any[]): Float32Array {
    const sampleRate = 48000;
    const lengthSeconds = 2.0;
    const samples = Math.floor(lengthSeconds * sampleRate);
    const impulseResponse = new Float32Array(samples);

    for (const ray of collectedRays) {
      const sampleIndex = Math.floor(ray.arrivalTime * sampleRate);
      if (sampleIndex >= 0 && sampleIndex < samples) {
        // Apply distance attenuation
        const attenuation = 1.0 / (1.0 + ray.distanceToListener);
        impulseResponse[sampleIndex] += ray.energy * attenuation;
      }
    }

    return impulseResponse;
  }

  /**
   * Estimate RT60 from impulse response
   */
  private estimateRT60(impulseResponse: Float32Array): number {
    // Find peak
    let peak = 0;
    for (let i = 0; i < impulseResponse.length; i++) {
      peak = Math.max(peak, Math.abs(impulseResponse[i]));
    }

    if (peak === 0) return 0;

    // Find -60dB point (1/1000 of peak)
    const target = peak / 1000;

    for (let i = 0; i < impulseResponse.length; i++) {
      if (Math.abs(impulseResponse[i]) <= target) {
        return i / 48000; // Convert to seconds
      }
    }

    return 2.0; // Max length
  }

  /**
   * Toggle ray visualization
   */
  private async toggleRayVisualization(): Promise<void> {
    this.rayVisualizationEnabled = !this.rayVisualizationEnabled;

    if (this.rayVisualizationEnabled) {
      console.log('Ray visualization enabled');
      await this.initializeRayVisualization();

      // Get current ray data for visualization
      if (this.raytracer) {
        const sourcePos = this.sphere.getPosition();
        console.log('Generating rays for visualization from source:', sourcePos);

        // Generate fresh rays
        const rays = await this.raytracer.testRayGeneration(sourcePos, RayDistributionType.UNIFORM_SPHERE);
        console.log(`Generated ${rays.length} rays for visualization`);

        // Create visualization based on mode
        this.updateRayVisualizationData(rays, sourcePos);
        console.log(`Visualizing ${this.currentRayData.length} ray segments`);
      }
    } else {
      console.log('Ray visualization disabled');
      this.currentRayData = [];
    }
  }

  /**
   * Initialize ray visualization pipeline
   */
  private async initializeRayVisualization(): Promise<void> {
    if (this.rayRenderPipeline) return; // Already initialized

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
    this.rayUniformBuffer = this.device.createBuffer({
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
    this.rayBindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.rayUniformBuffer }
      }]
    });

    // Create pipeline
    this.rayRenderPipeline = this.device.createRenderPipeline({
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
   * Create ray visualization data with proper paths from source
   */
  private createRayVisualizationData(rays: any[], sourcePos: Float32Array | vec3): any[] {
    const visualRays = [];
    const maxRaysToShow = 100; // Limit for performance
    const rayLength = this.calculateRayLength(); // Length of each ray segment based on room size

    // Take a subset of rays for visualization
    const selectedRays = rays.filter(ray => ray.active > 0.5).slice(0, maxRaysToShow);

    for (const ray of selectedRays) {
      // Create ray starting from the actual source position
      const rayStart = [sourcePos[0], sourcePos[1], sourcePos[2]];

      // Calculate ray end point based on direction
      const rayEnd = [
        rayStart[0] + ray.direction[0] * rayLength,
        rayStart[1] + ray.direction[1] * rayLength,
        rayStart[2] + ray.direction[2] * rayLength
      ];

      // Create visualization ray data
      visualRays.push({
        start: rayStart,
        end: rayEnd,
        energy: ray.energy,
        bounceCount: ray.bounceCount,
        direction: ray.direction,
        active: ray.active
      });
    }

    console.log(`Created ${visualRays.length} ray segments from source position [${sourcePos[0].toFixed(2)}, ${sourcePos[1].toFixed(2)}, ${sourcePos[2].toFixed(2)}]`);
    return visualRays;
  }

  /**
   * Create bounced ray visualization showing current ray positions and directions
   */
  private createBouncedRayVisualization(rays: any[], sourcePos: Float32Array | vec3): any[] {
    const visualRays = [];
    const maxRaysToShow = 150; // Show more rays for bounced visualization
    const rayLength = this.calculateRayLength() * 0.75; // Slightly shorter for bounced rays

    // Take a subset of active rays for visualization
    const activeRays = rays.filter(ray => ray.active > 0.5);
    const selectedRays = activeRays.slice(0, maxRaysToShow);

    console.log(`Creating bounced ray visualization: ${selectedRays.length} active rays out of ${rays.length} total`);

    for (const ray of selectedRays) {
      // Use the ray's current position (after bounces) as start point
      const rayStart = [ray.origin[0], ray.origin[1], ray.origin[2]];

      // Calculate ray end point based on current direction
      const rayEnd = [
        rayStart[0] + ray.direction[0] * rayLength,
        rayStart[1] + ray.direction[1] * rayLength,
        rayStart[2] + ray.direction[2] * rayLength
      ];

      // Create visualization ray data
      visualRays.push({
        start: rayStart,
        end: rayEnd,
        energy: ray.energy,
        bounceCount: ray.bounceCount,
        direction: ray.direction,
        active: ray.active
      });
    }

    console.log(`Created ${visualRays.length} bounced ray segments`);
    return visualRays;
  }

  /**
   * Create full ray path visualization showing complete trajectories with all bounces
   */
  private createFullPathVisualization(rays: any[], sourcePos: Float32Array | vec3): any[] {
    const visualSegments = [];
    const maxRaysToShow = 50; // Fewer rays since each has multiple segments
    const roomBounds = this.getRoomBounds();

    // Take a subset of rays for visualization
    const selectedRays = rays.filter(ray => ray.active > 0.5).slice(0, maxRaysToShow);

    console.log(`Creating full path visualization for ${selectedRays.length} rays`);

    for (let rayIndex = 0; rayIndex < selectedRays.length; rayIndex++) {
      const ray = selectedRays[rayIndex];

      // Simulate the complete ray path from source to final position
      const rayPath = this.traceCompleteRayPath(ray, sourcePos, roomBounds);

      // Create line segments for each bounce
      for (let i = 0; i < rayPath.length - 1; i++) {
        const segmentStart = rayPath[i];
        const segmentEnd = rayPath[i + 1];

        // Color based on bounce number (progression from green to red)
        const bounceProgress = i / Math.max(rayPath.length - 2, 1);
        const color = [
          bounceProgress,           // Red increases with bounces
          1.0 - bounceProgress,     // Green decreases with bounces
          0.3                       // Blue constant
        ];

        visualSegments.push({
          start: segmentStart,
          end: segmentEnd,
          energy: ray.energy * Math.pow(0.9, i), // Energy decreases with each bounce
          bounceCount: i,
          rayIndex: rayIndex,
          segmentIndex: i
        });
      }
    }

    console.log(`Created ${visualSegments.length} path segments for ${selectedRays.length} rays`);
    return visualSegments;
  }

  /**
   * Trace complete ray path from source through all bounces
   */
  private traceCompleteRayPath(ray: any, sourcePos: Float32Array | vec3, roomBounds: any): number[][] {
    const path = [];
    const maxBounces = 15;

    // Start from source position
    let currentPos = [sourcePos[0], sourcePos[1], sourcePos[2]];
    let currentDir = [ray.direction[0], ray.direction[1], ray.direction[2]];
    let currentEnergy = ray.energy;

    path.push([...currentPos]);

    for (let bounce = 0; bounce < maxBounces && currentEnergy > 0.01; bounce++) {
      // Find intersection with room bounds
      const intersection = this.findRayIntersection(currentPos, currentDir, roomBounds);

      if (!intersection.hit) {
        break; // Ray escaped room
      }

      // Add intersection point to path
      path.push([...intersection.point]);

      // Calculate reflected direction
      const reflectedDir = this.calculateReflection(currentDir, intersection.normal);

      // Update for next iteration
      currentPos = [...intersection.point];
      currentDir = reflectedDir;
      currentEnergy *= 0.9; // 10% energy loss per bounce

      // Move slightly away from surface to avoid self-intersection
      const epsilon = 0.01;
      currentPos[0] += intersection.normal[0] * epsilon;
      currentPos[1] += intersection.normal[1] * epsilon;
      currentPos[2] += intersection.normal[2] * epsilon;
    }

    return path;
  }

  /**
   * Find ray intersection with room boundaries
   */
  private findRayIntersection(origin: number[], direction: number[], roomBounds: any): any {
    let closestT = Infinity;
    let closestPoint = [0, 0, 0];
    let closestNormal = [0, 0, 0];
    let hit = false;

    // Check each axis (X, Y, Z)
    for (let axis = 0; axis < 3; axis++) {
      if (Math.abs(direction[axis]) > 0.001) {
        // Check positive face
        const tPos = (roomBounds.max[axis] - origin[axis]) / direction[axis];
        if (tPos > 0.001 && tPos < closestT) {
          const hitPoint = [
            origin[0] + direction[0] * tPos,
            origin[1] + direction[1] * tPos,
            origin[2] + direction[2] * tPos
          ];

          // Check if hit point is within face bounds
          let valid = true;
          for (let checkAxis = 0; checkAxis < 3; checkAxis++) {
            if (checkAxis !== axis) {
              if (hitPoint[checkAxis] < roomBounds.min[checkAxis] ||
                  hitPoint[checkAxis] > roomBounds.max[checkAxis]) {
                valid = false;
                break;
              }
            }
          }

          if (valid) {
            closestT = tPos;
            closestPoint = hitPoint;
            closestNormal = [0, 0, 0];
            closestNormal[axis] = -1; // Inward normal
            hit = true;
          }
        }

        // Check negative face
        const tNeg = (roomBounds.min[axis] - origin[axis]) / direction[axis];
        if (tNeg > 0.001 && tNeg < closestT) {
          const hitPoint = [
            origin[0] + direction[0] * tNeg,
            origin[1] + direction[1] * tNeg,
            origin[2] + direction[2] * tNeg
          ];

          // Check if hit point is within face bounds
          let valid = true;
          for (let checkAxis = 0; checkAxis < 3; checkAxis++) {
            if (checkAxis !== axis) {
              if (hitPoint[checkAxis] < roomBounds.min[checkAxis] ||
                  hitPoint[checkAxis] > roomBounds.max[checkAxis]) {
                valid = false;
                break;
              }
            }
          }

          if (valid) {
            closestT = tNeg;
            closestPoint = hitPoint;
            closestNormal = [0, 0, 0];
            closestNormal[axis] = 1; // Inward normal
            hit = true;
          }
        }
      }
    }

    return {
      hit,
      point: closestPoint,
      normal: closestNormal,
      distance: closestT
    };
  }

  /**
   * Calculate reflection direction
   */
  private calculateReflection(incident: number[], normal: number[]): number[] {
    // r = d - 2(d·n)n
    const dotProduct = incident[0] * normal[0] + incident[1] * normal[1] + incident[2] * normal[2];

    return [
      incident[0] - 2 * dotProduct * normal[0],
      incident[1] - 2 * dotProduct * normal[1],
      incident[2] - 2 * dotProduct * normal[2]
    ];
  }

  /**
   * Update ray visualization data based on current mode
   */
  private updateRayVisualizationData(rays: any[], sourcePos: Float32Array | vec3): void {
    if (this.rayVisualizationMode === 'initial') {
      // Show rays from source position
      this.currentRayData = this.createRayVisualizationData(rays, sourcePos);
    } else if (this.rayVisualizationMode === 'bounced') {
      // Show bounced rays
      let bouncedRays = rays;
      for (let i = 0; i < 2; i++) {
        bouncedRays = this.simulateRayBouncing(bouncedRays);
      }
      this.currentRayData = this.createBouncedRayVisualization(bouncedRays, sourcePos);
    } else if (this.rayVisualizationMode === 'full-path') {
      // Show complete ray paths with all bounces
      this.currentRayData = this.createFullPathVisualization(rays, sourcePos);
    }
    this.updateRayVisualization();
  }

  /**
   * Update ray visualization mode
   */
  private updateRayVisualizationMode(): void {
    if (!this.raytracer) return;

    // Regenerate rays with new mode
    const sourcePos = this.sphere.getPosition();
    this.raytracer.testRayGeneration(sourcePos, RayDistributionType.UNIFORM_SPHERE)
      .then(rays => {
        this.updateRayVisualizationData(rays, sourcePos);
        console.log(`Switched to ${this.rayVisualizationMode} ray visualization mode`);
      });
  }

  /**
   * Update ray visualization data
   */
  private updateRayVisualization(): void {
    if (!this.rayVisualizationEnabled || this.currentRayData.length === 0) return;

    // Create vertex data for ray lines
    const vertices: number[] = [];

    for (const rayData of this.currentRayData) {
      // Use the pre-calculated start and end points
      const start = rayData.start;
      const end = rayData.end;

      // Color based on visualization mode and data
      let color = [1.0, 1.0, 1.0]; // Default white

      if (this.rayVisualizationMode === 'full-path') {
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
        const bounceNormalized = Math.min((rayData.bounceCount || 0) / 5.0, 1.0);

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

    console.log(`Generated ${vertices.length / 12} ray lines for rendering`); // 12 floats per line (2 vertices * 6 floats each)

    // Create or update vertex buffer
    if (this.rayVertexBuffer) {
      this.rayVertexBuffer.destroy();
    }

    this.rayVertexBuffer = this.device.createBuffer({
      size: vertices.length * 4, // 4 bytes per float
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });

    this.device.queue.writeBuffer(this.rayVertexBuffer, 0, new Float32Array(vertices));
  }

  /**
   * Render rays in the scene
   */
  private renderRays(renderPass: GPURenderPassEncoder, viewProjection: mat4): void {
    if (!this.rayRenderPipeline || !this.rayVertexBuffer || !this.rayBindGroup || !this.rayUniformBuffer) {
      return;
    }

    // Update uniform buffer with view-projection matrix
    const uniformData = new Float32Array(20); // 80 bytes / 4 bytes per float
    uniformData.set(viewProjection, 0); // mat4x4 (16 floats)
    uniformData[16] = 0.7; // opacity

    this.device.queue.writeBuffer(this.rayUniformBuffer, 0, uniformData);

    // Set pipeline and render rays
    renderPass.setPipeline(this.rayRenderPipeline);
    renderPass.setBindGroup(0, this.rayBindGroup);
    renderPass.setVertexBuffer(0, this.rayVertexBuffer);

    // Draw lines (2 vertices per ray)
    const vertexCount = this.currentRayData.length * 2;
    renderPass.draw(vertexCount);
  }


}




// Animation loop
let lastTime = 0;
function animate(main: Main, time: number) {
  const deltaTime = (time - lastTime) / 1000; // Convert to seconds
  lastTime = time;

  main.resize();
  main.render(deltaTime);
  requestAnimationFrame((time) => animate(main, time));
}

// Initialize and start
async function init() {
  const canvas = document.querySelector("canvas");
  if (!canvas) throw new Error("No canvas element found");

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found");

  const device = await adapter.requestDevice();
  const main = new Main(canvas, device, adapter);

  // Start animation loop
  requestAnimationFrame((time) => animate(main, time));
}

init().catch(console.error);
