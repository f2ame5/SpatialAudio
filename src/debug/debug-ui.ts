/**
 * Debug UI - Manages dat.GUI interface for room and sphere controls
 */

import * as dat from "dat.gui";
import { RoomConfig } from "../room/room";
import { Sphere } from "../objects/sphere";
import { AcousticRaytracer } from "../audio/acoustic-raytracer";
import { RoomAcoustics } from "../room/room-acoustics";
import { vec3 } from "gl-matrix";

export interface SourceControllers {
  x: dat.GUIController;
  y: dat.GUIController;
  z: dat.GUIController;
}

export class DebugUI {
  private gui: dat.GUI;
  private roomConfig: RoomConfig;
  private sphere: Sphere;
  private sourceControllers: SourceControllers;
  private onRoomUpdate: () => void;
  private device: GPUDevice;
  private raytracer: AcousticRaytracer | null = null;
  private roomAcoustics: RoomAcoustics | null = null;
  private onRayUpdate?: (rays: any[]) => void;
  private raySimulationActive = false;
  private raySimulationInterval: number | null = null;

  constructor(
    roomConfig: RoomConfig,
    sphere: Sphere,
    onRoomUpdate: () => void,
    device: GPUDevice,
    onRayUpdate?: (rays: any[]) => void
  ) {
    this.roomConfig = roomConfig;
    this.sphere = sphere;
    this.onRoomUpdate = onRoomUpdate;
    this.device = device;
    this.onRayUpdate = onRayUpdate;
    this.gui = new dat.GUI();
    this.setupDebugUI();
    this.initializeAcoustics();
  }

  /**
   * Set up the debug UI with room and sphere controls
   */
  private setupDebugUI(): void {
    this.setupRoomControls();
    this.setupSphereControls();
    this.setupAudioControls();
  }

  /**
   * Set up room dimension controls
   */
  private setupRoomControls(): void {
    const roomFolder = this.gui.addFolder("Room Dimensions");
    
    roomFolder
      .add(this.roomConfig.dimensions, "width", 2, 20)
      .onChange(() => {
        this.onRoomUpdate();
        this.updateSphereControlRanges();
      });
    
    roomFolder
      .add(this.roomConfig.dimensions, "height", 2, 10)
      .onChange(() => {
        this.onRoomUpdate();
        this.updateSphereControlRanges();
      });
    
    roomFolder
      .add(this.roomConfig.dimensions, "depth", 2, 20)
      .onChange(() => {
        this.onRoomUpdate();
        this.updateSphereControlRanges();
      });

    roomFolder.open();
  }

  /**
   * Set up sphere position controls
   */
  private setupSphereControls(): void {
    // Create a data object for the sphere position
    const spherePosition = {
      x: 0,
      y: 1.7,
      z: 0,
    };

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

    sphereFolder.open();
  }

  /**
   * Initialize acoustic components
   */
  private async initializeAcoustics(): Promise<void> {
    try {
      // Create room acoustics
      this.roomAcoustics = new RoomAcoustics(
        { config: this.roomConfig } as any, // Simple room interface
        { temperature: 20, humidity: 50, airPressure: 101325 }
      );

      // Create acoustic raytracer
      this.raytracer = new AcousticRaytracer(this.device, this.roomAcoustics);
      await this.raytracer.initialize();

      console.log('Acoustic components initialized');
    } catch (error) {
      console.error('Failed to initialize acoustic components:', error);
    }
  }

  /**
   * Set up audio and raytracing controls
   */
  private setupAudioControls(): void {
    const audioFolder = this.gui.addFolder("Spatial Audio");

    // Create actions object for button callbacks
    const actions = {
      calculateIR: async () => {
        if (!this.raytracer) {
          console.error("Raytracer not initialized");
          alert("Raytracer not initialized. Please wait and try again.");
          return;
        }

        try {
          const sourcePosition = this.sphere.getPosition();
          const listenerPosition = vec3.fromValues(-2, 1.7, 2);

          // Run raytracing simulation
          const rays = await this.raytracer.simulateRays(
            vec3.fromValues(sourcePosition[0], sourcePosition[1], sourcePosition[2]),
            listenerPosition
          );

          // Update visualization if callback provided
          if (this.onRayUpdate) {
            this.onRayUpdate(rays);
          }

        } catch (error) {
          console.error("Ray simulation failed:", error);
          alert(`Ray simulation failed: ${error}`);
        }
      },

      startContinuousSimulation: () => {
        if (this.raySimulationActive) return;

        this.raySimulationActive = true;
        this.raySimulationInterval = window.setInterval(async () => {
          if (!this.raytracer || !this.raySimulationActive) return;

          try {
            const sourcePosition = this.sphere.getPosition();
            const listenerPosition = vec3.fromValues(-2, 1.7, 2);

            const rays = await this.raytracer.simulateRays(
              vec3.fromValues(sourcePosition[0], sourcePosition[1], sourcePosition[2]),
              listenerPosition
            );

            if (this.onRayUpdate) {
              this.onRayUpdate(rays);
            }
          } catch (error) {
            console.error("Continuous ray simulation failed:", error);
            this.stopContinuousSimulation();
          }
        }, 100); // Update every 100ms
      },

      stopContinuousSimulation: () => {
        this.raySimulationActive = false;
        if (this.raySimulationInterval) {
          clearInterval(this.raySimulationInterval);
          this.raySimulationInterval = null;
        }
      }
    };

    // Add buttons
    audioFolder.add(actions, "calculateIR").name("Calculate IR");
    audioFolder.add(actions, "startContinuousSimulation").name("Start Animation");
    audioFolder.add(actions, "stopContinuousSimulation").name("Stop Animation");

    audioFolder.open();
  }

  /**
   * Stop continuous simulation
   */
  private stopContinuousSimulation(): void {
    this.raySimulationActive = false;
    if (this.raySimulationInterval) {
      clearInterval(this.raySimulationInterval);
      this.raySimulationInterval = null;
    }
  }

  /**
   * Cleanup method
   */
  destroy(): void {
    this.stopContinuousSimulation();
    this.gui.destroy();
  }

  /**
   * Update sphere control ranges when room dimensions change
   */
  public updateSphereControlRanges(): void {
    this.sourceControllers.x
      .min(-this.roomConfig.dimensions.width / 2)
      .max(this.roomConfig.dimensions.width / 2);
    
    this.sourceControllers.y
      .min(0)
      .max(this.roomConfig.dimensions.height);
    
    this.sourceControllers.z
      .min(-this.roomConfig.dimensions.depth / 2)
      .max(this.roomConfig.dimensions.depth / 2);
  }

  /**
   * Add a new folder to the GUI
   */
  public addFolder(name: string): dat.GUI {
    return this.gui.addFolder(name);
  }

  /**
   * Get the main GUI instance
   */
  public getGUI(): dat.GUI {
    return this.gui;
  }

  /**
   * Clean up the GUI
   */
  public dispose(): void {
    this.gui.destroy();
  }
}
