import { Room, RoomConfig } from "./room/room";
import { Camera } from "./camera/camera";
import { vec3 } from "gl-matrix";
import { Sphere } from "./objects/sphere";
import { InputHandler } from "./input/input-handler";
import { DebugUI } from "./debug/debug-ui";
import { RoomManager } from "./room/room-manager";
import { Renderer } from "./rendering/renderer";
import { RayVisualization } from "./visualization/ray-visualization";


export class Main {
  private camera: Camera;
  private sphere: Sphere;
  private roomConfig: RoomConfig;

  // Modular components
  private inputHandler: InputHandler;
  private debugUI: DebugUI;
  private roomManager: RoomManager;
  private renderer: Renderer;
  private rayVisualization: RayVisualization;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice, private adapter: GPUAdapter) {
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

    // Initialize modular components
    this.renderer = new Renderer(canvas, device);
    this.inputHandler = new InputHandler(this.camera);
    this.roomManager = new RoomManager(device, this.roomConfig, this.camera, this.sphere);
    this.rayVisualization = new RayVisualization(device);
    this.debugUI = new DebugUI(
      this.roomConfig,
      this.sphere,
      () => this.updateRoom(),
      device,
      (rays) => this.updateRayVisualization(rays)
    );

    // Initialize ray visualization
    this.rayVisualization.initialize();


  }





  private updateRoom(): void {
    // Update room using room manager
    this.roomManager.updateRoom();

    // Update debug UI ranges
    this.debugUI.updateSphereControlRanges();
  }

  private updateRayVisualization(rays: any[]): void {
    // Convert ray data to visualization format
    const rayData = rays.map(ray => {
      // Show ray as a line from current position in the direction it's traveling
      const visualLength = 2.0; // Fixed length to see direction

      return {
        start: ray.origin,
        end: [
          ray.origin[0] + ray.direction[0] * visualLength,
          ray.origin[1] + ray.direction[1] * visualLength,
          ray.origin[2] + ray.direction[2] * visualLength
        ],
        energy: ray.energy,
        bounceCount: ray.bounceCount
      };
    });

    // Update ray visualization
    this.rayVisualization.setRayData(rayData);
    console.log(`Updated ray visualization with ${rayData.length} rays - Ray at:`,
      rays.length > 0 ? rays[0].origin : 'none',
      'Energy:', rays.length > 0 ? rays[0].energy : 'none',
      'Bounces:', rays.length > 0 ? rays[0].bounceCount : 'none');
  }



  private handleInput(deltaTime: number): void {
    this.inputHandler.handleInput(deltaTime);
  }

  public render(deltaTime: number): void {
    // Handle input
    this.handleInput(deltaTime);

    // Render using the renderer component
    this.renderer.render(this.roomManager.getRoom(), this.camera, this.sphere, this.rayVisualization);
  }

  public resize(): void {
    this.renderer.resize();
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




