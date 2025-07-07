/**
 * Input Handler - Manages keyboard and mouse input for camera movement
 */

import { Camera } from "../camera/camera";

export class InputHandler {
  private keys: { [key: string]: boolean } = {};
  private camera: Camera;

  constructor(camera: Camera) {
    this.camera = camera;
    this.setupInputHandlers();
  }

  /**
   * Set up keyboard event listeners
   */
  private setupInputHandlers(): void {
    // Keyboard controls for movement
    window.addEventListener(
      "keydown",
      (e) => (this.keys[e.key.toLowerCase()] = true)
    );
    window.addEventListener(
      "keyup",
      (e) => (this.keys[e.key.toLowerCase()] = false)
    );

    // Arrow key rotation controls
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

  /**
   * Handle input for camera movement based on current key states
   */
  public handleInput(deltaTime: number): void {
    // Apply movement based on pressed keys
    if (this.keys["w"]) this.camera.moveForward(deltaTime);
    if (this.keys["s"]) this.camera.moveForward(-deltaTime);
    if (this.keys["a"]) this.camera.moveRight(-deltaTime);
    if (this.keys["d"]) this.camera.moveRight(deltaTime);
    if (this.keys[" "]) this.camera.moveUp(deltaTime);
    if (this.keys["shift"]) this.camera.moveUp(-deltaTime);
  }

  /**
   * Check if a specific key is currently pressed
   */
  public isKeyPressed(key: string): boolean {
    return this.keys[key.toLowerCase()] || false;
  }

  /**
   * Get all currently pressed keys
   */
  public getPressedKeys(): string[] {
    return Object.keys(this.keys).filter(key => this.keys[key]);
  }

  /**
   * Clean up event listeners
   */
  public dispose(): void {
    // Remove event listeners to prevent memory leaks
    window.removeEventListener("keydown", this.handleKeyDown);
    window.removeEventListener("keyup", this.handleKeyUp);
  }

  private handleKeyDown = (e: KeyboardEvent) => {
    this.keys[e.key.toLowerCase()] = true;
  };

  private handleKeyUp = (e: KeyboardEvent) => {
    this.keys[e.key.toLowerCase()] = false;
  };
}
