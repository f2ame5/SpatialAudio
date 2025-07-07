/**
 * Room Manager - Handles room creation, updates, and spatial constraints
 */

import { vec3 } from "gl-matrix";
import { Room, RoomConfig } from "./room";
import { Camera } from "../camera/camera";
import { Sphere } from "../objects/sphere";

export class RoomManager {
  private device: GPUDevice;
  private roomConfig: RoomConfig;
  private room: Room;
  private camera: Camera;
  private sphere: Sphere;

  constructor(
    device: GPUDevice,
    roomConfig: RoomConfig,
    camera: Camera,
    sphere: Sphere
  ) {
    this.device = device;
    this.roomConfig = roomConfig;
    this.camera = camera;
    this.sphere = sphere;
    this.room = new Room(device, roomConfig);
  }

  /**
   * Update room with new configuration
   */
  public updateRoom(): void {
    // Recreate room with new dimensions
    this.room = new Room(this.device, this.roomConfig);

    // Ensure camera stays within room bounds
    this.constrainCamera();

    // Keep sphere at current position unless it's outside new bounds
    const currentPos = this.sphere.getPosition();
    const validPos = this.room.getClosestValidPosition([
      currentPos[0],
      currentPos[1],
      currentPos[2],
    ]);
    this.sphere.setPosition(validPos);
  }

  /**
   * Constrain camera position to stay within room bounds
   */
  public constrainCamera(): void {
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

  /**
   * Get room bounds for calculations
   */
  public getRoomBounds(): { min: number[]; max: number[] } {
    const { width, height, depth } = this.roomConfig.dimensions;
    return {
      min: [-width / 2, 0, -depth / 2],
      max: [width / 2, height, depth / 2]
    };
  }

  /**
   * Check if a position is within room bounds
   */
  public isPositionInBounds(position: vec3, margin: number = 0): boolean {
    const bounds = this.getRoomBounds();
    return (
      position[0] >= bounds.min[0] + margin &&
      position[0] <= bounds.max[0] - margin &&
      position[1] >= bounds.min[1] + margin &&
      position[1] <= bounds.max[1] - margin &&
      position[2] >= bounds.min[2] + margin &&
      position[2] <= bounds.max[2] - margin
    );
  }

  /**
   * Get the closest valid position within room bounds
   */
  public getClosestValidPosition(position: number[]): vec3 {
    const bounds = this.getRoomBounds();
    const margin = 0.1;

    return vec3.fromValues(
      Math.max(bounds.min[0] + margin, Math.min(bounds.max[0] - margin, position[0])),
      Math.max(bounds.min[1] + margin, Math.min(bounds.max[1] - margin, position[1])),
      Math.max(bounds.min[2] + margin, Math.min(bounds.max[2] - margin, position[2]))
    );
  }

  /**
   * Calculate room volume
   */
  public getRoomVolume(): number {
    const { width, height, depth } = this.roomConfig.dimensions;
    return width * height * depth;
  }

  /**
   * Calculate room surface area
   */
  public getRoomSurfaceArea(): number {
    const { width, height, depth } = this.roomConfig.dimensions;
    return 2 * (width * height + width * depth + height * depth);
  }

  /**
   * Get room diagonal length
   */
  public getRoomDiagonal(): number {
    const { width, height, depth } = this.roomConfig.dimensions;
    return Math.sqrt(width * width + height * height + depth * depth);
  }

  /**
   * Get the room instance
   */
  public getRoom(): Room {
    return this.room;
  }

  /**
   * Get the room configuration
   */
  public getRoomConfig(): RoomConfig {
    return this.roomConfig;
  }

  /**
   * Update room configuration
   */
  public setRoomConfig(config: RoomConfig): void {
    this.roomConfig = config;
    this.updateRoom();
  }
}
