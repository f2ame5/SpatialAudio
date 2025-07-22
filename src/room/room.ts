import { mat4, vec3 } from 'gl-matrix';
import { WallMaterial } from './room-materials';
import { ROOM_PRESETS, RoomPreset } from './room-presets';

enum Surface {
    FLOOR = 0.0,
    CEILING = 1.0,
    WALL_FRONT_BACK = 2.0,
    WALL_LEFT_RIGHT = 3.0
}

export interface RoomConfig {
    dimensions: RoomDimensions;
    materials: RoomMaterials;
}

export interface RoomDimensions {
    width: number;
    height: number;
    depth: number;
}

export interface RoomMaterials {
    walls: WallMaterial;
    ceiling: WallMaterial;
    floor: WallMaterial;
}

// Add default materials
export const DEFAULT_WALL_MATERIAL: WallMaterial = {
    absorptionLow: 0.3,
    absorptionMid: 0.2,
    absorptionHigh: 0.1,
    scatteringLow: 0.5,
    scatteringMid: 0.5,
    scatteringHigh: 0.5,
    roughness: 0.1,
    phaseShift: 0.0,
    phaseRandomization: 0.0
};

const DEFAULT_ROOM_MATERIALS: RoomMaterials = {
    walls: DEFAULT_WALL_MATERIAL,
    ceiling: DEFAULT_WALL_MATERIAL, // Or a specific default for ceiling
    floor: DEFAULT_WALL_MATERIAL,   // Or a specific default for floor
};

export class Room {
    private device: GPUDevice;
    private pipeline!: GPURenderPipeline;
    private vertexBuffer!: GPUBuffer;
    private indexBuffer!: GPUBuffer;
    private uniformBuffer!: GPUBuffer;
    private uniformBindGroup!: GPUBindGroup;
    private projectionMatrix: mat4;
    private viewMatrix: mat4;
    public config: RoomConfig;

    constructor(device: GPUDevice, config: RoomConfig) {
        this.device = device;

        // Apply default materials if not provided
        this.config = {
            dimensions: config.dimensions,
            materials: { ...DEFAULT_ROOM_MATERIALS, ...config.materials }, // Merge defaults with provided
        };
        // Ensure that individual wall materials are also defaulted
        this.config.materials.walls = { ...DEFAULT_WALL_MATERIAL, ...this.config.materials.walls };
        this.config.materials.ceiling = { ...DEFAULT_WALL_MATERIAL, ...this.config.materials.ceiling };
        this.config.materials.floor = { ...DEFAULT_WALL_MATERIAL, ...this.config.materials.floor };

        this.projectionMatrix = mat4.create();
        this.viewMatrix = mat4.create();

        // Create uniform buffer for view-projection matrix
        this.uniformBuffer = device.createBuffer({
            size: 16 * 4, // 4x4 matrix of floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize buffers and pipeline
        this.initializeGeometry();
        this.createPipeline();

        // Create bind group after pipeline creation
        this.uniformBindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer,
                },
            }],
        });
    }

    /**
     * Apply a room preset to configure the room dimensions and materials
     * @param preset The room preset to apply
     */
    public applyPreset(preset: RoomPreset): void {
        // Update dimensions
        this.config.dimensions = { ...preset.dimensions };
        
        // Update materials, merging preset values with defaults
        this.config.materials = {
            walls: { ...DEFAULT_WALL_MATERIAL, ...preset.materials.walls },
            ceiling: { ...DEFAULT_WALL_MATERIAL, ...preset.materials.ceiling },
            floor: { ...DEFAULT_WALL_MATERIAL, ...preset.materials.floor }
        };

        // Reinitialize geometry with new dimensions
        this.initializeGeometry();
        
        // Update view-projection matrix to account for new dimensions
        this.updateViewProjection(this.canvasAspect);
        
        console.log(`Applied room preset: ${preset.name}`);
    }

    /**
     * Get available room presets
     * @returns Array of available room presets
     */
    public getPresets(): { [key: string]: RoomPreset } {
        return ROOM_PRESETS;
    }

    /**
     * Get a specific room preset by name
     * @param name The name of the preset to retrieve
     * @returns The room preset or undefined if not found
     */
    public getPreset(name: string): RoomPreset | undefined {
        return ROOM_PRESETS[name as keyof typeof ROOM_PRESETS];
    }

    // Store canvas aspect ratio for view projection updates
    private canvasAspect: number = 1.0;

    private initializeGeometry(): void {
        const { width, height, depth } = this.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;

        // Generate vertices for a proper square room, centered at origin
        const vertices = new Float32Array([
            // Position (XYZ) + Surface Type
            // Floor (Y=0)
            -halfWidth, 0, -halfDepth, Surface.FLOOR,
            halfWidth, 0, -halfDepth, Surface.FLOOR,
            halfWidth, 0, halfDepth, Surface.FLOOR,
            -halfWidth, 0, halfDepth, Surface.FLOOR,

            // Ceiling (Y=height)
            -halfWidth, height, -halfDepth, Surface.CEILING,
            halfWidth, height, -halfDepth, Surface.CEILING,
            halfWidth, height, halfDepth, Surface.CEILING,
            -halfWidth, height, halfDepth, Surface.CEILING,

            // Front wall (Z=halfDepth)
            -halfWidth, 0, halfDepth, Surface.WALL_FRONT_BACK,
            halfWidth, 0, halfDepth, Surface.WALL_FRONT_BACK,
            halfWidth, height, halfDepth, Surface.WALL_FRONT_BACK,
            -halfWidth, height, halfDepth, Surface.WALL_FRONT_BACK,

            // Back wall (Z=-halfDepth)
            -halfWidth, 0, -halfDepth, Surface.WALL_FRONT_BACK,
            halfWidth, 0, -halfDepth, Surface.WALL_FRONT_BACK,
            halfWidth, height, -halfDepth, Surface.WALL_FRONT_BACK,
            -halfWidth, height, -halfDepth, Surface.WALL_FRONT_BACK,

            // Left wall (X=-halfWidth)
            -halfWidth, 0, -halfDepth, Surface.WALL_LEFT_RIGHT,
            -halfWidth, 0, halfDepth, Surface.WALL_LEFT_RIGHT,
            -halfWidth, height, halfDepth, Surface.WALL_LEFT_RIGHT,
            -halfWidth, height, -halfDepth, Surface.WALL_LEFT_RIGHT,

            // Right wall (X=halfWidth)
            halfWidth, 0, -halfDepth, Surface.WALL_LEFT_RIGHT,
            halfWidth, 0, halfDepth, Surface.WALL_LEFT_RIGHT,
            halfWidth, height, halfDepth, Surface.WALL_LEFT_RIGHT,
            halfWidth, height, -halfDepth, Surface.WALL_LEFT_RIGHT,
        ]);

        // Define indices for drawing triangles
        const indices = new Uint16Array([
            // Floor
            0, 1, 2,    0, 2, 3,
            // Ceiling
            4, 6, 5,    4, 7, 6,  // Note: Changed winding order for ceiling
            // Front wall
            8, 9, 10,   8, 10, 11,
            // Back wall
            12, 14, 13, 12, 15, 14,
            // Left wall
            16, 17, 18, 16, 18, 19,
            // Right wall
            20, 22, 21, 20, 23, 22
        ]);

        // Create and initialize the vertex buffer
        this.vertexBuffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);

        // Create and initialize the index buffer
        this.indexBuffer = this.device.createBuffer({
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.indexBuffer, 0, indices);
    }

    private createPipeline(): void {
        const shaderModule = this.device.createShaderModule({
            code: `
                struct VertexInput {
                    @location(0) position: vec3f,
                    @location(1) surfaceType: f32
                };

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) surfaceType: f32,
                    @location(1) worldPos: vec3f
                };

                struct Uniforms {
                    viewProjectionMatrix: mat4x4f
                };

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;

                @vertex
                fn vertexMain(input: VertexInput) -> VertexOutput {
                    var output: VertexOutput;
                    output.position = uniforms.viewProjectionMatrix * vec4f(input.position, 1.0);
                    output.surfaceType = input.surfaceType;
                    output.worldPos = input.position;
                    return output;
                }

                @fragment
                fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                    // Using more neutral, muted colors
                    let floorColor = vec3f(0.6, 0.55, 0.5);     // Beige/wood tone
                    let ceilingColor = vec3f(0.95, 0.95, 0.95); // Off-white
                    let wallColorFB = vec3f(0.8, 0.8, 0.75);    // Light warm gray
                    let wallColorLR = vec3f(0.75, 0.75, 0.7);   // Slightly darker warm gray

                    var baseColor: vec3f;
                    if (input.surfaceType < 0.5) {          // FLOOR = 0.0
                        baseColor = floorColor;
                    } else if (input.surfaceType < 1.5) {   // CEILING = 1.0
                        baseColor = ceilingColor;
                    } else if (input.surfaceType < 2.5) {   // WALL_FRONT_BACK = 2.0
                        baseColor = wallColorFB;
                    } else {                                // WALL_LEFT_RIGHT = 3.0
                        baseColor = wallColorLR;
                    }

                    // Simple lighting
                    let lightPos = vec3f(2.0, 2.0, 2.0);
                    let lightDir = normalize(lightPos - input.worldPos);

                    // Calculate normal from world position derivatives
                    let normal = normalize(cross(
                        dpdx(input.worldPos),
                        dpdy(input.worldPos)
                    ));

                    // Basic lighting calculation with stronger ambient
                    let ambient = 0.7;  // Increased ambient light
                    let diffuse = max(dot(normal, lightDir), 0.0) * 0.3;
                    let finalColor = baseColor * (ambient + diffuse);

                    return vec4f(finalColor, 1.0);
                }
            `
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.device.createBindGroupLayout({
                entries: [{
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: 'uniform' }
                }]
            })]
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 16, // vec3 position (12 bytes) + float surfaceType (4 bytes)
                    attributes: [
                        {
                            format: 'float32x3',
                            offset: 0,
                            shaderLocation: 0  // position
                        },
                        {
                            format: 'float32',
                            offset: 12,
                            shaderLocation: 1  // surfaceType
                        }
                    ]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: 'bgra8unorm'
                }]
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none'
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            }
        });

        this.uniformBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer,
                }
            }]
        });
    }

    public updateDimensions(dimensions: RoomDimensions): void {
        this.config.dimensions = dimensions;
        this.initializeGeometry();
    }

    public updateMaterials(materials: RoomMaterials): void {
        this.config.materials = materials;
        // IMPORTANT:  We would need to update the GPU buffers here
        // that hold the material properties.  This is a key step
        // for dynamic material updates.  For simplicity, I'm omitting
        // the buffer update code here, but it's essential for a real
        // implementation.
    }

    public render(pass: GPURenderPassEncoder): void {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.uniformBindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.setIndexBuffer(this.indexBuffer, 'uint16');
        pass.drawIndexed(36, 1, 0, 0, 0);
    }

    public updateViewProjection(aspect: number): void {
        // Store aspect ratio for future use
        this.canvasAspect = aspect;
        
        // Create perspective projection matrix
        const fov = Math.PI / 4; // 45 degrees FOV
        mat4.perspective(this.projectionMatrix, fov, aspect, 0.1, 100.0);

        // Calculate camera position
        const { width, height, depth } = this.config.dimensions;
        const maxDim = Math.max(width, depth);
        const distance = maxDim * 1.5; // Camera distance from center

        // Position camera for a 3/4 view of the room
        const cameraPos = vec3.fromValues(
            distance * 0.8,  // X: Slightly right
            distance * 0.8,  // Y: Higher up for better floor visibility
            distance * 1.0   // Z: Further back for better perspective
        );

        const target = vec3.fromValues(0, 0, 0);  // Look at center of room
        const up = vec3.fromValues(0, 1, 0);      // Y-up orientation

        // Create view matrix
        mat4.lookAt(this.viewMatrix, cameraPos, target, up);

        // Combine view and projection matrices
        const viewProjection = mat4.create();
        mat4.multiply(viewProjection, this.projectionMatrix, this.viewMatrix);

        // Update uniform buffer with new matrix
        // Create a standard Float32Array with compatible ArrayBuffer
        const matrixValues = Array.isArray(viewProjection) ?
            viewProjection :
            Array.from(viewProjection);
        const matrixArray = new Float32Array(matrixValues);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, matrixArray);
    }

    public isPointInside(point: vec3): boolean {
        const halfWidth = this.config.dimensions.width / 2;
        const halfDepth = this.config.dimensions.depth / 2;

        return point[0] >= -halfWidth && point[0] <= halfWidth &&
               point[1] >= 0 && point[1] <= this.config.dimensions.height &&
               point[2] >= -halfDepth && point[2] <= halfDepth;
    }

    public getClosestValidPosition(point: vec3): vec3 {
        const halfWidth = this.config.dimensions.width / 2;
        const halfDepth = this.config.dimensions.depth / 2;

        return vec3.fromValues(
            Math.max(-halfWidth + 0.5, Math.min(halfWidth - 0.5, point[0])),
            Math.max(0.5, Math.min(this.config.dimensions.height - 0.5, point[1])),
            Math.max(-halfDepth + 0.5, Math.min(halfDepth - 0.5, point[2]))
        );
    }

    public getVolume(): number {
        return this.config.dimensions.width *
               this.config.dimensions.height *
               this.config.dimensions.depth;
    }

    public getSurfaceArea(): number {
        const w = this.config.dimensions.width;
        const h = this.config.dimensions.height;
        const d = this.config.dimensions.depth;

        // Calculate surface area of all walls, floor, and ceiling
        return 2 * (w * h + h * d + w * d);
    }

    public getTemperature(): number {
        // Default room temperature in Celsius
        return 20;
    }

    public getHumidity(): number {
        // Default relative humidity percentage
        return 50;
    }
}