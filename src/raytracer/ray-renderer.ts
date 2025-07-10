import { vec3 } from 'gl-matrix';

export class RayRenderer {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline;
    private vertexBuffer: GPUBuffer;
    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;

    constructor(device: GPUDevice) {
        this.device = device;

        // Create pipeline with shader that renders rays with color based on energy
        const shader = this.device.createShaderModule({
            code: `
                struct Uniforms {
                    viewProjection: mat4x4f,
                };

                struct VertexInput {
                    @location(0) position: vec3f,
                    @location(1) energy: f32,
                };

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) energy: f32,
                };

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;

                @vertex
                fn vertexMain(input: VertexInput) -> VertexOutput {
                    var output: VertexOutput;
                    output.position = uniforms.viewProjection * vec4f(input.position, 1.0);
                    output.energy = input.energy;
                    return output;
                }

                @fragment
                fn fragmentMain(@location(0) energy: f32) -> @location(0) vec4f {
                    // Make rays invisible below threshold
                    if (energy < 0.05) {
                        return vec4f(0.0);
                    }
                    // Make rays more visible with higher alpha
                    let alpha = mix(0.3, 1.0, energy);
                    return vec4f(1.0, energy, 0.0, alpha);
                }
            `
        });

        // Create pipeline
        this.pipeline = device.createRenderPipeline({
            vertex: {
                module: shader,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 16, // vec3f position + f32 energy
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' },
                        { shaderLocation: 1, offset: 12, format: 'float32' }
                    ]
                }]
            },
            fragment: {
                module: shader,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: 'bgra8unorm',
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                    },
                }]
            },
            primitive: {
                topology: 'line-list'
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            },
            layout: 'auto'
        });

        // Create uniform buffer
        this.uniformBuffer = device.createBuffer({
            size: 64, // mat4x4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create bind group
        this.uniformBindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer }
            }]
        });

        // Create empty vertex buffer (will be updated with ray data)
        this.vertexBuffer = device.createBuffer({
            size: 1024, // Initial size, will be recreated as needed
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
    }

    public render(
        pass: GPURenderPassEncoder,
        viewProjection: Float32Array,
        rays: { origin: vec3, direction: vec3, energy: number }[],
        roomDimensions: { width: number, height: number, depth: number }
    ): void {
        // Update uniform buffer with view projection matrix
        this.device.queue.writeBuffer(this.uniformBuffer, 0, viewProjection);

        // Create vertex data for rays
        const vertices = new Float32Array(rays.length * 8); // 2 points per ray, 4 floats per point
        let vertexOffset = 0;

        const { width, height, depth } = roomDimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;

        for (const ray of rays) {
            // Start point
            vertices[vertexOffset++] = ray.origin[0];
            vertices[vertexOffset++] = ray.origin[1];
            vertices[vertexOffset++] = ray.origin[2];
            vertices[vertexOffset++] = ray.energy;

            // Calculate intersection with room boundaries
            let minT = Infinity;
            let validIntersection = false;

            // Check X planes (left/right walls)
            if (Math.abs(ray.direction[0]) > 0.0001) {
                const tx1 = (-halfWidth - ray.origin[0]) / ray.direction[0];
                const tx2 = (halfWidth - ray.origin[0]) / ray.direction[0];
                const tx = tx1 > 0 ? tx1 : tx2 > 0 ? tx2 : Infinity;
                if (tx > 0) {
                    minT = Math.min(minT, tx);
                    validIntersection = true;
                }
            }

            // Check Y planes (floor/ceiling)
            if (Math.abs(ray.direction[1]) > 0.0001) {
                const ty1 = (0 - ray.origin[1]) / ray.direction[1];  // Floor intersection
                const ty2 = (height - ray.origin[1]) / ray.direction[1];  // Ceiling intersection

                // Take the closest valid intersection
                let ty = Infinity;

                // Check both floor and ceiling intersections
                if (ty1 > 0 && ray.origin[1] + ty1 * ray.direction[1] <= height) {
                    ty = ty1;
                }
                if (ty2 > 0 && ray.origin[1] + ty2 * ray.direction[1] >= 0 && ty2 < ty) {
                    ty = ty2;
                }

                if (ty < Infinity) {
                    minT = Math.min(minT, ty);
                    validIntersection = true;
                }
            }

            // Check Z planes (front/back walls)
            if (Math.abs(ray.direction[2]) > 0.0001) {
                const tz1 = (-halfDepth - ray.origin[2]) / ray.direction[2];
                const tz2 = (halfDepth - ray.origin[2]) / ray.direction[2];
                const tz = tz1 > 0 ? tz1 : tz2 > 0 ? tz2 : Infinity;
                if (tz > 0) {
                    minT = Math.min(minT, tz);
                    validIntersection = true;
                }
            }

            // Use the intersection distance for ray length
            const rayLength = validIntersection ? minT : 50.0; // Use default length if no intersection
            const endPoint = vec3.scaleAndAdd(vec3.create(), ray.origin, ray.direction, rayLength);
            vertices[vertexOffset++] = endPoint[0];
            vertices[vertexOffset++] = endPoint[1];
            vertices[vertexOffset++] = endPoint[2];
            vertices[vertexOffset++] = ray.energy * 0.2;
        }

        // Recreate vertex buffer if needed
        if (this.vertexBuffer.size < vertices.byteLength) {
            this.vertexBuffer.destroy();
            this.vertexBuffer = this.device.createBuffer({
                size: vertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
        }

        // Update vertex buffer
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);

        // Draw rays
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.uniformBindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.draw(rays.length * 2, 1, 0, 0); // 2 vertices per ray
    }
}