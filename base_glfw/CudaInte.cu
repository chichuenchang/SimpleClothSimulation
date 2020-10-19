#include "CudaInte.cuh"

void checkCudaError(const char* msg)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s: %s. \n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/////////////////////////////////////////////////////////////////////////
//test
__global__ void test_simple_vbo_kernel(float* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    //check if out of the boundary
    if (x > width || y > height) return;
    float u = x / (float)width * 10.0f;
    float v = y / (float)height * 10.0f;

    float freq = 0.5f;
    float w = glm::sin(u * 1.3f*freq + time) * glm::cos(v * 1.7f*freq + time) * 0.5f;

    // write output vertex
    // each vbo component size = vec3 + vec2, let kernel write to the vec3 
    pos[(x * height + y)* 5 + 0] = u;
    pos[(x * height + y)* 5 + 1] = w;
    pos[(x * height + y)* 5 + 2] = v;

    //TODO: compute normal


}

void test_launch_kernel(float* pos, const unsigned int mesh_width,
    const unsigned int mesh_height, float time)
{
    // execute the kernel
    dim3 block(32, 32, 1);
    dim3 grid(ceil(mesh_width / block.x), ceil(mesh_height / block.y), 1);
    test_simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time);

    checkCudaError("simple_vbo_kernel launch fail ");

    cudaDeviceSynchronize();
}

///////////////////////////////////////////////////////////////////////
//OLD function
__global__ void simple_vbo_kernel(glm::vec3* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height) return;
    // calculate uv coordinates
    //float u = x / (float)width;
    //float v = y / (float)height;
    float u = x;
    float v = y;
    //u = u * 2.0f - 1.0f;
    //v = v * 2.0f - 1.0f;


    // calculate simple sine wave pattern
    float freq = 10.0f;
    float w = glm::sin(u * freq + time) * glm::cos(v * freq + time) * 0.5f;

    // write output vertex
    // vbo resource is now vec3 + vec2 each vertex, let kernel write to the vec3 
    //pos[y * width + x] = glm::vec3(u, w, v);
    //pos[x * height + y][0] = u;
    //pos[x * height + y][1] = w;
    //pos[x * height + y][2] = v;
    pos[x * height + y] = glm::vec3(u, w, v);

}

void launch_kernel(glm::vec3* pos, const unsigned int mesh_width,
    const unsigned int mesh_height, float time)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(ceil(mesh_width / block.x), ceil(mesh_height / block.y), 1);
    simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time);

    checkCudaError("simple_vbo_kernel launch fail ");

    cudaDeviceSynchronize();
}
