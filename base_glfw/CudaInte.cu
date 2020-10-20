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
__global__ void test_simple_vbo_kernel(float* pos, unsigned int width, unsigned int height, float time, unsigned int vboStridInFloat)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    //check if out of the boundary
    if (x > width || y > height) return;
    float u = x / (float)width * 10.0f;
    float v = y / (float)height * 10.0f;

    float freq = 0.5f;
    float w = glm::sin(u * 1.3f*freq + time) * glm::cos(v * 1.7f*freq + time) * 0.5f;

    //write to pos
    pos[(x * height + y)* vboStridInFloat + 0] = u;
    pos[(x * height + y)* vboStridInFloat + 1] = w;
    pos[(x * height + y)* vboStridInFloat + 2] = v;
    //write to normal
    pos[(x * height + y)* vboStridInFloat + 5] = 1.0f;
    pos[(x * height + y)* vboStridInFloat + 6] = 1.0f;
    pos[(x * height + y)* vboStridInFloat + 7] = 1.0f;
    //uv generated in VBO already    

}

void test_launch_kernel(float* pos, const unsigned int mesh_width,
    const unsigned int mesh_height, float time, unsigned int vboStridInFloat)
{
    // execute the kernel
    dim3 block(32, 32, 1);
    dim3 grid(ceil(mesh_width / block.x), ceil(mesh_height / block.y), 1);
    test_simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time, vboStridInFloat);

    checkCudaError("simple_vbo_kernel launch fail ");

    cudaDeviceSynchronize();
}
