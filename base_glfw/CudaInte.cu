//TODO
//1. pingpong buffer
//2. pass variables to kernel with constant

#include "CudaInte.cuh"

__constant__ 
ClothConstant clothConst;
__constant__
FixedClothConstant fxConst;


void CheckCudaErr(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        printf("\ncustomized check error///////////////////////////////////////////////\n");
        fprintf(stderr, "CUDA error: %s: %s. \n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void updateClothConst(ClothConstant *in_passVar) {

    cudaMemcpyToSymbol(clothConst, in_passVar, sizeof(ClothConstant));
    CheckCudaErr("cloth constant memory copy fail");
}

void copyFixClothConst(FixedClothConstant* in_fxConst) {

    cudaMemcpyToSymbol(fxConst, in_fxConst, sizeof(FixedClothConstant));
    CheckCudaErr("fixed constant memory copy fail");
}

__device__
glm::vec3 readFromVBO(float* d_vboPtr, unsigned int ind_x, unsigned int ind_y,
    const int offsetInEachVBOElement) {

    float x = d_vboPtr[(ind_x * fxConst.height + ind_y) * fxConst.vboStrdFlt + offsetInEachVBOElement + 0];
    float y = d_vboPtr[(ind_x * fxConst.height + ind_y) * fxConst.vboStrdFlt + offsetInEachVBOElement + 1];
    float z = d_vboPtr[(ind_x * fxConst.height + ind_y) * fxConst.vboStrdFlt + offsetInEachVBOElement + 2];

    return glm::vec3(x, y, z);
}

__device__ 
void writeToVBO(glm::vec3 outPos, float* d_vboPtr,
    unsigned int ind_x, unsigned int ind_y, const int offsetInEachVBOElement) {

    d_vboPtr[(ind_x * fxConst.height + ind_y) * fxConst.vboStrdFlt + offsetInEachVBOElement + 0] = outPos.x;
    d_vboPtr[(ind_x * fxConst.height + ind_y) * fxConst.vboStrdFlt + offsetInEachVBOElement + 1] = outPos.y;
    d_vboPtr[(ind_x * fxConst.height + ind_y) * fxConst.vboStrdFlt + offsetInEachVBOElement + 2] = outPos.z;
}

__device__
float length(glm::vec3 a, glm::vec3 b) {
    return glm::length(b -a );
}

__device__//normal is here also
glm::vec3 computeInnerForce(float* vboPtr, unsigned int x, unsigned int y) {
    //                |y-
    //                |
    //                |
    //          ------*-------x+
    //                |
    //                |
    //                |y+
    
    glm::vec3 pos, posL, posR, posU, posD, posLU, posLD, posRU, posRD, posL2,
        posR2, posU2, posD2;
    pos = readFromVBO(vboPtr, x, y, fxConst.OffstPos);
    //if neighbor point out of bound, neighbor point = center point
    //structure neighbor
    if ((x) < 1) { posL = pos; }
    else { posL = readFromVBO(vboPtr, x - 1, y + 0, fxConst.OffstPos); }
    if ((x + 1) > fxConst.width - 1) { posR = pos; }
    else { posR = readFromVBO(vboPtr, x + 1, y + 0, fxConst.OffstPos); }
    if ((y) < 1) { posU = pos; }
    else { posU = readFromVBO(vboPtr, x + 0, y - 1, fxConst.OffstPos); }
    if ((y + 1) > fxConst.height - 1) { posD = pos; }
    else { posD = readFromVBO(vboPtr, x + 0, y + 1, fxConst.OffstPos); }
    //shear neighbor
    if ((x) < 1 || (y) < 1) { posLU = pos; }
    else { posLU = readFromVBO(vboPtr, x - 1, y - 1, fxConst.OffstPos); }
    if ((x) < 1 || (y + 1) > fxConst.height - 1) { posLD = pos; }
    else { posLD = readFromVBO(vboPtr, x - 1, y + 1, fxConst.OffstPos); }
    if ((x + 1) > fxConst.width - 1 || (y) < 1) { posRU = pos; }
    else { posRU = readFromVBO(vboPtr, x + 1, y - 1, fxConst.OffstPos); }
    if ((x + 1) > fxConst.width - 1 || (y + 1) > fxConst.height - 1) { posRD = pos; }
    else { posRD = readFromVBO(vboPtr, x + 1, y + 1, fxConst.OffstPos); }
    //bend neighbor
    if ((x) < 2) { posL2 = pos; }
    else { posL2 = readFromVBO(vboPtr, x - 2, y + 0, fxConst.OffstPos); }
    if ((x + 2) > fxConst.width - 1) { posR2 = pos; }
    else { posR2 = readFromVBO(vboPtr, x + 2, y + 0, fxConst.OffstPos); }
    if ((y) < 2) { posU2 = pos; }
    else { posU2 = readFromVBO(vboPtr, x + 0, y - 2, fxConst.OffstPos); }
    if ((y + 2) > fxConst.height - 1) { posD2 = pos; }
    else { posD2 = readFromVBO(vboPtr, x + 0, y + 2, fxConst.OffstPos); }



    //glm::vec3 normal = glm::normalize(glm::cross((posR - posL), (posU - posD)));
    
    //pass const test 
    float a = clothConst.in_testFloat;
    glm::vec3 normal = glm::vec3(0.0f, a, 0.0f);
    writeToVBO(normal, vboPtr, x, y, fxConst.OffstNm);



    return glm::vec3(9999999999999999999999999.0f);
}

__device__ 
glm::vec3 computeForceNet(float* vboPtr, unsigned int ptclInd_x, unsigned int ptclInd_y) {
    
    
    
    glm::vec3 inForce = computeInnerForce(vboPtr, ptclInd_x, ptclInd_y);

    return glm::vec3(9999999999999999);

}

__device__
glm::vec3 RungeKutta4th() {


    return glm::vec3(99999999999);
}

__device__
glm::vec3 Verlet() {



    return glm::vec3(99999999999);
}

__device__
glm::vec3 explicitIntegration(const float timeStep) {

    //call RK4 or Verlet


}


__global__ 
void computeParticlePos_Kernel(float* vboPtr, unsigned int width, 
    unsigned int height, unsigned int vboStridInFloat)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height) return;

    //write normal
    //glm::vec3 normal = computeNormal(d_ptrVBO, x, y);
    //writeToVBO(normal, d_ptrVBO, x, y, OffsetNormal);
    
    
    //**test read function
    glm::vec3 posRead = readFromVBO(vboPtr, x, y, fxConst.OffstPos);

    //test output
    float u = x /100.0f;
    float v = y /100.0f;
    float freq = 0.5f;
    float w = glm::sin(u  + clothConst.time * 1.3f * freq) * glm::cos(v + clothConst.time * 1.7f * freq) * 0.5f;

    glm::vec3 ForceNet = computeForceNet(vboPtr, x, y);

    glm::vec3 NextPos = explicitIntegration(clothConst.stp);



    //write NextPos to VBO
    //test position
    glm::vec3 testPos = glm::vec3(u, w, v);
    writeToVBO(testPos, vboPtr, x, y, fxConst.OffstPos);
    
  



}


void Cloth_Launch_Kernel(float* vboPtr, const unsigned int mesh_width, const unsigned int mesh_height, 
    unsigned int vboStridInFloat)
{
 
    dim3 block(32, 32, 1);
    dim3 grid(ceil(mesh_width / block.x), ceil(mesh_height / block.y), 1);
    computeParticlePos_Kernel << < grid, block >> > (vboPtr, mesh_width, mesh_height, vboStridInFloat);
    CheckCudaErr("simple_vbo_kernel launch fail ");
    
    cudaDeviceSynchronize();
    CheckCudaErr("cudaDeviceSynghconize fail ");

}


