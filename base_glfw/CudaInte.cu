//TODO
//1. pingpong buffer
//2. pass variables to kernel with constant

#include "CudaInte.cuh"

__constant__ 
ClothConstant cVar;
__constant__
FixedClothConstant fxVar;
/// <summary>
/// /????????????????????????????????
/// </summary>
__constant__
glm::vec3 vel;
__constant__
glm::vec3 lastPos;


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

    cudaMemcpyToSymbol(cVar, in_passVar, sizeof(ClothConstant));
    CheckCudaErr("cloth constant memory copy fail");
}

void copyFixClothConst(FixedClothConstant* in_fxConst) {

    cudaMemcpyToSymbol(fxVar, in_fxConst, sizeof(FixedClothConstant));
    CheckCudaErr("fixed constant memory copy fail");

    cudaMemcpyToSymbol(vel, &in_fxConst->initVel, sizeof(glm::vec3));
    CheckCudaErr("fixed constant memory copy fail");
}

__device__
glm::vec3 readFromVBO(float* d_vboPtr, unsigned int ind_x, unsigned int ind_y,
    const int offsetInEachVBOElement) {

    float x = d_vboPtr[(ind_x * fxVar.height + ind_y) * fxVar.vboStrdFlt + offsetInEachVBOElement + 0];
    float y = d_vboPtr[(ind_x * fxVar.height + ind_y) * fxVar.vboStrdFlt + offsetInEachVBOElement + 1];
    float z = d_vboPtr[(ind_x * fxVar.height + ind_y) * fxVar.vboStrdFlt + offsetInEachVBOElement + 2];

    return glm::vec3(x, y, z);
}

__device__ 
void writeToVBO(glm::vec3 outPos, float* d_vboPtr,
    unsigned int ind_x, unsigned int ind_y, const int offsetInEachVBOElement) {

    d_vboPtr[(ind_x * fxVar.height + ind_y) * fxVar.vboStrdFlt + offsetInEachVBOElement + 0] = outPos.x;
    d_vboPtr[(ind_x * fxVar.height + ind_y) * fxVar.vboStrdFlt + offsetInEachVBOElement + 1] = outPos.y;
    d_vboPtr[(ind_x * fxVar.height + ind_y) * fxVar.vboStrdFlt + offsetInEachVBOElement + 2] = outPos.z;
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
    pos = readFromVBO(vboPtr, x, y, fxVar.OffstPos);
    //if neighbor point out of bound, neighbor point = center point
    //structure neighbor
    if ((x) < 1) { posL = pos; }
    else { posL = readFromVBO(vboPtr, x - 1, y + 0, fxVar.OffstPos); }
    if ((x + 1) > fxVar.width - 1) { posR = pos; }
    else { posR = readFromVBO(vboPtr, x + 1, y + 0, fxVar.OffstPos); }
    if ((y) < 1) { posU = pos; }
    else { posU = readFromVBO(vboPtr, x + 0, y - 1, fxVar.OffstPos); }
    if ((y + 1) > fxVar.height - 1) { posD = pos; }
    else { posD = readFromVBO(vboPtr, x + 0, y + 1, fxVar.OffstPos); }
    //shear neighbor
    if ((x) < 1 || (y) < 1) { posLU = pos; }
    else { posLU = readFromVBO(vboPtr, x - 1, y - 1, fxVar.OffstPos); }
    if ((x) < 1 || (y + 1) > fxVar.height - 1) { posLD = pos; }
    else { posLD = readFromVBO(vboPtr, x - 1, y + 1, fxVar.OffstPos); }
    if ((x + 1) > fxVar.width - 1 || (y) < 1) { posRU = pos; }
    else { posRU = readFromVBO(vboPtr, x + 1, y - 1, fxVar.OffstPos); }
    if ((x + 1) > fxVar.width - 1 || (y + 1) > fxVar.height - 1) { posRD = pos; }
    else { posRD = readFromVBO(vboPtr, x + 1, y + 1, fxVar.OffstPos); }
    //bend neighbor
    if ((x) < 2) { posL2 = pos; }
    else { posL2 = readFromVBO(vboPtr, x - 2, y + 0, fxVar.OffstPos); }
    if ((x + 2) > fxVar.width - 1) { posR2 = pos; }
    else { posR2 = readFromVBO(vboPtr, x + 2, y + 0, fxVar.OffstPos); }
    if ((y) < 2) { posU2 = pos; }
    else { posU2 = readFromVBO(vboPtr, x + 0, y - 2, fxVar.OffstPos); }
    if ((y + 2) > fxVar.height - 1) { posD2 = pos; }
    else { posD2 = readFromVBO(vboPtr, x + 0, y + 2, fxVar.OffstPos); }

    //structure
    glm::vec3 f_L = glm::normalize(posL - pos) * cVar.k * (glm::length(posL - pos) - cVar.rLen);
    glm::vec3 f_R = glm::normalize(posR - pos) * cVar.k * (glm::length(posR - pos) - cVar.rLen);
    glm::vec3 f_U = glm::normalize(posU - pos) * cVar.k * (glm::length(posU - pos) - cVar.rLen);
    glm::vec3 f_D = glm::normalize(posD - pos) * cVar.k * (glm::length(posD - pos) - cVar.rLen);
    //shear
    glm::vec3 f_LU = glm::normalize(posLU - pos) * cVar.k * (glm::length(posLU - pos) - cVar.rLen * 1.41421356237f);
    glm::vec3 f_LD = glm::normalize(posLD - pos) * cVar.k * (glm::length(posLD - pos) - cVar.rLen * 1.41421356237f);
    glm::vec3 f_RU = glm::normalize(posRU - pos) * cVar.k * (glm::length(posRU - pos) - cVar.rLen * 1.41421356237f);
    glm::vec3 f_RD = glm::normalize(posRD - pos) * cVar.k * (glm::length(posRD - pos) - cVar.rLen * 1.41421356237f);
    //bend
    glm::vec3 f_L2 = glm::normalize(posL2 - pos) * cVar.k * (glm::length(posL2 - pos) - cVar.rLen * 1.41421356237f);
    glm::vec3 f_R2 = glm::normalize(posR2 - pos) * cVar.k * (glm::length(posR2 - pos) - cVar.rLen * 1.41421356237f);
    glm::vec3 f_U2 = glm::normalize(posU2 - pos) * cVar.k * (glm::length(posU2 - pos) - cVar.rLen * 1.41421356237f);
    glm::vec3 f_D2 = glm::normalize(posD2 - pos) * cVar.k * (glm::length(posD2 - pos) - cVar.rLen * 1.41421356237f);
    //might as well compute the normal
    //glm::vec3 normal = glm::normalize(glm::cross((posR - posL), (posU - posD)));
    
    //pass const test 

  

    float a = cVar.in_testFloat;
    glm::vec3 normal = glm::vec3(0.0f, a, 0.0f);

    writeToVBO(normal, vboPtr, x, y, fxVar.OffstNm);



    return f_L + f_R + f_U + f_D + f_LU + f_LD + f_RU + f_RD + f_L2 + f_R2 + f_U2 + f_D2;
}

__device__ 
glm::vec3 computeForceNet(float* vboPtr, unsigned int ptclInd_x, unsigned int ptclInd_y) {
    
    
    
    glm::vec3 inForce = computeInnerForce(vboPtr, ptclInd_x, ptclInd_y);


    glm::vec3 netF = cVar.M * glm::vec3(0.0f, cVar.g, 0.0f) + cVar.Fw +
        (-cVar.a) * glm::normalize(vel) * (glm::length(vel)) * (glm::length(vel)) - inForce;


    return netF;

}

__device__
glm::vec3 RungeKutta4th(glm::vec3 pos, glm::vec3 acc, float dt) {

    


    return glm::vec3(99999999999);
}

__device__
glm::vec3 Verlet(glm::vec3 pos, glm::vec3 oldPos, glm::vec3 acc, float dt) {

    

    return glm::vec3(99999999999);
}

__device__
glm::vec3 explicitIntegration(glm::vec3 pos, glm::vec3 nForce, float dt) {

    //call RK4 or Verlet
    glm::vec3 acc = nForce / cVar.M;

    

    return RungeKutta4th(pos, acc, dt);

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
    glm::vec3 posRead = readFromVBO(vboPtr, x, y, fxVar.OffstPos);

    //test output
    float u = x /100.0f;
    float v = y /100.0f;
    float freq = 0.5f;
    float w = glm::sin(u  + cVar.time * 1.3f * freq) * glm::cos(v + cVar.time * 1.7f * freq) * 0.5f;

    glm::vec3 ForceNet = computeForceNet(vboPtr, x, y);



    glm::vec3 NextPos = explicitIntegration(posRead, ForceNet, cVar.stp * cVar.dt);

    

    //write NextPos to VBO
    //test position
    glm::vec3 testPos = glm::vec3(u, w, v);
    writeToVBO(testPos, vboPtr, x, y, fxVar.OffstPos);
    
  



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


