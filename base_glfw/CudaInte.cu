//TODO
//1. pingpong buffer
//2. pass variables to kernel with constant

#include "CudaInte.cuh"

__device__
float* d_ptrVBO;
__device__
unsigned int ClothWidth, ClothHeight, StrideInFltVBO;
__device__
const int OffsetPosition = 0, OffsetNormal = 5;


//__device__
//glm::vec3 posPtcl, posL, posR, posU, posD, posLU, posLD, posRU, posRD, posL2, posR2, posU2, posD2;

__device__//cloth variables are passed from GUI
float m = 0.1f, G = -9.8f, k = 0.01f, restLength = 0.02f;
__device__//passed from GUI
glm::vec3 Fwind;
__device__
const float tStep = 0.001f;
__device__
float deltaT;
__device__
glm::vec3 forceNet;



__constant__ 
passVar* d_const_passVar;


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

void copyConstMem(passVar *in_passVar) {

    cudaMemcpyToSymbol(d_const_passVar, in_passVar, sizeof(passVar));
    CheckCudaErr("constant memory copy fail");


}
__device__
glm::vec3 readFromVBO(float* d_in_particleBuff, unsigned int ind_x, unsigned int ind_y,
    const int offsetInEachVBOElement) {

    float x = d_ptrVBO[(ind_x * ClothHeight + ind_y) * StrideInFltVBO + offsetInEachVBOElement + 0];
    float y = d_ptrVBO[(ind_x * ClothHeight + ind_y) * StrideInFltVBO + offsetInEachVBOElement + 1];
    float z = d_ptrVBO[(ind_x * ClothHeight + ind_y) * StrideInFltVBO + offsetInEachVBOElement + 2];

    return glm::vec3(x, y, z);
}

__device__ 
void writeToVBO(glm::vec3 outPos, float* d_particleBuff,
    unsigned int ind_x, unsigned int ind_y, const int offsetInEachVBOElement) {

    d_ptrVBO[(ind_x * ClothHeight + ind_y) * StrideInFltVBO + offsetInEachVBOElement + 0] = outPos.x;
    d_ptrVBO[(ind_x * ClothHeight + ind_y) * StrideInFltVBO + offsetInEachVBOElement + 1] = outPos.y;
    d_ptrVBO[(ind_x * ClothHeight + ind_y) * StrideInFltVBO + offsetInEachVBOElement + 2] = outPos.z;
}


__device__ 
glm::vec3 computeNormal(float* d_particleBuff, unsigned int in_x, unsigned int in_y) {
    //      *-------x+
    //      |
    //      |
    //      |y+
    glm::vec3 up, down, left, right;
    glm::vec3 pos = readFromVBO(d_ptrVBO, in_x, in_y, OffsetPosition);

    if (in_y < 1) { up = pos; }
    else { up = readFromVBO(d_ptrVBO, in_x, in_y - 1, OffsetPosition); }
    if (in_x < 1) { left = pos; }
    else { left = readFromVBO(d_ptrVBO, in_x - 1, in_y, OffsetPosition); }
    if ((in_y + 1) > ClothHeight - 1) { down = pos; }
    else {down = readFromVBO(d_ptrVBO, in_x, in_y + 1, OffsetPosition); }
    if ((in_x + 1) > ClothHeight -1 ) { right = pos; }
    else {right = readFromVBO(d_ptrVBO, in_x + 1, in_y, OffsetPosition); }

    return  glm::normalize(glm::cross((right - left), (up - down)));
}

__device__
glm::vec3 computeInnerForce(float k, unsigned int x, unsigned int y) {

    glm::vec3 pos, posL, posR, posU, posD, posLU, posLD, posRU, posRD, posL2,
        posR2, posU2, posD2;
    pos = readFromVBO(d_ptrVBO, x, y, OffsetPosition);
    //if neighbor point out of bound, neighbor point = center point
    //structure neighbor
    if ((x) < 1) { posL = pos; }
    else { posL = readFromVBO(d_ptrVBO, x - 1, y + 0, OffsetPosition); }
    if ((x + 1) > ClothWidth - 1) { posR = pos; }
    else { posR = readFromVBO(d_ptrVBO, x + 1, y + 0, OffsetPosition); }
    if ((y) < 1) { posU = pos; }
    else { posU = readFromVBO(d_ptrVBO, x + 0, y - 1, OffsetPosition); }
    if ((y + 1) > ClothHeight - 1) { posD = pos; }
    else { posD = readFromVBO(d_ptrVBO, x + 0, y + 1, OffsetPosition); }
    //shear neighbor
    if ((x) < 1 || (y) < 1) { posLU = pos; }
    else { posLU = readFromVBO(d_ptrVBO, x - 1, y - 1, OffsetPosition); }
    if ((x) < 1 || (y + 1) > ClothHeight - 1) { posLD = pos; }
    else { posLD = readFromVBO(d_ptrVBO, x - 1, y + 1, OffsetPosition); }
    if ((x + 1) > ClothWidth - 1 || (y) < 1) { posRU = pos; }
    else { posRU = readFromVBO(d_ptrVBO, x + 1, y - 1, OffsetPosition); }
    if ((x + 1) > ClothWidth - 1 || (y + 1) > ClothHeight - 1) { posRD = pos; }
    else { posRD = readFromVBO(d_ptrVBO, x + 1, y + 1, OffsetPosition); }
    //bend neighbor
    if ((x) < 2) { posL2 = pos; }
    else { posL2 = readFromVBO(d_ptrVBO, x - 2, y + 0, OffsetPosition); }
    if ((x + 2) > ClothWidth - 1) { posR2 = pos; }
    else { posR2 = readFromVBO(d_ptrVBO, x + 2, y + 0, OffsetPosition); }
    if ((y) < 2) { posU2 = pos; }
    else { posU2 = readFromVBO(d_ptrVBO, x + 0, y - 2, OffsetPosition); }
    if ((y + 2) > ClothHeight - 1) { posD2 = pos; }
    else { posD2 = readFromVBO(d_ptrVBO, x + 0, y + 2, OffsetPosition); }

    //glm::vec3 normal = glm::normalize(glm::cross((posR - posL), (posU - posD)));
    
    //test****
    float a = d_const_passVar->in_testFloat;
    glm::vec3 normal = glm::vec3(0.0f, a, 0.0f);
    writeToVBO(normal, d_ptrVBO, x, y, OffsetNormal);






    return glm::vec3(9999999999999999999999999.0f);
}

__device__ 
glm::vec3 computeForceNet(float m, glm::vec3 g, float k, float restLen, 
    unsigned int ptclInd_x, unsigned int ptclInd_y) {
    
    forceNet = m * g - computeInnerForce(k, ptclInd_x, ptclInd_y);

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

//__global__ // a temporary solution to the r-w collision
//void getParticlePosition_Kernel(float* d_InOut_particleBuff, unsigned int width,
//    unsigned int height, float time, unsigned int vboStridInFloat) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if (x > ClothWidth || y > ClothHeight) return;
//
//
//    glm::vec3 posPtcl, posL, posR, posU, posD, posLU, posLD, posRU, posRD, posL2,
//        posR2, posU2, posD2;
//
//
//
//    posPtcl = readFromVBO(d_InOut_particleBuff, x, y, OffsetPosition);
//    //structure neighbor
//    if ((x - 1) < 0) { posL = posPtcl; }
//    else { posL = readFromVBO(d_InOut_particleBuff, x - 1, y + 0, OffsetPosition); }
//    if ((x + 1) > ClothWidth - 1) { posR = posPtcl; }
//    else { posR = readFromVBO(d_InOut_particleBuff, x + 1, y + 0, OffsetPosition); }
//    if ((y - 1) < 0) { posU = posPtcl; }
//    else { posU = readFromVBO(d_InOut_particleBuff, x + 0, y - 1, OffsetPosition); }
//    if ((y + 1) > ClothHeight -1) { posD = posPtcl; }
//    else { posD = readFromVBO(d_InOut_particleBuff, x + 0, y + 1, OffsetPosition); }
//    //shear neighbor
//    if ((x - 1) < 0 || (y - 1) < 0) { posLU = posPtcl;}
//    else { posLU = readFromVBO(d_InOut_particleBuff, x - 1, y - 1, OffsetPosition); }
//    if ((x - 1) < 0 || (y + 1) > ClothHeight -1) { posLD = posPtcl;}
//    else { posLD = readFromVBO(d_InOut_particleBuff, x - 1, y + 1, OffsetPosition); }
//    if ((x + 1) > ClothWidth -1 || (y - 1) < 0) { posRU = posPtcl;}
//    else { posRU = readFromVBO(d_InOut_particleBuff, x + 1, y - 1, OffsetPosition); }
//    if ((x + 1) > ClothWidth -1 || (y + 1) > ClothHeight -1) { posRD = posPtcl;}
//    else { posRD = readFromVBO(d_InOut_particleBuff, x + 1, y + 1, OffsetPosition); }
//    //bend neighbor
//    if ((x - 2) < 0) { posL2 = posPtcl; }
//    else { posL2 = readFromVBO(d_InOut_particleBuff, x - 2, y + 0, OffsetPosition); }
//    if ((x + 2) > ClothWidth - 1) { posR2 = posPtcl; }
//    else { posR2 = readFromVBO(d_InOut_particleBuff, x + 2, y + 0, OffsetPosition); }
//    if ((y - 2) < 0) { posU2 = posPtcl; }
//    else { posU2 = readFromVBO(d_InOut_particleBuff, x + 0, y - 2, OffsetPosition); }
//    if ((y + 2) > ClothHeight - 1) { posD2 = posPtcl; }
//    else { posD2 = readFromVBO(d_InOut_particleBuff, x + 0, y + 2, OffsetPosition); }
//
//
//}


__global__ 
void computeParticlePos_Kernel(float* d_InOut_particleBuff, unsigned int width, 
    unsigned int height, float time, unsigned int vboStridInFloat)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height) return;
    d_ptrVBO = d_InOut_particleBuff;
    ClothWidth = width;
    ClothHeight = height;
    StrideInFltVBO = vboStridInFloat;



    //write normal
    //glm::vec3 normal = computeNormal(d_ptrVBO, x, y);
    //writeToVBO(normal, d_ptrVBO, x, y, OffsetNormal);
    
    
    //**test read function
    glm::vec3 posRead = readFromVBO(d_ptrVBO, x, y, OffsetPosition);

    //test output
    float u = x /100.0f;
    float v = y /100.0f;
    float freq = 0.5f;
    float w = glm::sin(u  + time * 1.3f * freq) * glm::cos(v + time * 1.7f * freq) * 0.5f;

    glm::vec3 ForceNet = computeForceNet(m, glm::vec3(0.0f, G, 0.0f), k, restLength, x, y);

    glm::vec3 NextPos = explicitIntegration(tStep);



    //write NextPos to VBO
    //test position
    glm::vec3 testPos = glm::vec3(u, w, v);
    writeToVBO(testPos, d_ptrVBO, x, y, OffsetPosition);
    
  



}


void Cloth_Launch_Kernel(float* pos, const unsigned int mesh_width, const unsigned int mesh_height, 
    float time, unsigned int vboStridInFloat)
{
    //***************
    //float* d_localBuffer;
    //cudaMalloc((void**)&d_localBuffer, mesh_width * mesh_height * sizeof(float)* 8);
    //checkCudaError("cudamalloc");

    // execute the kernel
    dim3 block(32, 32, 1);
    dim3 grid(ceil(mesh_width / block.x), ceil(mesh_height / block.y), 1);

    computeParticlePos_Kernel << < grid, block >> > (pos, mesh_width, mesh_height, time, vboStridInFloat);
    CheckCudaErr("simple_vbo_kernel launch fail ");
    
    //cudaDeviceSynchronize();
    //CheckCudaErr("cudaDeviceSynghconize fail ");

}


//////////////////////////////////////////////////////////////////////////////////////
//util func called from clothrender.h
//////////////////////////////////////////////////////////////////////////////////////

