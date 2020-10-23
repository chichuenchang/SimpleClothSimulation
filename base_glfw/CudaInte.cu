//TODO
//1. pingpong buffer
//2. pass variables to kernel with constant

#include "CudaInte.cuh"

__constant__ 
ClothConstant cVar;
__constant__
FixedClothConstant fxVar;


__device__
float curr, last;

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
glm::vec3 computeInnerForce(float* readBuff, float* writeBuff, unsigned int x, unsigned int y) {
    //                |y-
    //                |
    //                |
    //          ------*-------x+
    //                |
    //                |
    //                |y+
    
    glm::vec3 pos, posL, posR, posU, posD, posLU, posLD, posRU, posRD, posL2,
        posR2, posU2, posD2;
    pos = readFromVBO(readBuff, x, y, fxVar.OffstPos);

    glm::vec3 innF = glm::vec3(0.0f);

    //structure
    if ((x) < 1) { innF += glm::vec3(0.0); }
    else { 
        posL = readFromVBO(readBuff, x - 1, y + 0, fxVar.OffstPos); 
        innF += glm::normalize((posL - pos)) * cVar.k * ((glm::length(posL - pos)) - cVar.rLen);
    }
    if ((x + 1) > fxVar.width - 1) { innF += glm::vec3(0.0f); }
    else { 
        posR = readFromVBO(readBuff, x + 1, y + 0, fxVar.OffstPos); 
        innF += glm::normalize(posR - pos) * cVar.k * ((glm::length(posR - pos)) - cVar.rLen);
    }
    if ((y) < 1) { innF += glm::vec3(0.0f); }
    else { 
        posU = readFromVBO(readBuff, x + 0, y - 1, fxVar.OffstPos);
        innF += glm::normalize((posU - pos)) * cVar.k * ((glm::length(posU - pos)) - cVar.rLen);
    }
    if ((y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else { 
        posD = readFromVBO(readBuff, x + 0, y + 1, fxVar.OffstPos); 
        innF += glm::normalize((posD - pos)) * cVar.k * ((glm::length(posD - pos)) - cVar.rLen);
    }

    //shear neighbor
    if ((x) < 1 || (y) < 1) { innF += glm::vec3(0.0f); }
    else { 
        posLU = readFromVBO(readBuff, x - 1, y - 1, fxVar.OffstPos); 
        innF += glm::normalize(posLU - pos) * cVar.k * (glm::length(posLU - pos) - cVar.rLen * 1.41421356237f);
    }
    if ((x) < 1 || (y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else { 
        posLD = readFromVBO(readBuff, x - 1, y + 1, fxVar.OffstPos); 
        innF += glm::normalize(posLD - pos) * cVar.k * (glm::length(posLD - pos) - cVar.rLen * 1.41421356237f);
    }
    if ((x + 1) > fxVar.width - 1 || (y) < 1) { innF += glm::vec3(0.0f); }
    else {
        posRU = readFromVBO(readBuff, x + 1, y - 1, fxVar.OffstPos); 
        innF += glm::normalize(posRU - pos) * cVar.k * (glm::length(posRU - pos) - cVar.rLen * 1.41421356237f);
    }
    if ((x + 1) > fxVar.width - 1 || (y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else { 
        posRD = readFromVBO(readBuff, x + 1, y + 1, fxVar.OffstPos); 
        innF += glm::normalize(posRD - pos) * cVar.k * (glm::length(posRD - pos) - cVar.rLen * 1.41421356237f);
    }
    //bend neighbor
    if ((x) < 2) { innF += glm::vec3(0.0f); }
    else {
        posL2 = readFromVBO(readBuff, x - 2, y + 0, fxVar.OffstPos); 
        innF += glm::normalize(posL2 - pos) * cVar.k * (glm::length(posL2 - pos) - cVar.rLen * 2.0f);
    }
    if ((x + 2) > fxVar.width - 1) { innF += glm::vec3(0.0f); }
    else { 
        posR2 = readFromVBO(readBuff, x + 2, y + 0, fxVar.OffstPos); 
        innF += glm::normalize(posR2 - pos) * cVar.k * (glm::length(posR2 - pos) - cVar.rLen * 2.0f);
    }
    if ((y) < 2) { innF += glm::vec3(0.0f); }
    else { 
        posU2 = readFromVBO(readBuff, x + 0, y - 2, fxVar.OffstPos); 
        innF += glm::normalize(posU2 - pos) * cVar.k * (glm::length(posU2 - pos) - cVar.rLen * 2.0f);
    }
    if ((y + 2) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else { 
        posD2 = readFromVBO(readBuff, x + 0, y + 2, fxVar.OffstPos); 
        innF += glm::normalize(posD2 - pos) * cVar.k * (glm::length(posD2 - pos) - cVar.rLen * 2.0f);
    }
    
    //might as well compute the normal
    glm::vec3 normal = glm::normalize(glm::cross((posR - posL), (posU - posD)));
    
    glm::vec3 col = glm::vec3(glm::length(innF), 0.0f, 1.0f - glm::length(innF));

    writeToVBO(col, writeBuff, x, y, fxVar.OffstCol);
    writeToVBO(normal, writeBuff, x, y, fxVar.OffstNm);

    float a = cVar.in_testFloat;
  /*  glm::vec3 testcol = glm::vec3(0.0f, a, a);
    if (x == 0 && y == 0) {

        curr = a;

        printf(" innerForce = %f \n", glm::length(innF));
        last = curr;

    }*/
    return innF;
}

__device__ 
glm::vec3 computeForceNet(float* readBuff, float* writeBuff, unsigned int x, unsigned int y) {
    
    
    glm::vec3 innF = computeInnerForce(readBuff, writeBuff, x, y);

    glm::vec3 vel = readFromVBO(readBuff, x, y, fxVar.OffstVel);


    //F = m*g + Fwind - air * vel* vel + innF - damp = m*Acc;
    glm::vec3 pos = readFromVBO(readBuff, x, y, fxVar.OffstPos);
    glm::vec3 Fwind = cVar.in_testFloat *
        glm::vec3(glm::sin(pos.y * 1.3), -glm::sin(pos.z * 0.7), glm::cos(1.7f * pos.x));

    glm::vec3 netF = cVar.M * glm::vec3(0.0f, cVar.g, 0.0f)  +0.0f*cVar.Fw + Fwind 
        + (-cVar.a) *vel * (glm::length(vel))
        + innF - cVar.Dp * vel * (glm::length(vel));


    return netF;

}

__device__
glm::vec3 RungeKutta4th(glm::vec3 pos, glm::vec3 acc, float dt) {

    return glm::vec3(99999999999);
}

__device__
glm::vec3 Verlet(glm::vec3 pos, glm::vec3 oldPos, glm::vec3 acc, float dt) {

    return 2.0f * pos - oldPos + acc * dt * dt;
}

//__device__
//glm::vec3 explicitIntegration(glm::vec3 pos, glm::vec3 nForce, float dt) {
//
//    //call RK4 or Verlet
//    glm::vec3 acc = nForce / cVar.M;
//
//    
//
//    return RungeKutta4th(pos, acc, dt);
//
//}


__global__ 
void computeParticlePos_Kernel(float* readBuff, float* writeBuff, unsigned int width,
    unsigned int height, unsigned int vboStridInFloat)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height) return;

    //write normal
    //glm::vec3 normal = computeNormal(d_ptrVBO, x, y);
    //writeToVBO(normal, d_ptrVBO, x, y, OffsetNormal);
    
    
    //get current position from read buffer
    glm::vec3 pos = readFromVBO(readBuff, x, y, fxVar.OffstPos);
    //get old position from the write buffer
    glm::vec3 lastPos = readFromVBO(writeBuff, x, y, fxVar.OffstPos);

    glm::vec3 ForceNet = computeForceNet(readBuff, writeBuff, x, y);

    glm::vec3 Acc = ForceNet / cVar.M;
    //glm::vec3 Acc = glm::vec3(1.0f);
    glm::vec3 lastV = readFromVBO(readBuff, x, y, fxVar.OffstVel);
    glm::vec3 newV = lastV + Acc * cVar.stp;
    writeToVBO(newV, writeBuff, x, y, fxVar.OffstVel);


    glm::vec3 nextPos;
    if ((x == 0 && y == 0) || (x == 0 && y == height - 1) ) {


        nextPos = pos;
    }
    else {

        nextPos = Verlet(pos, lastPos, Acc, cVar.stp);

    }

    ///////////////////////////////////
    //test output
    float u = x /100.0f;
    float v = y /100.0f;
    float freq = 0.5f;
    float w = glm::sin(u  + cVar.time * 1.3f * freq) * glm::cos(v + cVar.time * 1.7f * freq) * 0.5f;

    //write NextPos to VBO
    //test position
    //glm::vec3 testPos = glm::vec3(u, w, v);
    writeToVBO(nextPos, writeBuff, x, y, fxVar.OffstPos);
    //writeToVBO(posRead, readBuff, x, y, fxVar.OffstPos);

}


void Cloth_Launch_Kernel(float* readBuff, float* writeBuff, const unsigned int mesh_width, const unsigned int mesh_height,
    unsigned int vboStridInFloat)
{
 
    dim3 block(32, 32, 1);
    dim3 grid(ceil(mesh_width / block.x), ceil(mesh_height / block.y), 1);
    computeParticlePos_Kernel << < grid, block >> > (readBuff, writeBuff, mesh_width,
        mesh_height, vboStridInFloat);
    CheckCudaErr("simple_vbo_kernel launch fail ");
    
    cudaDeviceSynchronize();
    CheckCudaErr("cudaDeviceSynghconize fail ");

}


