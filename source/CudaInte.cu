
#include "CudaInte.cuh"

__constant__ 
ClothConstant cVar;
__constant__
FixedClothConstant fxVar;


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

__device__
glm::vec3 constraintForce(glm::vec3 p_to_nbor, float lCoeff) {
    //constrain Type = 1.0f: structure
    //constrain Type = 1.41421356237f : shear
    //constrain Type = 2.0f: bend
    float L = glm::length(p_to_nbor);
    if (L < lCoeff * cVar.MxL) { 
        return glm::normalize(p_to_nbor) * cVar.k * (L - lCoeff * cVar.rLen); 
    }
    else {
        return glm::normalize(p_to_nbor) * (cVar.k * (lCoeff * cVar.MxL - lCoeff * cVar.rLen) +
            cVar.k * 1.3f * (L - lCoeff*cVar.MxL));
    }
}

__device__//normal is here also
glm::vec3 computeInnerForce(float* readBuff, float* writeBuff, unsigned int x,
    unsigned int y, glm::vec3 curP) {
    //                |y-
    //                |
    //                |
    //          ------*-------x+
    //                |
    //                |
    //                |y+

    glm::vec3 tempV = glm::vec3(0.0f);
    glm::vec3 innF = glm::vec3(0.0f);
    //structure
    //left
    if ((x) < 1) { innF += glm::vec3(0.0);}
    else {
        tempV = readFromVBO(readBuff, x - 1, y + 0, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.0f);
    }
    //right
    if ((x + 1) > fxVar.width - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x + 1, y + 0, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.0f);
    }
    //up
    if ((y) < 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x + 0, y - 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.0f);
    }
    //down
    if ((y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x + 0, y + 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.0f);
    }

    //shear neighbor
    //left up
    if ((x) < 1 || (y) < 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x - 1, y - 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.41421356237f);
    }
    //left down
    if ((x) < 1 || (y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x - 1, y + 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.41421356237f);
    }
    //right up
    if ((x + 1) > fxVar.width - 1 || (y) < 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x + 1, y - 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.41421356237f);
    }
    //right down
    if ((x + 1) > fxVar.width - 1 || (y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x + 1, y + 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.41421356237f);
    }

    //bend neighbor
    //left 2
    if ((x) < 2) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x - 2, y + 0, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 2.0f);
    }
    //right 2
    if ((x + 2) > fxVar.width - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x + 2, y + 0, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 2.0f);
    }
    //up 2
    if ((y) < 2) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x + 0, y - 2, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 2.0f);
    }
    //down 2
    if ((y + 2) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(readBuff, x + 0, y + 2, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 2.0f);
    }

    //color represents the magnitude of inner force
    glm::vec3 col = glm::vec3(3.0f*glm::length(innF) , 0.3f, 0.7f - glm::length(innF));
    writeToVBO(col, writeBuff, x, y, fxVar.OffstCol);

    //float a = cVar.in_testFloat;
  /*  glm::vec3 testcol = glm::vec3(0.0f, a, a);
    if (x == 0 && y == 0) {

        curr = a;

        printf(" innerForce = %f \n", glm::length(innF));
        last = curr;

    }*/
    return innF;
}

__device__ 
glm::vec3 computeForceNet(glm::vec3 currPos, float* readBuff, float* writeBuff, 
                            unsigned int x, unsigned int y  ) {
    
    
    glm::vec3 innF = computeInnerForce(readBuff, writeBuff, x, y, currPos);

    glm::vec3 vel = readFromVBO(readBuff, x, y, fxVar.OffstVel);

    glm::vec3 Fwind = cVar.WStr*
        glm::vec3(1.0f + cVar.WDir.x* glm::sin(cVar.offsCo.x* currPos.z + cVar.cyclCo.x* cVar.time),
            cVar.WDir.y* glm::sin(cVar.offsCo.y * currPos.y + cVar.cyclCo.y* cVar.time),
            cVar.WDir.z * glm::cos(cVar.offsCo.z * currPos.y + cVar.cyclCo.z * cVar.time));
    
    //***********************************
    //F = m*g + Fwind - air * vel* vel + innF - damp = m*Acc;
    glm::vec3 netF =
        1.0f * cVar.M * glm::vec3(0.0f, cVar.g, 0.0f)
        /*+ 1.0f * cVar.in_testFloat * cVar.Fw*/
        + Fwind
        -1.0f * cVar.a * vel * (glm::length(vel))
        + 1.0f * innF
        - 1.0f * cVar.Dp * vel * (glm::length(vel));


    return netF;
}

__device__ glm::vec3 RK4func(float stpT, glm::vec3 pos, glm::vec3 acc, glm::vec3 vel) {
    return pos + vel * stpT + 0.5f * acc * stpT * stpT;
}

__device__
glm::vec3 RungeKutt(float stpT, glm::vec3 pT0, glm::vec3 acc, glm::vec3 vel) {

    glm::vec3 K1 = pT0;
    glm::vec3 K2 = RK4func(stpT / 2.0f, pT0 + K1 / 2.0f, acc, vel);
    glm::vec3 K3 = RK4func(stpT / 2.0f, pT0 + K2 / 2.0f, acc, vel);
    glm::vec3 K4 = RK4func(stpT, pT0 + K3, acc, vel);

    return pT0 + 1.0f / 6.0f * stpT * (K1 + 2.0f * K2 + 2.0f * K3 + K4);
}

__device__
glm::vec3 VerletAlg(glm::vec3 pos, glm::vec3 oldPos, glm::vec3 acc, float stepT) {

    return 2.0f * pos - oldPos + acc * stepT * stepT;
}


__device__
glm::vec3 ComputeNomral(float* readBuff, unsigned int x, unsigned int y, glm::vec3 curP) {

    glm::vec3 posL, posR, posU, posD;
    
    //left
    if ((x) < 1) { posL = curP; }
    else {
        posL = readFromVBO(readBuff, x - 1, y + 0, fxVar.OffstPos);
    }
    //right
    if ((x + 1) > fxVar.width - 1) { posR = curP; }
    else {
        posR = readFromVBO(readBuff, x + 1, y + 0, fxVar.OffstPos);
    }
    //up
    if ((y) < 1) {  posU = curP; }
    else {
        posU = readFromVBO(readBuff, x + 0, y - 1, fxVar.OffstPos);
    }
    //down
    if ((y + 1) > fxVar.height - 1) {  posD = curP; }
    else {
        posD = readFromVBO(readBuff, x + 0, y + 1, fxVar.OffstPos);
    }
    
    return glm::normalize(glm::cross((posR - posL), (posU - posD)));
}

__global__
void computeParticlePos_Kernel(float* readBuff, float* writeBuff, unsigned int width,
    unsigned int height, unsigned int vboStridInFloat)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height) return;

    //current pos and last frame pos
    glm::vec3 Pos = readFromVBO(readBuff, x, y, fxVar.OffstPos);
    glm::vec3 lastPos = readFromVBO(writeBuff, x, y, fxVar.OffstPos);
    //normal
    glm::vec3 normal = ComputeNomral(readBuff, x, y, Pos);
    writeToVBO(normal, writeBuff, x, y, fxVar.OffstNm);
    
    //ForceNet
    glm::vec3 ForceNet = computeForceNet(Pos, readBuff, writeBuff, x, y);

    glm::vec3 Acc = ForceNet / cVar.M;

    //velocity
    glm::vec3 lastV = readFromVBO(readBuff, x, y, fxVar.OffstVel);
    glm::vec3 Vel = lastV + Acc * cVar.stp;
    writeToVBO(Vel, writeBuff, x, y, fxVar.OffstVel);

    glm::vec3 nextPos;

    if ((x == 0 && y == 0) || (x == 0 && y == height - 1) ||
        (x == 0 && y == height / 4) || (x == 0 && y == 3 * height / 4)
        ) {

        glm::vec3 dir = glm::vec3(0.0f, 0.0f, 0.5f * fxVar.height / 10.0f) - Pos;

        nextPos = Pos + 0.001f* dir * cVar.in_testFloat;
    }
    else {

        nextPos = !cVar.frz? VerletAlg(Pos, lastPos, Acc, cVar.stp): Pos;
        //nextPos = RungeKutt(cVar.stp, Pos, Vel, Acc);
    }

    writeToVBO(nextPos, writeBuff, x, y, fxVar.OffstPos);

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


