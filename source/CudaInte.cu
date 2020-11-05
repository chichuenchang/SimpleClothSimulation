
#include "CudaInte.cuh"
//cloth const
__constant__ 
ClothConstant cVar;
__constant__
FixedClothConstant fxVar;
__constant__
float* ppReadBuff, *ppWriteBuff;
__device__
bool* colFlag;
__device__
int* collCount;

//customized obj
__constant__
float* objBuff;
__constant__
unsigned int* objIndBuff;
__constant__
objConst objVar;
__device__
glm::vec3* objN;


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
//copy ptr to kernel
void passCstmObjPtr(float* d_vbo, unsigned int* d_ibo, glm::vec3* d_objN) {
    
    cudaMemcpyToSymbol(objBuff, &d_vbo, sizeof(float*));
    CheckCudaErr("sphere vbo pointer copy fail");
    cudaMemcpyToSymbol(objIndBuff, &d_ibo, sizeof(float*));
    CheckCudaErr("sphere vbo pointer copy fail");
    cudaMemcpyToSymbol(objN, &d_objN, sizeof(float*));
    CheckCudaErr("object normal array pointer copy fail");

}

void cpyObjConst(objConst* in_Var) {

    cudaMemcpyToSymbol(objVar, in_Var, sizeof(objConst));
    CheckCudaErr("obj constant memory copy fail");
}

//cloth
void passPPbuffPtr(float* d_vbo1, float* d_vbo2) {
    cudaMemcpyToSymbol(ppReadBuff, &d_vbo1, sizeof(float*));
    CheckCudaErr("pp read buffer pointer copy fail");
    cudaMemcpyToSymbol(ppWriteBuff, &d_vbo2, sizeof(float*));
    CheckCudaErr("pp write buffer pointer copy fail");
}

void updateClothConst(ClothConstant *in_passVar) {

    cudaMemcpyToSymbol(cVar, in_passVar, sizeof(ClothConstant));
    CheckCudaErr("cloth constant memory copy fail");
}

void copyFixClothConst(FixedClothConstant* in_fxConst) {

    cudaMemcpyToSymbol(fxVar, in_fxConst, sizeof(FixedClothConstant));
    CheckCudaErr("fixed constant memory copy fail");
}

void copyCollisionArrayPtr(bool* d_collPtr, int* d_collCountPtr) {

    cudaMemcpyToSymbol(colFlag, &d_collPtr, sizeof(bool*));
    CheckCudaErr("collision flag array pointer copy fail");

    cudaMemcpyToSymbol(collCount, &d_collCountPtr, sizeof(int*));
    CheckCudaErr("collision flag array pointer copy fail");
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
glm::vec3 readObjVbo(int threadInd, float* objVbo, unsigned int* objIbo, unsigned int strdFlt,
    unsigned int offst) {

    return glm::vec3(
        objVbo[objIbo[threadInd] * strdFlt + offst + 0],
        objVbo[objIbo[threadInd] * strdFlt + offst + 1],
        objVbo[objIbo[threadInd] * strdFlt + offst + 2]);
}

__device__
void writeObjVbo(glm::vec3 in_vec3, int threadInd, float* objVbo, unsigned int* objIbo, unsigned int strdFlt,
    unsigned int offst) {

    objVbo[objIbo[threadInd] * strdFlt + offst + 0] = in_vec3.x;
    objVbo[objIbo[threadInd] * strdFlt + offst + 1] = in_vec3.y;
    objVbo[objIbo[threadInd] * strdFlt + offst + 2] = in_vec3.z;

}

__device__
void writeCollFlagArray(bool collision, unsigned int ind_x, unsigned int ind_y) {

    colFlag[ind_x * fxVar.height + ind_y] = collision;
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
            cVar.k * 1.4f * (L - lCoeff*cVar.MxL));
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
        tempV = readFromVBO(ppReadBuff, x - 1, y + 0, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.0f);
    }
    //right
    if ((x + 1) > fxVar.width - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x + 1, y + 0, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.0f);
    }
    //up
    if ((y) < 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x + 0, y - 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.0f);
    }
    //down
    if ((y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x + 0, y + 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.0f);
    }

    //shear neighbor
    //left up
    if ((x) < 1 || (y) < 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x - 1, y - 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.41421356237f);
    }
    //left down
    if ((x) < 1 || (y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x - 1, y + 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.41421356237f);
    }
    //right up
    if ((x + 1) > fxVar.width - 1 || (y) < 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x + 1, y - 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.41421356237f);
    }
    //right down
    if ((x + 1) > fxVar.width - 1 || (y + 1) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x + 1, y + 1, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 1.41421356237f);
    }

    //bend neighbor
    //left 2
    if ((x) < 2) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x - 2, y + 0, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 2.0f);
    }
    //right 2
    if ((x + 2) > fxVar.width - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x + 2, y + 0, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 2.0f);
    }
    //up 2
    if ((y) < 2) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x + 0, y - 2, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 2.0f);
    }
    //down 2
    if ((y + 2) > fxVar.height - 1) { innF += glm::vec3(0.0f); }
    else {
        tempV = readFromVBO(ppReadBuff, x + 0, y + 2, fxVar.OffstPos) - curP;
        innF += constraintForce(tempV, 2.0f);
    }

    //for color mode
    //if (cVar.colorMode == 0) {
    //    glm::vec3 col = glm::vec3(4.0f*glm::length(innF) , 0.3f, 0.8f - 3.0f*glm::length(innF));
    //    writeToVBO(col, ppWriteBuff, x, y, fxVar.OffstCol);
    //}
    //else { writeToVBO(glm::vec3(0.961f, 0.961f, 0.863f), ppWriteBuff, x, y, fxVar.OffstCol); }
    /////////////////////////////////////////////////////////////////////////////////





    //float a = cVar.in_testFloat;
    //glm::vec3 testcol = glm::vec3(0.0f, a, a);
    if (x == 0 && y == 0) {

 /*       printf(" objVar.stride = %d \n", objVar.vboStrdFlt);
        printf(" objVar.offset pos= %d \n", objVar.OffstPos);
        printf(" objVar.offset normal = %d \n", objVar.OffstNm);
        printf(" objVar.offset color = %d \n", objVar.OffstCol);
        printf(" objVar.nVerts = %d \n", objVar.nVerts);
        printf(" objVar.nInd = %d \n", objVar.nInd);*/

        //for (int i = 0; i < objVar.nVerts; i++) {

        //    objBuff[i * objVar.vboStrdFlt + objVar.OffstCol + 0] = glm::sin(cVar.time);
        //    objBuff[i * objVar.vboStrdFlt + objVar.OffstCol + 1] = glm::sin(cVar.time);

        //}

        //objBuff[9] += 0.1f;
        //objBuff[11] += 0.01f*glm::sin(cVar.time);
        //objBuff[8] += 0.1f;


    }
    return innF;
}

__device__ 
glm::vec3 computeForceNet(glm::vec3 currPos, float* readBuff, float* writeBuff, 
                            unsigned int x, unsigned int y  ) {
    
    
    glm::vec3 innF = computeInnerForce(ppReadBuff, ppWriteBuff, x, y, currPos);

    glm::vec3 vel = readFromVBO(ppReadBuff, x, y, fxVar.OffstVel);

    glm::vec3 Fwind = cVar.WStr *
        glm::vec3(1.0f + cVar.WDir.x * glm::sin(cVar.offsCo.x * currPos.z + cVar.cyclCo.x * cVar.time),
            cVar.WDir.y * glm::sin(cVar.offsCo.y * currPos.y + cVar.cyclCo.y * cVar.time),
            cVar.WDir.z * glm::cos(cVar.offsCo.z * currPos.y + cVar.cyclCo.z * cVar.time));
    
    //***********************************
    //F = m*g + Fwind - air * vel* vel + innF - damp = m*Acc;
    glm::vec3 netF =
        + 1.0f * cVar.M * glm::vec3(0.0f, cVar.g, 0.0f)
        + 1.0f * Fwind
        - 1.0f * cVar.a * vel * (glm::length(vel))
        + 1.0f * innF
        - 1.0f * cVar.Dp * vel * (glm::length(vel));

    return netF;
}

__device__ glm::vec3 samplFunc(float stpT, glm::vec3 pos) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > fxVar.width || y > fxVar.height) return;
    
    glm::vec3 ForceNet = computeForceNet(pos, ppReadBuff, ppWriteBuff, x, y);
    glm::vec3 Acc = ForceNet / cVar.M;
    glm::vec3 Vel = Acc * stpT;
    
    return pos + Vel * stpT + 0.5f * Acc * stpT * stpT;
}

__device__
glm::vec3 RungeKutt(float stpT, glm::vec3 pT0) {

    glm::vec3 K1 = pT0;
    glm::vec3 K2 = samplFunc(stpT / 2.0f, pT0 + K1 / 2.0f);
    glm::vec3 K3 = samplFunc(stpT / 2.0f, pT0 + K2 / 2.0f);
    glm::vec3 K4 = samplFunc(stpT, pT0 + K3);

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
        posL = readFromVBO(ppReadBuff, x - 1, y + 0, fxVar.OffstPos);
    }
    //right
    if ((x + 1) > fxVar.width - 1) { posR = curP; }
    else {
        posR = readFromVBO(ppReadBuff, x + 1, y + 0, fxVar.OffstPos);
    }
    //up
    if ((y) < 1) {  posU = curP; }
    else {
        posU = readFromVBO(ppReadBuff, x + 0, y - 1, fxVar.OffstPos);
    }
    //down
    if ((y + 1) > fxVar.height - 1) {  posD = curP; }
    else {
        posD = readFromVBO(ppReadBuff, x + 0, y + 1, fxVar.OffstPos);
    }
    
    return glm::normalize(glm::cross((posR - posL), (posU - posD)));
}

__device__
bool furtherCheck(glm::vec3 pn, float r, glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 n) {

    glm::vec3 S1 = glm::normalize(glm::cross(n, (B - A)));
    glm::vec3 S2 = glm::normalize(glm::cross(n, (C - B)));
    glm::vec3 S3 = glm::normalize(glm::cross(n, (A - C)));
    //glm::vec3 S1 = glm::normalize(glm::cross(n, (A - B)));
    //glm::vec3 S2 = glm::normalize(glm::cross(n, (B - C)));
    //glm::vec3 S3 = glm::normalize(glm::cross(n, (C - A)));
    if ((glm::dot((pn - A), S1) > -r) &&
        (glm::dot((pn - B), S2) > -r) &&
        (glm::dot((pn - C), S3) > -r)) {
        return true;
    }
    else return false;
}

__device__
float getPerpDist(glm::vec3 Pos, float r, glm::vec3 knownP, glm::vec3 n) {
    return glm::dot((Pos - knownP), glm::normalize(n)) - r ;

}

__device__
bool sphrTrigCollision(glm::vec3 pos, glm::vec3 posNext, float r,
    glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 n) {

    float d0 = getPerpDist(pos, r, A, n);
    float dn = getPerpDist(posNext, r, A, n);
    //printf("dn = %f \n", dn);

    if (d0 * dn > 0) return false;
    else if (d0 * dn == 0)
    {
        if (dn >= 0) return false;
        else if (dn < 0) {
            return furtherCheck(posNext, r, A, B, C, n);
        }
    }
    else if (d0 * dn < 0) 
    {
        if (dn > 0) return false;
        if (dn < 0) return furtherCheck(posNext, r, A, B, C, n);
    }

}

__device__
void clothObjCollision(glm::vec3 Pos, glm::vec3& NextPos, unsigned int x, unsigned int y) {

    float r = 0.005f;

    for (int i = 0; i < objVar.nTrig - 1; i++) {
        
        //triangles
        //glm::vec3 A = readObjVbo(3* i + 0, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        //glm::vec3 B = readObjVbo(3* i + 1, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        //glm::vec3 C = readObjVbo(3* i + 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);

        //triangle strip
        glm::vec3 A = readObjVbo(i + 0, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        glm::vec3 B = readObjVbo(i + 1 + (i + 1) % 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        glm::vec3 C = readObjVbo(i + 1 + i % 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        glm::vec3 n = objN[i];

        if (sphrTrigCollision(Pos, NextPos, r, A, B, C, n)) {
            
            //if collision true write to the bool array
            colFlag[x * fxVar.height + y] = true;
            collCount[x * fxVar.height + y] = 0;

            //if (x == 30 && y == 40) {
            //    printf("collision status is %d \n", colFlag[x * fxVar.height + y]);

            //}

            //writeToVBO(glm::vec3(1.0f, 0.0f, 1.0f), ppWriteBuff, x, y, fxVar.OffstCol);

            float dn = getPerpDist(NextPos, r, A, n);
            NextPos += 1.01f * (-dn) * n;


            break;

        }
        else {


              //TODO: make a bool array that stores the state of collision
              //this frame false, last frame true
            if (colFlag[x * fxVar.height + y]) {// if last frame collision is true
                writeToVBO(glm::vec3(0.0f), ppWriteBuff, x, y, fxVar.OffstVel);
                collCount[x * fxVar.height + y] = 10;
            }
            else {
                if (collCount[x * fxVar.height + y] >9) {
                    collCount[x * fxVar.height + y] += 1;
                    //writeToVBO(glm::vec3(0.0f), ppWriteBuff, x, y, fxVar.OffstVel);
                }

            
            }
              //writeToVBO(glm::vec3(0.0f), ppWriteBuff, x, y, fxVar.OffstCol);
              
              //if collision false write to the bool array
              colFlag[x * fxVar.height + y] = false;
              
              //if (x == 30 && y == 40) {
              //    printf("collision status is %d \n", colFlag[x * fxVar.height + y]);

              //}
        }

    }

   




}



__global__
void computeParticlePos_Kernel(unsigned int width,
    unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height) return;


    //current pos and last frame pos
    glm::vec3 Pos = readFromVBO(ppReadBuff, x, y, fxVar.OffstPos);
    glm::vec3 lastPos = readFromVBO(ppWriteBuff, x, y, fxVar.OffstPos);
    //normal
    glm::vec3 normal = ComputeNomral(ppReadBuff, x, y, Pos);
    writeToVBO(normal, ppWriteBuff, x, y, fxVar.OffstNm);
    
    //ForceNet
    glm::vec3 ForceNet = computeForceNet(Pos, ppReadBuff, ppWriteBuff, x, y);
    //Acceleration
    glm::vec3 Acc = ForceNet/cVar.M;

    //velocity
    glm::vec3 Vel = readFromVBO(ppReadBuff, x, y, fxVar.OffstVel);
    Vel += Acc * cVar.stp;
    writeToVBO(!cVar.frz ? Vel: glm::vec3(0.0f), ppWriteBuff, x, y, fxVar.OffstVel);

    //damping
    glm::vec3 ForceDamp = -cVar.Dp * Vel * (glm::length(Vel));

    ForceNet += ForceDamp;
    //Acc = ForceNet / cVar.M;

    glm::vec3 nextPos;

    if ((x == 0 && y == 0) || (x == 0 && y == height - 1) ||
        (x == 0 && y == height / 4) || (x == 0 && y == 3 * height / 4)
        ) {

        //glm::vec3 dir = glm::vec3(0.0f, 0.0f, 0.5f * fxVar.height / 10.0f) - Pos;
        glm::vec3 dir = glm::vec3(1.0f, 0.0f, 0.0f);

        nextPos = Pos + 0.001f* dir * cVar.folding;
    }
    else {

        //nextPos = RungeKutt(cVar.stp, Pos, Vel, Acc);
        nextPos = !cVar.frz? VerletAlg(Pos, lastPos, Acc, cVar.stp): Pos;
        //nextPos = Pos + Vel;
        
    }

    clothObjCollision(Pos, nextPos, x, y); 

    //color collision
    if(colFlag[x * fxVar.height + y]) writeToVBO(glm::vec3(1.0f, 0.0f, 1.0f), ppWriteBuff, x, y, fxVar.OffstCol);
    else writeToVBO(glm::vec3(0.0f, 0.0f, 0.0f), ppWriteBuff, x, y, fxVar.OffstCol);


    if (collCount[x * fxVar.height + y] >= 50) {
        collCount[x * fxVar.height + y] = 0;
        writeToVBO(glm::vec3(0.0f), ppWriteBuff, x, y, fxVar.OffstVel);
        nextPos = Pos;
    }

    if (x == 30 && y == 41) {
        printf("collision status is %d \n", colFlag[x * fxVar.height + y]);
        printf("collision count = %d \d", collCount[x * fxVar.height + y]);
    }


    writeToVBO(nextPos, ppWriteBuff, x, y, fxVar.OffstPos);

}


void Cloth_Launch_Kernel(const unsigned int mesh_width, const unsigned int mesh_height)
{
    dim3 block(32, 32, 1);
    dim3 grid(ceil(mesh_width / block.x), ceil(mesh_height / block.y), 1);

    //std::cout << " readBuff = " << readBuff << std::endl;
    //std::cout << "ppReadBuff = " << ppReadBuff << std::endl;

    computeParticlePos_Kernel << < grid, block >> > (mesh_width, mesh_height);
    CheckCudaErr("simple_vbo_kernel launch fail ");

    cudaDeviceSynchronize();
    CheckCudaErr("cudaDeviceSynghconize fail ");


    //std::cout << "objConst. vboStdinFlt = " << objVar.vboStrdFlt << std::endl;



}



__global__
void preCompObjNm_Kernel() {
    
    //TRIANGLES
    //unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    //if (x > objVar.nTrig - 1) return;
    //glm::vec3 p1, p2, p3;

    //p1 = readObjVbo(x * 3 + 0, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
    //p2 = readObjVbo(x * 3 + 1, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
    //p3 = readObjVbo(x * 3 + 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
    //objN[x] = glm::normalize(glm::cross((p3 - p1), (p2 - p1)));

    //writeObjVbo(objN[x], x * 3 + 0, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstCol);
    //writeObjVbo(objN[x], x * 3 + 1, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstCol);
    //writeObjVbo(objN[x], x * 3 + 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstCol);

    //printf("this is thread%d \n", x);
  
    //if (x == 3) {
    //  printf("objVar.vboStrdinflt = %d \n", objVar.vboStrdFlt);
    //  printf("objVar.nTrig = %d \n", objVar.nTrig);


    //  for (int i = 0; i < objVar.nTrig ; i++) {
    //      printf(" objN[%d].x = %f \n", i, objN[i].x);
    //      printf(" objN[%d].y = %f \n" , i, objN[i].y);
    //      printf(" objN[%d].z = %f \n", i, objN[i].z);
    //  }

    //}


    //TRIANGLE STRIP METHOD==========================================
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > objVar.nTrig -1 ) return;

    glm::vec3 p1, p2, p3;
    
    if (x % 2 == 0) {
        p1 = readObjVbo(x, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        p2 = readObjVbo(x + 1, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        p3 = readObjVbo(x + 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
    }
    else {
        p1 = readObjVbo(x, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        p2 = readObjVbo(x + 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
        p3 = readObjVbo(x + 1, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos);
    }

    objN[x] = glm::normalize(glm::cross((p3 - p1), (p2 - p1)));
    if (x == 3) {
     printf("objVar.vboStrdinflt = %d \n", objVar.vboStrdFlt);
     printf("objVar.nTrig = %d \n", objVar.nTrig);


     for (int i = 0; i < objVar.nTrig ; i++) {
         printf(" objN[%d].x = %f \n", i, objN[i].x);
         printf(" objN[%d].y = %f \n" , i, objN[i].y);
         printf(" objN[%d].z = %f \n", i, objN[i].z);
     }

   }
  
    ////==================================================================
}

void ComptObjNormal_Kernel(unsigned int nTriangles) {

    //std::cout << "objVar.vboStrdFlt = " << objVar.vboStrdFlt << std::endl;
    std::cout << "kernel n triangle pass in = " << nTriangles << std::endl;

    dim3 blckD(32, 1, 1);
    dim3 grdD((int)(nTriangles/blckD.x) + 1, 1, 1);

    preCompObjNm_Kernel << <grdD, blckD >> > ();
    CheckCudaErr("preCompute Obj Normal launch fail ");

    cudaDeviceSynchronize();
    CheckCudaErr("preCompute Obj Normal launch fail; cuda Device Synghconize fail ");

}

//void debugPrint() {
//
//
//
//    if (x == 0) {
//
//
//        printf(" objVar.stride = %d \n", objVar.vboStrdFlt);
//        printf(" objVar.offset pos= %d \n", objVar.OffstPos);
//        printf(" objVar.offset normal = %d \n", objVar.OffstNm);
//        printf(" objVar.offset color = %d \n", objVar.OffstCol);
//        printf(" objVar.nVerts = %d \n", objVar.nVerts);
//        printf(" objVar.nInd = %d \n", objVar.nInd);
//
//
//        printf(" p1.x = %f \n", p1.x);
//        printf(" p1.y = %f \n", p1.y);
//        printf(" p1.x = %f \n", p1.z);
//        printf(" p2.x = %f \n", p2.x);
//        printf(" p2.x = %f \n", p2.y);
//        printf(" p2.x = %f \n", p2.z);
//        printf(" p3.x = %f \n", p3.x);
//        printf(" p3.x = %f \n", p3.y);
//        printf(" p3.x = %f \n", p3.z);
//
//
//        for (int i = 0; i < objVar.nInd; i++) {
//
//            printf("objind[%d] = %d \n", i, objIndBuff[i]);
//
//        }
//        for (int i = 0; i < objVar.nInd - 2; i++) {
//
//            printf("pos[%d].x = %f \n", i, readObjVbo(i, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos).x);
//            printf("pos[%d].y = %f \n", i, readObjVbo(i, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos).y);
//            printf("pos[%d].z = %f \n", i, readObjVbo(i, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos).z);
//
//
//        }
//
//        for (int i = 0; i < objVar.nInd - 2; i++) {
//
//            printf(" obj normal %d.x = %f \n", i, objN[i].x);
//            printf(" obj normal %d.y = %f \n", i, objN[i].y);
//            printf(" obj normal %d.z = %f \n", i, objN[i].z);
//
//        }
//
//
//    }
//
//
//
//
//}

