#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
//using namespace std;
#include <stdio.h>
//#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>

//glm
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "util.hpp"


void ComptObjNormal_Kernel(unsigned int nTriangles);

void Cloth_Launch_Kernel(unsigned int mesh_width,
	unsigned int mesh_height);
void CheckCudaErr(const char* msg);
//from cloth to kernel
void passPPbuffPtr(float* d_vbo1, float* d_vbo2);
void updateClothConst(ClothConstant* in_passVar);
void copyFixClothConst(FixedClothConstant* in_fxConst);
void copyCollisionArrayPtr(bool* d_collPtr, int* d_collCountPtr);

//from customized obj to kernel
void passCstmObjPtr(float* d_vbo, unsigned int* d_ibo, glm::vec3* d_objN);
void cpyObjConst(objConst* in_Var);