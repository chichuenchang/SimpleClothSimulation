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


struct passVar {

	float in_testFloat;

};

void Cloth_Launch_Kernel(float* pos, unsigned int mesh_width,
	unsigned int mesh_height, float time, unsigned int vboStridInFloat);

void CheckCudaErr(const char* msg);

void copyConstMem(passVar *in_passVar);
