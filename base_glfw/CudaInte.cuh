#pragma once

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

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


void launch_kernel(glm::vec3* pos, unsigned int mesh_width,
	unsigned int mesh_height, float time);

void test_launch_kernel(glm::vec1* pos, unsigned int mesh_width,
	unsigned int mesh_height, float time);

