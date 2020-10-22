
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


class ClothParticles {

public:
	ClothParticles();

private:
	glm::vec3 pos;
	glm::vec3 vel;
	glm::vec3 acc;
	float mass;
	glm::vec3 gravity;
	//
	bool constraint;

	glm::vec3 Force_Wind;
	glm::vec3 Air_resis;

	__device__ glm::vec3 ComputeInnerForce();
	__device__ glm::vec3 ComputeNetForce();



};