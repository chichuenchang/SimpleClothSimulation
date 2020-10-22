#pragma once
//#include "util.hpp"
#include "Stdfx.h"

class testClothRender
{
public:
	testClothRender();

	void initCloth(const unsigned int numVertsWidth,
		const unsigned int numVertsHeight, GLuint attribLoc);
	void CudaUpdateCloth(float in_time);
	void DrawCloth();


	//test
	void passVarToCudaConst(float in_variable);
	//test

private:
	unsigned int cloth_width;
	unsigned int cloth_height;

	GLuint cudaVAO;
	GLuint cudaVBO;
	cudaGraphicsResource* CudaVboRes;
	unsigned int VBOStrideInFLoat;
	unsigned int sizeOfVerts;
	//ibo
	unsigned int indexBuffSize;
	const int RestartInd = 9999999;

	void initVBO(GLuint AttribLocation);

	//struct passVar {

	//float testfloat;

	//}testPassVar;


	//void copyConstMem(passVar in_passVar);



};

