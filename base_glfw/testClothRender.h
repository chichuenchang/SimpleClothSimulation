#pragma once
#include "util.hpp"
//#include "Stdfx.h"

class testClothRender
{
public:
	testClothRender();

	void initCloth(const unsigned int numVertsWidth,
		const unsigned int numVertsHeight, GLuint attribLoc,
		ClothConstant& clthConst, FixedClothConstant& fxConst);
	void CudaUpdateCloth(ClothConstant clothConst);
	void DrawCloth();


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
	void initClothConstValue(ClothConstant& clothConst, FixedClothConstant& fxClothConst);

};

