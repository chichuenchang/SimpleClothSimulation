#pragma once
#include "util.hpp"

class testClothRender
{
public:
	testClothRender();

	void initCloth(const unsigned int numVertsWidth,
		const unsigned int numVertsHeight, GLuint attribLoc);
	void CudaUpdateCloth(float in_time);
	void DrawCloth();

private:
	unsigned int cloth_width;
	unsigned int cloth_height;

	GLuint test_cudaVAO;
	cudaGraphicsResource* test_CudaVboRes;
	unsigned int VBOStrideInFLoat;
	//ibo
	unsigned int indexBuffSize;
	const int RestartInd = 9999999;

};

