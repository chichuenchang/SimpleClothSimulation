#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <glad/glad.h>
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
	unsigned int test_width;
	unsigned int test_height;

	//cuda test
	GLuint test_cudaVAO;
	cudaGraphicsResource* test_CudaVboRes;
	//index buffer size
	unsigned int indexBuffSize;

};

