#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <glad/glad.h>
#include "util.hpp"


class clothRender
{
public:
	clothRender();
	
	void initCloth(const unsigned int numVertsWidth, 
		const unsigned int numVertsHeight, GLuint attribLoc);
	void CudaUpdateCloth(float in_time);
	void DrawCloth();
	
	////////////////////////////////////////////////////
	//OLD
	void initVAO(float offset, GLuint shaderAttribLoc);
	void drawVAO(GLuint VAO_ID);

private:
	unsigned int test_width;
	unsigned int test_height;

	//cuda test
	GLuint test_cudaVAO;
	cudaGraphicsResource *test_CudaVboRes;

	//////////////////////////////////////////////////////
	//OLD=============================================
	const int numGrid = 32;
	GLuint _assignVAOID;

};

