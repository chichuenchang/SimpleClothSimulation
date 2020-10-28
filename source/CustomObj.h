#pragma once
#include "util.hpp"

class CustomObj
{
public:

	CustomObj();

	void CreateVbo(float* vertices, unsigned int* indices, 
		unsigned int numOfFloat, unsigned int numOfIndices);
	void CreateVboVector(std::vector<float> vertices, 
		std::vector<unsigned int> indices, unsigned int numOfFloat, 
		unsigned int numOfIndices);

	void DrawObjStrip();
	void passVboPtrKernel();
	void unmapResource();

	~CustomObj();

private:
	GLuint VAO, VBO, IBO;
	GLsizei indexCount;

	cudaGraphicsResource* cstmObjRes;

	objConst objConst;


	void Clearobj();
};

