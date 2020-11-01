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
	void CustomObj::CreateImptObjVbo(std::vector<float>* vertPos,
		std::vector<float>* uv, std::vector<float>* normal,
		std::vector<unsigned int>* indices, unsigned int numOfFloat,
		unsigned int numOfIndices);

	void DrawObjStrip(GLenum PrimitiveType, GLenum PolygonMode);
	void passObjPtrToKernel();
	void unmapResource();

	~CustomObj();

private:
	GLuint VAO, VBO, IBO;
	GLsizei indexCount;

	cudaGraphicsResource* cstmObjRes;
	cudaGraphicsResource* ObjIboRes;

	objConst objConst;


	void Clearobj();
};

