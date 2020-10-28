#include "CustomObj.h"
#include "CudaInte.cuh"

CustomObj::CustomObj() {

	VAO = -1;
	VBO = -1;
	IBO = -1;

	indexCount = 0;
	cstmObjRes = nullptr;

	objConst.vboStrdFlt = 0;
	objConst.OffstPos = 0;
	objConst.OffstNm = 0;
	objConst.OffstCol = 0;
}

CustomObj::~CustomObj() {

	Clearobj();
}

void CustomObj::Clearobj() {

	if (IBO != 0)
	{
		glDeleteBuffers(1, &IBO);
		IBO = - 1;
	}

	if (VBO != 0)
	{
		glDeleteBuffers(1, &VBO);
		VBO = -1;
	}

	if (VAO != 0)
	{
		glDeleteVertexArrays(1, &VAO);
		VAO = -1;
	}

	indexCount = 0;
}



void CustomObj::CreateVboVector(std::vector<float> vertices,
	std::vector<unsigned int> indices, unsigned int numOfFloat,
	unsigned int numOfIndices){

	objConst.vboStrdFlt = 11;
	objConst.OffstPos = 0;
	objConst.OffstNm = 5;
	objConst.OffstCol = 8;
	objConst.nVerts = numOfFloat / 11;
	cpyObjConst(&objConst);

	Clearobj();

	indexCount = numOfIndices;

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &IBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * numOfIndices, indices.data(), GL_DYNAMIC_DRAW);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * numOfFloat, vertices.data(), GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]) * 11, 0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]) * 11, (void*)(sizeof(vertices[0]) * 3));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]) * 11, (void*)(sizeof(vertices[0]) * 5));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]) * 11, (void*)(sizeof(vertices[0]) * 8));
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//only for read in cuda
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cstmObjRes, VBO, cudaGraphicsMapFlagsNone));

}

void CustomObj::CreateVbo(float* vertices, unsigned int* indices, unsigned int numOfFloat, unsigned int numOfIndices) {

	objConst.vboStrdFlt = 11;
	objConst.OffstPos = 0;
	objConst.OffstNm = 5;
	objConst.OffstCol = 8;
	objConst.nVerts = numOfFloat/11;
	cpyObjConst(&objConst);

	Clearobj();

	indexCount = numOfIndices;

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &IBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * numOfIndices, indices, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * numOfFloat, vertices, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]) * 11, 0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]) * 11, (void*)(sizeof(vertices[0]) * 3));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]) * 11, (void*)(sizeof(vertices[0]) * 5));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]) * 11, (void*)(sizeof(vertices[0]) * 8));
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//only for read in cuda
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cstmObjRes, VBO, cudaGraphicsRegisterFlagsReadOnly));
}

void CustomObj::DrawObjStrip() {

	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glDrawElements(GL_TRIANGLE_STRIP, indexCount, GL_UNSIGNED_INT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

}

//call before kernel
void CustomObj::passVboPtrKernel() {

	float* d_ObjPtr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &cstmObjRes, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_ObjPtr, &num_bytes, cstmObjRes));

	passCstmObjPtr(d_ObjPtr);
}

//call after kernel
void CustomObj::unmapResource() {
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cstmObjRes, 0));
}