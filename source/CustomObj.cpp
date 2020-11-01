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
	objConst.nInd = numOfIndices;
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
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cstmObjRes, VBO, cudaGraphicsMapFlagsReadOnly));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&ObjIboRes, IBO, cudaGraphicsMapFlagsReadOnly));

}

void CustomObj::CreateVbo(float* vertices, unsigned int* indices, unsigned int numOfFloat, unsigned int numOfIndices) {

	objConst.vboStrdFlt = 11;
	objConst.OffstPos = 0;
	objConst.OffstNm = 5;
	objConst.OffstCol = 8;
	objConst.nVerts = numOfFloat/11;
	objConst.nInd = numOfIndices;
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

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 11, 0);
	glEnableVertexAttribArray(0);						   
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 11, (void*)(sizeof(float) * 3));
	glEnableVertexAttribArray(1);						   							   
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 11, (void*)(sizeof(float) * 5));
	glEnableVertexAttribArray(2);						   							   
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 11, (void*)(sizeof(float) * 8));
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//only for read in cuda
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cstmObjRes, VBO, cudaGraphicsRegisterFlagsReadOnly));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&ObjIboRes, IBO, cudaGraphicsMapFlagsNone));

}

void CustomObj::CreateImptObjVbo(std::vector<float>* vertPos,
	std::vector<float>* uv, std::vector<float>* normal,
	std::vector<unsigned int>* indices, unsigned int numOfFloat,
	unsigned int numOfIndices)
{
	//rearrange data to interleave========================
	unsigned int nEntries = vertPos->size() / 3;
	std::vector<float> impVbo;
	for (int i = 0; i < nEntries - 1; i++) {
		impVbo.push_back(vertPos->at(3 * size_t(i) + 0));
		impVbo.push_back(vertPos->at(3 * size_t(i) + 1));
		impVbo.push_back(vertPos->at(3 * size_t(i) + 2));
		//		impVbo.push_back(uv->at(2 * size_t(i) + 0));
		//		impVbo.push_back(uv->at(2 * size_t(i) + 1));
		impVbo.push_back(0.0f);
		impVbo.push_back(0.0f);
		impVbo.push_back(normal->at(3 * size_t(i) + 0));
		impVbo.push_back(normal->at(3 * size_t(i) + 1));
		impVbo.push_back(normal->at(3 * size_t(i) + 2));
	}
	//================================================

	std::cout << "n float element = " << impVbo.size() << std::endl;

	objConst.vboStrdFlt = 8;
	objConst.OffstPos = 0;
	objConst.OffstNm = 5;
	objConst.OffstCol = 0;
	objConst.nVerts = numOfFloat / 8;
	objConst.nInd = numOfIndices;

	cpyObjConst(&objConst);

	Clearobj();

	indexCount = numOfIndices;

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &IBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices->size(), indices->data(), GL_DYNAMIC_DRAW);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * impVbo.size(), impVbo.data(), GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, 0);
	glEnableVertexAttribArray(0);						   
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 3));
	glEnableVertexAttribArray(1);							  						  
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 5));
	glEnableVertexAttribArray(2);							  						  
	//glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 8));
	//glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//only for read in cuda
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cstmObjRes, VBO, cudaGraphicsRegisterFlagsReadOnly));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&ObjIboRes, IBO, cudaGraphicsMapFlagsNone));

}



void CustomObj::DrawObjStrip(GLenum PrimitiveType, GLenum PolygonMode) {
	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glPolygonMode(GL_FRONT_AND_BACK, PolygonMode);
	glDrawElements(PrimitiveType, indexCount, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

//call before kernel
void CustomObj::passObjPtrToKernel() {

	float* d_ObjPtr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &cstmObjRes, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_ObjPtr, &num_bytes, cstmObjRes));
	
	unsigned int* d_iboPtr;
	size_t num_bytes2;
	checkCudaErrors(cudaGraphicsMapResources(1, &ObjIboRes, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_iboPtr, &num_bytes2, ObjIboRes));

	glm::vec3* d_objNormal;
	checkCudaErrors(cudaMalloc((void**)&d_objNormal, ((long long)indexCount - 2) * sizeof(glm::vec3)));
	
	passCstmObjPtr(d_ObjPtr, d_iboPtr, d_objNormal);
}

//call after kernel
void CustomObj::unmapResource() {
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cstmObjRes, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &ObjIboRes, 0));
}