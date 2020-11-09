#include "testClothRender.h"
#include "CudaInte.cuh"


testClothRender::testClothRender()
{
	cloth_width = 0;
	cloth_height = 0;

	cudaVAO1 = -1;
	cudaVAO2 = -1;
	cudaVBO1 = -1;
	cudaVBO2 = -1;
	AssignIBO = -1;
	CudaVboRes1 = nullptr;
	CudaVboRes2 = nullptr;
	inAttributeLocation = -1;
	VBOStrideInFloat = sizeof(testVert) / sizeof(float);

	indexBuffSize = 0;

	pp = false;
	resetClothFlag = false;

	DrawPolygonMode = 0;


}

void testClothRender::setUpClothBuffer() {
	///////////////////////////////////////////////////////////
	//VBO#1
	glBindVertexArray(cudaVAO1);

	// create buffer object
	glBindBuffer(GL_ARRAY_BUFFER, cudaVBO1);
	glBufferData(GL_ARRAY_BUFFER, testGrid.size() * sizeof(testVert), testGrid.data(), GL_DYNAMIC_DRAW);

	// register this buffer object with CUDA
	glEnableVertexAttribArray(inAttributeLocation);//layout location = attribLoc in vs
	glVertexAttribPointer(inAttributeLocation, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), 0);
	glEnableVertexAttribArray(inAttributeLocation + 1);//layout location = attribLoc in vs
	glVertexAttribPointer(inAttributeLocation + 1, 2, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, texCrd));
	glEnableVertexAttribArray(inAttributeLocation + 2);//layout location = attribLoc in vs
	glVertexAttribPointer(inAttributeLocation + 2, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, normal));
	glEnableVertexAttribArray(inAttributeLocation + 3);//layout location = attribLoc in vs
	glVertexAttribPointer(inAttributeLocation + 3, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, col));


	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, AssignIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexData.size() * sizeof(unsigned int), IndexData.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	///////////////////////////////////////////////////////////
	//VBO#2
	glBindVertexArray(cudaVAO2);
	// create buffer object
	glBindBuffer(GL_ARRAY_BUFFER, cudaVBO2);
	glBufferData(GL_ARRAY_BUFFER, testGrid.size() * sizeof(testVert), testGrid.data(), GL_DYNAMIC_DRAW);

	// register this buffer object with CUDA
	glEnableVertexAttribArray(inAttributeLocation);//layout location = attribLoc in vs
	glVertexAttribPointer(inAttributeLocation, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), 0);
	glEnableVertexAttribArray(inAttributeLocation + 1);//layout location = attribLoc in vs
	glVertexAttribPointer(inAttributeLocation + 1, 2, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, texCrd));
	glEnableVertexAttribArray(inAttributeLocation + 2);//layout location = attribLoc in vs
	glVertexAttribPointer(inAttributeLocation + 2, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, normal));
	glEnableVertexAttribArray(inAttributeLocation + 3);//layout location = attribLoc in vs
	glVertexAttribPointer(inAttributeLocation + 3, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, col));



	unsigned int AssignIBO2;
	glGenBuffers(1, &AssignIBO2);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, AssignIBO2);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexData.size() * sizeof(unsigned int), IndexData.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


}

void testClothRender::fillBufferData() {
	//vbo data
	testGrid.clear();
	for (int i = 0; i < cloth_width; i++) {
		for (int j = 0; j < cloth_height; j++) {
			testGrid.push_back({glm::vec3((float)i * 0.02f, 0.0f, (float)j * 0.02f), //pos
								glm::vec2((float)i / (float)(cloth_width - 1), (float)j / (float)(cloth_height - 1)),//texCoord
								glm::vec3(0.0f, 1.0f, 1.0f),//
								glm::vec3(0.961f, 0.961f, 0.863f),//
								glm::vec3(0.0f)});	//for kernel to write velocity

		}
	}

	//IBO
	IndexData.clear();
	for (int i = 0; i < cloth_width - 1; i++)
	{
		for (int j = 0; j < cloth_height; j++)
		{
			IndexData.push_back(i * cloth_height + j);
			IndexData.push_back(i * cloth_height + j + cloth_height);
		}
		IndexData.push_back(RestartInd);
	}
	//IBO size
	indexBuffSize = IndexData.size();
}

void testClothRender::initVBO(GLuint in_attribLoc) {

	fillBufferData();
	VBOStrideInFloat = sizeof(testVert) / sizeof(float);
	inAttributeLocation = in_attribLoc;

	if (cudaVBO1 > -1) {
		glDeleteBuffers(1, &cudaVAO1);
		glDeleteBuffers(1, &cudaVAO2);
		glDeleteBuffers(1, &cudaVBO1);
		glDeleteBuffers(1, &cudaVBO2);
		glDeleteBuffers(1, &AssignIBO);
	}
	else 
	{
		//gen buffer obj IDs
		glGenVertexArrays(1, &cudaVAO1);
		glGenVertexArrays(1, &cudaVAO2);
		glGenBuffers(1, &cudaVBO1);
		glGenBuffers(1, &cudaVBO2);
		glGenBuffers(1, &AssignIBO);
	}

	setUpClothBuffer();

}

void testClothRender::initClothConstValue(ClothConstant& clothConst, FixedClothConstant& fxClothConst,
	unsigned int clothW, unsigned int clothH) {
	
	//clothConst.WStr = 0.0f;
	//clothConst.WDir = glm::vec3(0.874f, 0.68f, 0.01f);
	//clothConst.offsCo = glm::vec3(19.347f, 7.36f, 1.06f);
	//clothConst.cyclCo = glm::vec3(3.201f, 1.71f, 1.92f);

	//clothConst.M = 0.001f;
	//clothConst.g = -10.0f;
	//clothConst.k = 45.0f;
	//clothConst.rLen = 0.02f;
	//clothConst.MxL = 0.025f;

	//clothConst.stp = 0.000001f;
	//clothConst.dt = 0.000005f;
	//clothConst.time = 0.0f;
	//clothConst.a = 1500.0f;
	//clothConst.Dp = 2000.0f;
	//clothConst.folding = 0.000f;
	//clothConst.frz = false;
	//clothConst.colorMode = 3;
	
	//verlet
	clothConst.WStr = 0.0f;
	clothConst.WDir = glm::vec3(0.874f, 0.68f, 0.01f);
	clothConst.offsCo = glm::vec3(19.347f, 7.36f, 1.06f);
	clothConst.cyclCo = glm::vec3(3.201f, 1.71f, 1.92f);

	clothConst.M = 0.01f;
	clothConst.g = -10.0f;
	clothConst.k = 100.0f;
	clothConst.rLen = 0.02f;
	clothConst.MxL = 0.03f;

	clothConst.stp = 0.004f;
	clothConst.dt = 0.000005f;
	clothConst.time = 0.0f;
	clothConst.a = 0.15f;
	clothConst.Dp = 0.15f;
	clothConst.folding = 0.000f;
	clothConst.frz = false;
	clothConst.colorMode = 3;

	fxClothConst.width = clothW;
	fxClothConst.height = clothH;
	fxClothConst.vboStrdFlt = VBOStrideInFloat;
	//by the layout in vbo
	fxClothConst.OffstPos = 0;
	fxClothConst.OffstNm = 5;
	fxClothConst.OffstCol = 8;
	fxClothConst.OffstVel = 11;
	fxClothConst.sphR = 0.01f; 
	fxClothConst.cellUnit = fxClothConst.sphR;
	//fxclothConst.spcSt and spaceMX are the BB of space
	fxClothConst.spcSt = glm::vec3(-0.2f, -1.0f, -0.1f);
	glm::vec3 spaceMx = glm::vec3(1.0f, 0.1f, 1.4f);
	fxClothConst.spcDim = glm::vec3(
		(spaceMx.x - fxClothConst.spcSt.x) / fxClothConst.cellUnit, 
		(spaceMx.y - fxClothConst.spcSt.y) / fxClothConst.cellUnit,
		(spaceMx.z - fxClothConst.spcSt.z) / fxClothConst.cellUnit);
	std::cout << "spaceDim.x = " << fxClothConst.spcDim.x << " spaceDim. y = " << fxClothConst.spcDim.y <<
		"spaceDim.z = " << fxClothConst.spcDim.z << std::endl;
	std::cout << "spaceDim x *y *z = " << fxClothConst.spcDim.x * fxClothConst.spcDim.y * fxClothConst.spcDim.z << std::endl;
	cloth_width = clothW;
	cloth_height = clothH;

	//make sure all are initialized
	copyFixClothConst(&fxClothConst);
	updateClothConst(&clothConst);

	//cloth obj collision
	bool* d_collisionPtr;
	checkCudaErrors(cudaMalloc((void**)&d_collisionPtr, size_t(cloth_width) * size_t(cloth_height) * sizeof(bool)));
	
	int* d_collCountPtr;
	checkCudaErrors(cudaMalloc((void**)&d_collCountPtr, size_t(cloth_width) * size_t(cloth_height) * sizeof(int)));

	//cloth cloth collision
	bool* d_clthClthCollMarkPtr;
	checkCudaErrors(cudaMalloc((void**)&d_clthClthCollMarkPtr, size_t(cloth_width) * size_t(cloth_height) * sizeof(bool)));

	unsigned int* d_cellHashArryPtr;//size = particle number, holds cell ID
	checkCudaErrors(cudaMalloc((void**)&d_cellHashArryPtr, size_t(cloth_width) * size_t(cloth_height) * sizeof(unsigned int)));

	unsigned int* d_cellArrayPtr;//size = cell number; holds particle ID
	checkCudaErrors(cudaMalloc((void**)&d_cellArrayPtr, size_t(fxClothConst.spcDim.x) * size_t(fxClothConst.spcDim.y)* size_t(fxClothConst.spcDim.z) * sizeof(unsigned int)));

	int* d_clothClothcollCountPtr;
	checkCudaErrors(cudaMalloc((void**)&d_clothClothcollCountPtr, size_t(cloth_width) * size_t(cloth_height) * sizeof(int)));

	glm::vec3* d_nextPosArray;
	checkCudaErrors(cudaMalloc((void**)&d_nextPosArray, size_t(cloth_width) * size_t(cloth_height) * sizeof(glm::vec3)));
	

	copyArrayPtr(d_collisionPtr, d_collCountPtr, d_clthClthCollMarkPtr, d_clothClothcollCountPtr, d_cellHashArryPtr, d_cellArrayPtr, d_nextPosArray);
}

//creat cuda registered VBO
void testClothRender::initCloth(const unsigned int numVertsWidth, const unsigned int numVertsHeight,
	GLuint attribLoc, ClothConstant& clthConst, FixedClothConstant& fxConst) {
	
	initVBO(attribLoc);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&CudaVboRes1, cudaVBO1, cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&CudaVboRes2, cudaVBO2, cudaGraphicsMapFlagsNone));
}

void testClothRender::passVarsToKernel(ClothConstant in_clothConst) {
	
	updateClothConst(&in_clothConst);

	//std::cout << "value float = " << valuePass.in_testFloat << std::endl;

	float* d_testOutPtr1;
	size_t num_bytes1;
	checkCudaErrors(cudaGraphicsMapResources(1, &CudaVboRes1, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_testOutPtr1, &num_bytes1, CudaVboRes1));
	
	float* d_testOutPtr2;
	size_t num_bytes2;
	checkCudaErrors(cudaGraphicsMapResources(1, &CudaVboRes2, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_testOutPtr2, &num_bytes2, CudaVboRes2));

	passPPbuffPtr(!pp ? d_testOutPtr1 : d_testOutPtr2, !pp ? d_testOutPtr2 : d_testOutPtr1);

}

void testClothRender::unmapResource() {

	pp = !pp;

	checkCudaErrors(cudaGraphicsUnmapResources(1, &CudaVboRes1, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &CudaVboRes2, 0));
}

void testClothRender::PassPolygonMode(int in_polygonMode) {
	 DrawPolygonMode =  in_polygonMode;
}

void testClothRender::DrawCloth() {
	//element draw
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(RestartInd);
	//draw the write buffer
	glBindVertexArray(!pp ? cudaVAO2 : cudaVAO1);

	glPolygonMode(GL_FRONT_AND_BACK, DrawPolygonMode ==0? GL_FILL: GL_LINE);
	glPointSize(3.0f);
	glDrawElements(DrawPolygonMode == 2 ? GL_POINTS : GL_TRIANGLE_STRIP, indexBuffSize, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindVertexArray(0);
}

