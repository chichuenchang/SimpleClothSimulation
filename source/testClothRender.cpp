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
	VBOStrideInFloat = 0;

	indexBuffSize = 0;

	pp = false;
	resetClothFlag = false;

	DrawPolygonMode = 0;
}

void testClothRender::ReloadCloth() {

	resetClothFlag = true;

}

void testClothRender::ResetClothBuffer() {
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
	testGrid.clear();
	testGrid2.clear();

	for (int i = 0; i < cloth_width; i++) {
		for (int j = 0; j < cloth_height; j++) {
			testGrid.push_back({glm::vec3((float)i / 10.0f, 0.0f, (float)j / 10.0f), //pos
								glm::vec2((float)i / (float)(cloth_width - 1), (float)j / (float)(cloth_height - 1)),//texCoord
								glm::vec3(1.0f, 1.0f, 0.0f),//megenta
								glm::vec3(0.0f, 0.0f, 0.0f),//p color megenta
								glm::vec3(0.0f) });	//for kernel to write velocity

		}
	}

	//fill IBO
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
	else {
		//gen buffer obj IDs
		glGenVertexArrays(1, &cudaVAO1);
		glGenVertexArrays(1, &cudaVAO2);
		glGenBuffers(1, &cudaVBO1);
		glGenBuffers(1, &cudaVBO2);
		glGenBuffers(1, &AssignIBO);
	}

	ResetClothBuffer();

}

void testClothRender::initClothConstValue(ClothConstant& clothConst, FixedClothConstant& fxClothConst) {
	//verlet
	clothConst.M = 0.01f;
	clothConst.g = -30.0f;
	clothConst.k = 100.0f;
	clothConst.rLen = 0.1f;

	clothConst.Fw = glm::vec3(0.0f);
	clothConst.WStr = 0.0f;
	clothConst.a = 0.05f;
	clothConst.stp = 0.004f;
	clothConst.dt = 0.000005f;
	clothConst.time = 0.0f;
	clothConst.Dp = 0.05f;
	clothConst.MxL = 0.2f;
	clothConst.in_testFloat = 0.01f;

	fxClothConst.width = cloth_width;
	fxClothConst.height = cloth_height;
	fxClothConst.vboStrdFlt = VBOStrideInFloat;
	//by the layout in vbo
	fxClothConst.OffstPos = 0; 
	fxClothConst.OffstNm = 5;
	fxClothConst.OffstCol = 8;
	fxClothConst.OffstVel = 11;

	//rungekutta
	//clothConst.M = 0.01f;
	//clothConst.g = -10.0f;
	//clothConst.k = 500.0f;
	//clothConst.rLen = 0.1f;
	//clothConst.Fw = glm::vec3(0.0f);
	//clothConst.a = 0.05f;
	//clothConst.stp = 0.004f;
	//clothConst.dt = 0.000005f;
	//clothConst.time = 0.0f;
	//clothConst.MinL = 0.015f;
	//clothConst.MaxL = 0.025f;
	//clothConst.Dp = 0.05f;
	//clothConst.in_testFloat = 0.01f;
	//fxClothConst.width = cloth_width;
	//fxClothConst.height = cloth_height;
	//fxClothConst.vboStrdFlt = VBOStrideInFloat;
	////by the layout in vbo
	//fxClothConst.OffstPos = 0;
	//fxClothConst.OffstNm = 5;
	//fxClothConst.OffstCol = 8;
	//fxClothConst.OffstVel = 11;
}

//creat cuda registered VBO
void testClothRender::initCloth(const unsigned int numVertsWidth, const unsigned int numVertsHeight,
	GLuint attribLoc, ClothConstant& clthConst, FixedClothConstant& fxConst) {

	cloth_width = numVertsWidth;
	cloth_height = numVertsHeight;
	initVBO(attribLoc);
	
	initClothConstValue(clthConst, fxConst);
	
	copyFixClothConst(&fxConst);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&CudaVboRes1, cudaVBO1, cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&CudaVboRes2, cudaVBO2, cudaGraphicsMapFlagsWriteDiscard));
}

void testClothRender::CudaUpdateCloth(ClothConstant in_clothConst) {
	
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

	//ping pong
	Cloth_Launch_Kernel(!pp ? d_testOutPtr1: d_testOutPtr2, !pp? d_testOutPtr2 : d_testOutPtr1,
		cloth_width, cloth_height, VBOStrideInFloat);
	
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

