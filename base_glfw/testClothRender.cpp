#include "testClothRender.h"
#include "CudaInte.cuh"


testClothRender::testClothRender()
{
	cloth_width = 0;
	cloth_height = 0;

	cudaVAO = -1;
	cudaVBO = -1;
	CudaVboRes = nullptr;
	VBOStrideInFLoat = 0;

	indexBuffSize = 0;
}


void testClothRender::initVBO(GLuint AttribLocation) {
	//fill VBO
	struct testVert {
		glm::vec3 pos;
		glm::vec2 texCrd;
		glm::vec3 normal;
		glm::vec3 col;
	};
	std::vector<testVert> testGrid;
	for (int i = 0; i < cloth_width; i++) {
		for (int j = 0; j < cloth_height; j++) {
			testGrid.push_back({glm::vec3((float)i/ 100.0f, 0.0f, (float)j/100.0f ), //pos
								glm::vec2((float)i / (float)(cloth_width - 1), (float)j / (float)(cloth_height - 1)),//texCoord
								glm::vec3(1.0f, 0.0f, 1.0f),//megenta
								glm::vec3(1.0f, 1.0f, 0.0f) });	//point color	
		}
	}

	VBOStrideInFLoat = sizeof(testVert) / sizeof(float);

	//gen VAO
	glGenVertexArrays(1, &cudaVAO);
	glBindVertexArray(cudaVAO);

	// create buffer object
	glGenBuffers(1, &cudaVBO);
	glBindBuffer(GL_ARRAY_BUFFER, cudaVBO);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, testGrid.size() * sizeof(testVert), testGrid.data(), GL_DYNAMIC_DRAW);

	// register this buffer object with CUDA
	glEnableVertexAttribArray(AttribLocation);//layout location = attribLoc in vs
	glVertexAttribPointer(AttribLocation, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), 0);
	glEnableVertexAttribArray(AttribLocation + 1);//layout location = attribLoc in vs
	glVertexAttribPointer(AttribLocation + 1, 2, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, texCrd));
	glEnableVertexAttribArray(AttribLocation + 2);//layout location = attribLoc in vs
	glVertexAttribPointer(AttribLocation + 2, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, normal));
	glEnableVertexAttribArray(AttribLocation + 3);//layout location = attribLoc in vs
	glVertexAttribPointer(AttribLocation + 3, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, col));


	//fill IBO
	std::vector<unsigned int> testInd;
	for (int i = 0; i < cloth_width - 1; i++)
	{
		for (int j = 0; j < cloth_height; j++)
		{
			testInd.push_back(i * cloth_height + j);
			testInd.push_back(i * cloth_height + j + cloth_height);
		}
		testInd.push_back(RestartInd);
	}
	//IBO size
	indexBuffSize = testInd.size();

	unsigned int AssignIBO;
	glGenBuffers(1, &AssignIBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, AssignIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, testInd.size() * sizeof(unsigned int), testInd.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void testClothRender::initClothConstValue(ClothConstant& clothConst, FixedClothConstant& fxClothConst) {
	clothConst.m = 0.1f;
	clothConst.G = -9.8f;
	clothConst.k = 0.01f;
	clothConst.rLen = 0.02f;
	clothConst.Fw = glm::vec3(0.0f);
	clothConst.stp = 0.001f;
	clothConst.dt = 0.000005f;
	clothConst.time = 0.0f;
	clothConst.in_testFloat = 0.654f;

	fxClothConst.width = cloth_width;
	fxClothConst.height = cloth_height;
	fxClothConst.vboStrdFlt = VBOStrideInFLoat;
	//by the layout in vbo
	fxClothConst.OffstPos = 0; 
	fxClothConst.OffstNm = 5;
	fxClothConst.OffstCol = 8;
}



//creat cuda registered VBO
void testClothRender::initCloth(const unsigned int numVertsWidth, const unsigned int numVertsHeight,
	GLuint attribLoc, ClothConstant& clthConst, FixedClothConstant& fxConst) {

	cloth_width = numVertsWidth;
	cloth_height = numVertsHeight;
	initVBO(attribLoc);
	
	initClothConstValue(clthConst, fxConst);
	
	copyFixClothConst(&fxConst);


	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&CudaVboRes, cudaVBO, cudaGraphicsMapFlagsWriteDiscard));

}




void testClothRender::CudaUpdateCloth(ClothConstant in_clothConst) {
	
	updateClothConst(&in_clothConst);
	
	//std::cout << "value float = " << valuePass.in_testFloat << std::endl;

	float* d_testOutPtr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &CudaVboRes, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_testOutPtr, &num_bytes, CudaVboRes));

	//call launchKernel here
	Cloth_Launch_Kernel(d_testOutPtr, cloth_width, cloth_height, VBOStrideInFLoat);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &CudaVboRes, 0));
}

void testClothRender::DrawCloth() {
	//element draw
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(RestartInd);
	glBindVertexArray(cudaVAO);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawElements(GL_TRIANGLE_STRIP, indexBuffSize, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindVertexArray(0);
}

