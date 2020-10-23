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
	CudaVboRes1 = nullptr;
	CudaVboRes2 = nullptr;
	VBOStrideInFloat = 0;

	indexBuffSize = 0;

	pp = false;
}


void testClothRender::initVBO(GLuint AttribLocation) {
	/// <summary>
	/// ///////////////////////////////////////////
	/// BUFFER#1
	struct testVert {
		glm::vec3 pos;
		glm::vec2 texCrd;
		glm::vec3 normal;
		glm::vec3 col;
		glm::vec3 vel;
	};
	std::vector<testVert> testGrid;
	for (int i = 0; i < cloth_width; i++) {
		for (int j = 0; j < cloth_height; j++) {
			testGrid.push_back({glm::vec3((float)i/ 100.0f, 0.0f, (float)j/100.0f ), //pos
								glm::vec2((float)i / (float)(cloth_width - 1), (float)j / (float)(cloth_height - 1)),//texCoord
								glm::vec3(1.0f, 1.0f, 0.0f),//megenta
								glm::vec3(1.0f, 0.0f, 1.0f),//p color megenta
								glm::vec3(0.0f)});	//for kernel to write velocity
								
		}
	}

	std::vector<testVert> testGrid2;
	for (int i = 0; i < cloth_width; i++) {
		for (int j = 0; j < cloth_height; j++) {
			testGrid2.push_back({ glm::vec3((float)i / 100.0f, 0.0f, (float)j / 100.0f), //pos
								glm::vec2((float)i / (float)(cloth_width - 1), (float)j / (float)(cloth_height - 1)),//texCoord
								glm::vec3(0.0f, 1.0f, 1.0f),//
								glm::vec3(0.0f, 1.0f, 1.0f),//point color	
								glm::vec3(0.0f) });	
		}
	}

	VBOStrideInFloat = sizeof(testVert) / sizeof(float);

	//gen VAO
	glGenVertexArrays(1, &cudaVAO1);
	glBindVertexArray(cudaVAO1);

	// create buffer object
	glGenBuffers(1, &cudaVBO1);
	glBindBuffer(GL_ARRAY_BUFFER, cudaVBO1);
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
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	/// <summary>
	/// ///////////////////////////////////////////
	/// BUFFER#2
	/// 
	glGenVertexArrays(1, &cudaVAO2);
	glBindVertexArray(cudaVAO2);
	// create buffer object
	glGenBuffers(1, &cudaVBO2);
	glBindBuffer(GL_ARRAY_BUFFER, cudaVBO2);
	glBufferData(GL_ARRAY_BUFFER, testGrid.size() * sizeof(testVert), testGrid2.data(), GL_DYNAMIC_DRAW);

	// register this buffer object with CUDA
	glEnableVertexAttribArray(AttribLocation);//layout location = attribLoc in vs
	glVertexAttribPointer(AttribLocation, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), 0);
	glEnableVertexAttribArray(AttribLocation + 1);//layout location = attribLoc in vs
	glVertexAttribPointer(AttribLocation + 1, 2, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, texCrd));
	glEnableVertexAttribArray(AttribLocation + 2);//layout location = attribLoc in vs
	glVertexAttribPointer(AttribLocation + 2, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, normal));
	glEnableVertexAttribArray(AttribLocation + 3);//layout location = attribLoc in vs
	glVertexAttribPointer(AttribLocation + 3, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, col));
	
	unsigned int AssignIBO2;
	glGenBuffers(1, &AssignIBO2);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, AssignIBO2);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, testInd.size() * sizeof(unsigned int), testInd.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void testClothRender::initClothConstValue(ClothConstant& clothConst, FixedClothConstant& fxClothConst) {
	clothConst.M = 0.01f;
	clothConst.g = -5.0f;
	clothConst.k = 1.0f;
	clothConst.rLen = 0.02f;
	clothConst.Fw = glm::vec3(0.0f);
	clothConst.a = 0.001f;
	clothConst.stp = 0.008f;
	clothConst.dt = 0.00001f;
	clothConst.time = 0.0f;
	clothConst.MinL = 0.015f;
	clothConst.MaxL = 0.025f;
	clothConst.Dp = 0.002f;
	clothConst.in_testFloat = 0.654f;

	fxClothConst.width = cloth_width;
	fxClothConst.height = cloth_height;
	fxClothConst.vboStrdFlt = VBOStrideInFloat;
	//by the layout in vbo
	fxClothConst.OffstPos = 0; 
	fxClothConst.OffstNm = 5;
	fxClothConst.OffstCol = 8;
	fxClothConst.OffstVel = 11;
	
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

void testClothRender::DrawCloth() {
	//element draw
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(RestartInd);
	//draw the write buffer
	glBindVertexArray(!pp ? cudaVAO2 : cudaVAO1);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawElements(GL_TRIANGLE_STRIP, indexBuffSize, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindVertexArray(0);
}

