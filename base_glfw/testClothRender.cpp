#include "testClothRender.h"
#include "CudaInte.cuh"

testClothRender::testClothRender()
{
	cloth_width = 0;
	cloth_height = 0;

	test_cudaVAO = -1;
	test_CudaVboRes = nullptr;
	VBOStrideInFLoat = 0;

	indexBuffSize = 0;
}

//creat cuda registered VBO
void testClothRender::initCloth(const unsigned int numVertsWidth, const unsigned int numVertsHeight,
	GLuint attribLoc) {

	cloth_width = numVertsWidth;
	cloth_height = numVertsHeight;

	//fill VBO
	struct testVert {
		glm::vec3 pos;
		glm::vec2 texCrd;
		glm::vec3 normal;
	};
	std::vector<testVert> testGrid;
	for (int i = 0; i < cloth_width; i++) {
		for (int j = 0; j < cloth_height; j++) {
			testGrid.push_back({glm::vec3(0.0f), //pos
								glm::vec2(j / (cloth_height - 1), i / (cloth_width - 1)),//texCoord
								glm::vec3(0.0f)});//normal
		}
	}

	VBOStrideInFLoat = sizeof(testVert)/sizeof(float);

	//gen VAO
	glGenVertexArrays(1, &test_cudaVAO);
	glBindVertexArray(test_cudaVAO);

	// create buffer object
	GLuint cudaVBO;
	glGenBuffers(1, &cudaVBO);
	glBindBuffer(GL_ARRAY_BUFFER, cudaVBO);
	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, testGrid.size()*sizeof(testVert), testGrid.data(), GL_DYNAMIC_DRAW);

	// register this buffer object with CUDA
	glEnableVertexAttribArray(attribLoc);//layout location = attribLoc in vs
	glVertexAttribPointer(attribLoc, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), 0);
	glEnableVertexAttribArray(attribLoc + 1);//layout location = attribLoc in vs
	glVertexAttribPointer(attribLoc + 1, 2, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, texCrd));
	glEnableVertexAttribArray(attribLoc + 2);//layout location = attribLoc in vs
	glVertexAttribPointer(attribLoc + 2, 3, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, normal));

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

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&test_CudaVboRes, cudaVBO, cudaGraphicsMapFlagsWriteDiscard));
}

void testClothRender::CudaUpdateCloth(float in_time) {

	float* d_testOutPtr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &test_CudaVboRes, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_testOutPtr, &num_bytes, test_CudaVboRes));

	//call launchKernel here
	test_launch_kernel(d_testOutPtr, cloth_width, cloth_height, in_time, VBOStrideInFLoat);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &test_CudaVboRes, 0));
}

void testClothRender::DrawCloth() {
	//element draw
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(RestartInd);
	glBindVertexArray(test_cudaVAO);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawElements(GL_TRIANGLE_STRIP, indexBuffSize, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindVertexArray(0);
}

