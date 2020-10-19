#include "testClothRender.h"
#include "CudaInte.cuh"

testClothRender::testClothRender()
{
	test_width = 0;
	test_height = 0;

	test_cudaVAO = -1;
	test_CudaVboRes = nullptr;

	indexBuffSize = 0;
}



//creat cuda registered VBO
void testClothRender::initCloth(const unsigned int numVertsWidth, const unsigned int numVertsHeight,
	GLuint attribLoc) {

	test_width = numVertsWidth;
	test_height = numVertsHeight;

	struct testVert {
		glm::vec3 pos;
		glm::vec2 texCrd;
	};
	std::vector<testVert> testGrid;

	for (int i = 0; i < test_width; i++) {
		for (int j = 0; j < test_height; j++) {
			//each vertex has a vec3 pos and a vec2 uv
			testGrid.push_back({ glm::vec3(0.0f), glm::vec2(j / (test_height - 1), i / (test_width - 1)) });
		}
	}

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
	glEnableVertexAttribArray(attribLoc+1);//layout location = attribLoc in vs
	glVertexAttribPointer(attribLoc+1, 2, GL_FLOAT, GL_FALSE, sizeof(testVert), (GLvoid*)offsetof(testVert, texCrd));

	//IBO
	std::vector<unsigned int> testInd;
	for (int i = 0; i < test_width - 1; i++)
	{
		for (int j = 0; j < test_height; j++)
		{
			testInd.push_back(i * test_height + j);
			testInd.push_back(i * test_height + j + test_height);
		}
		testInd.push_back(99999);
	}

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

	float* testOutPtr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &test_CudaVboRes, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&testOutPtr, &num_bytes, test_CudaVboRes));

	//call launchKernel here
	test_launch_kernel(testOutPtr, test_width, test_height, in_time);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &test_CudaVboRes, 0));
}

void testClothRender::DrawCloth() {
	
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(99999);
	glBindVertexArray(test_cudaVAO);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawElements(GL_TRIANGLE_STRIP, indexBuffSize, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindVertexArray(0);

}

