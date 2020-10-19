#include "clothRender.h"
#include "CudaInte.cuh"

clothRender::clothRender()
{
	test_width = 0;
	test_height = 0;

	test_cudaVAO = -1;
	test_CudaVboRes = nullptr;
	_assignVAOID = -1;
}

//creat cuda registered VBO
void clothRender::initCloth( const unsigned int numVertsWidth, const unsigned int numVertsHeight,
								GLuint attribLoc) {
	
	test_width = numVertsWidth;
	test_height = numVertsHeight;

	//allocate w*h space for bufferdata()
	size_t emptyAllocSize = sizeof(glm::vec3) * test_width * test_height;


	glm::vec3* emptyAllocDataPtr = (glm::vec3*)malloc(emptyAllocSize);

	/////////////////////////////////////////
	//test
	{

		struct gridVert {
			glm::vec3 pos;
			glm::vec2 texCrd;
		};
		std::vector<gridVert> grid;

		for (int i = 0; i < test_width ; i++) {
			for (int j = 0; j < test_height ; j++) {
				//each vertex has a vec3 pos and a vec2 uv
				grid.push_back({ glm::vec3(0.0f), glm::vec2(j / (test_height - 1), i / (test_width - 1)) });
			}
		}





	}


	///////////////////////////////////////////////////

	//gen VAO
	glGenVertexArrays(1, &test_cudaVAO);
	glBindVertexArray(test_cudaVAO);

	// create buffer object
	GLuint cudaVBO;
	glGenBuffers(1, &cudaVBO);
	glBindBuffer(GL_ARRAY_BUFFER, cudaVBO);
	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, emptyAllocSize, emptyAllocDataPtr, GL_DYNAMIC_DRAW);

	// register this buffer object with CUDA
	glEnableVertexAttribArray(attribLoc);//layout location = attribLoc in vs
	glVertexAttribPointer(attribLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&test_CudaVboRes, cudaVBO, cudaGraphicsMapFlagsWriteDiscard));
}

void clothRender::CudaUpdateCloth(float in_time) {

	glm::vec3* cudaOutputPtr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &test_CudaVboRes, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&cudaOutputPtr, &num_bytes, test_CudaVboRes));
	
	//call launchKernel here
	//launch_kernel(cudaOutputPtr, test_width, test_height, time);
	launch_kernel(cudaOutputPtr, test_width, test_height, in_time);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &test_CudaVboRes, 0));
}

void clothRender::DrawCloth() {

	glBindVertexArray(test_cudaVAO);

	glDrawArrays(GL_POINTS, 0, test_width * test_height);
	
	glBindVertexArray(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
//Old method================================================
void clothRender::initVAO(float offset, GLuint shaderAttribLoc) {
	glEnable(GL_DEPTH_TEST);

	//struct to hold vertices data
	struct gridVert {
		glm::vec3 pos;
		glm::vec2 texCrd;
	};
	std::vector<gridVert> grid;

	for (int i = -numGrid / 2; i < numGrid / 2; i++) {
		for (int j = -numGrid / 2; j < numGrid / 2; j++) {
			grid.push_back({ glm::vec3(-j, 0.0f + offset, -i), glm::vec2((j + numGrid / 2.0f) / numGrid, (i + numGrid / 2.0f) / numGrid) });
		}
	}

	glGenVertexArrays(1, &_assignVAOID);
	glBindVertexArray(_assignVAOID);

	GLuint assignVBO;
	glGenBuffers(1, &assignVBO);
	glBindBuffer(GL_ARRAY_BUFFER, assignVBO);
	glBufferData(GL_ARRAY_BUFFER, grid.size() * sizeof(grid[0]), grid.data(), GL_STATIC_DRAW);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnableVertexAttribArray(shaderAttribLoc);
	glVertexAttribPointer(shaderAttribLoc, 3, GL_FLOAT, GL_FALSE, sizeof(gridVert), 0);
	glEnableVertexAttribArray(shaderAttribLoc + 1);//put the index 2 here as texCoord
	glVertexAttribPointer(shaderAttribLoc + 1, 2, GL_FLOAT, GL_FALSE, sizeof(gridVert), (GLvoid*)offsetof(gridVert, texCrd));

	std::vector<unsigned int> indices;

	for (int i = 0; i < numGrid - 1; i++)
	{
		for (int j = 0; j < numGrid; j++)
		{
			indices.push_back(i * numGrid + j);
			indices.push_back(i * numGrid + j + numGrid);
		}
		indices.push_back(99999);
	}

	unsigned int AssignIBO;
	glGenBuffers(1, &AssignIBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, AssignIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	//unbind in the last
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void clothRender::drawVAO(GLuint VAO_ID) {
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(99999);
	glBindVertexArray(VAO_ID);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawElements(GL_TRIANGLE_STRIP, (numGrid - 1) * 2 * numGrid + (numGrid - 1), GL_UNSIGNED_INT, 0);
	//glDrawElements(GL_TRIANGLE_STRIP, (numGrid - 1) * 2, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindVertexArray(0);
}

