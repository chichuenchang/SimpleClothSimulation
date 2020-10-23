#pragma once
#include "util.hpp"
//#include "Stdfx.h"

class testClothRender
{
public:
	testClothRender();

	void initCloth(const unsigned int numVertsWidth,
		const unsigned int numVertsHeight, GLuint attribLoc,
		ClothConstant& clthConst, FixedClothConstant& fxConst);
	void CudaUpdateCloth(ClothConstant clothConst);
	void DrawCloth();
	void ResetVBO();

private:
	unsigned int cloth_width;
	unsigned int cloth_height;


	struct testVert {
		glm::vec3 pos;
		glm::vec2 texCrd;
		glm::vec3 normal;
		glm::vec3 col;
		glm::vec3 vel;
	};
	std::vector<testVert> testGrid;
	std::vector<testVert> testGrid2;

	GLuint cudaVAO1;
	GLuint cudaVAO2;
	GLuint cudaVBO1;
	GLuint cudaVBO2;
	cudaGraphicsResource* CudaVboRes1;
	cudaGraphicsResource* CudaVboRes2;
	unsigned int VBOStrideInFloat;
	//signed int sizeOfVerts;
	
	//ibo
	unsigned int indexBuffSize;
	const int RestartInd = 9999999;

	void initVBO(GLuint AttribLocation);
	void initClothConstValue(ClothConstant& clothConst, FixedClothConstant& fxClothConst);

	bool pp;
};

