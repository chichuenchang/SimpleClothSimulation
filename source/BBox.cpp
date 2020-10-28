#include "BBox.h"

BBox::BBox() {

	BBVbo = -1;
	BBVao = -1;
	BBIbo = -1;
	AttribLoc = -1;
	strideBBVbo = -1;

}


void BBox::fillVBOData() {

	glm::vec3 bbVerts[] = {
		pMin,
		pMin + glm::vec3(0.0f, bbDim.y, 0.0f),
		pMin + glm::vec3(0.0f, bbDim.y, 0.0f) + glm::vec3(bbDim.x, 0.0f, 0.0f),
		pMin + glm::vec3(bbDim.x, 0.0f, 0.0f),

		pMin + glm::vec3(0.0f, 0.0f, bbDim.z),
		pMin + glm::vec3(0.0f, 0.0f, bbDim.z) + glm::vec3(0.0f, bbDim.y, 0.0f),
		pMin + glm::vec3(0.0f, 0.0f, bbDim.z) + glm::vec3(0.0f, bbDim.y, 0.0f) + glm::vec3(bbDim.x, 0.0f, 0.0f),
		pMin + glm::vec3(0.0f, 0.0f, bbDim.z) + glm::vec3(bbDim.x, 0.0f, 0.0f)
	};

	

}

void BBox::checkInputValidity(glm::vec3 a, glm::vec3 b) {

	if ((a.x >= b.x) || (a.y >= b.y) || (a.z >= b.z)) {
		std::cout << "input point of BBox is not correct" << std::endl;
		exit(EXIT_FAILURE);
	}

}

void BBox::initBBox(glm::vec3 a , glm::vec3 b) {
	
	checkInputValidity(a, b);
	pMin = a;
	pMax = b;
	bbDim = b - a;

	glGenVertexArrays(1, &BBVao);
	glBindVertexArray(BBVao);

	glGenBuffers(1, &BBVbo);
	glBindBuffer(GL_ARRAY_BUFFER, BBVbo);
	//glBufferData()


}