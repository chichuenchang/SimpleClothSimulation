#pragma once
#include "util.hpp"

class BBox {

public:
	BBox();
	void initBBox(glm::vec3 a, glm::vec3 b);
	void drawBBox();

	void updateBBoxKenel();


private:

	glm::vec3 pMin, pMax;

	glm::vec3 bbDim;

	GLuint BBVao;
	GLuint BBVbo;
	GLuint BBIbo;
	GLuint AttribLoc;
	GLuint strideBBVbo;


	void checkInputValidity(glm::vec3 a, glm::vec3 b);
	void fillVBOData();
};