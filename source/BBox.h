#pragma once
#include "util.hpp"

class BBox {

public:
	BBox();
	void initBBox();
	void drawBBox();

	void updateBBoxKenel();


private:
	glm::vec3 boxDim;

	GLuint BBVao;
	GLuint BBVbo;
	GLuint BBIbo;
	GLuint AttribLoc;
	GLuint strideBBVbo;


};