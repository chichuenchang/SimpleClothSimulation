#pragma once
#include "util.hpp"

class ObjData
{
public:

	void objData(glm::vec3 p, float s);
	~ObjData();
	
	float* vPtr_cb;
	unsigned int* indPtr_cb;
	unsigned int nFlt_cb;
	unsigned int nInd_cb;

	float* vPtr_q;
	unsigned int* indPtr_q;
	unsigned int nFlt_q;
	unsigned int nInd_q;

	std::vector<float> vPtr_s;
	std::vector<unsigned int> indPtr_s;
	unsigned int nFlt_s;
	unsigned int nInd_s;

private:

	void fillCubeVert(glm::vec3 p, float s);
	void fillSphereVert(glm::vec3 p, float s);
	void fillQuadVert(glm::vec3 p, float s);


};

