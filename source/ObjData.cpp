#include "ObjData.h"

void ObjData::objData(glm::vec3 p, float s) {
	vPtr_cb = nullptr;
	indPtr_cb = nullptr;
	nFlt_cb = 0;
	nInd_cb = 0;

	vPtr_q = nullptr;
	indPtr_q = nullptr;
	nFlt_q = 0;
	nInd_q = 0;


	nFlt_s = 0;
	nInd_s = 0;

	//0.3f, -0.5f, 0.5f
	fillCubeVert(glm::vec3(0.2f, -0.9f, 0.5f), s);
	fillQuadVert(p + glm::vec3(-0.8f, -0.4f, -0.8f), 3.0f);
	fillSphereVert(p + glm::vec3(0.0f, 0.0f, 0.0f), 0.1f);

}


void ObjData::fillCubeVert(glm::vec3 p, float s) {
	//p: starting point; s: scale
	
	
	vPtr_cb = new float[] {
		//   x			y				z		u	v			nx	ny	nz				col
		p.x+ s * 0.0f,	p.y+ s * 0.0f,	p.z+ s * 0.0f,		0.0f, 0.0f,		-0.5f, -0.5f, -0.5f,	0.2f, 0.5f, 0.6f,
		p.x+ s * 0.0f,	p.y+ s * 1.0f,	p.z+ s * 0.0f,		0.0f, 1.0f,		-0.5f, 0.5f, -0.5f,		0.2f, 0.5f, 0.6f,
		p.x+ s * 1.0f,	p.y+ s * 1.0f,	p.z+ s * 0.0f,		0.33f, 1.0f,	0.5f, 0.5f, -0.5f,		0.2f, 0.5f, 0.6f,
		p.x+ s * 1.0f,	p.y+ s * 0.0f,	p.z+ s * 0.0f,		0.33f, 0.0f,	0.5f, -0.5f, -0.5f,		0.2f, 0.5f, 0.6f,
		p.x+ s * 0.0f,	p.y+ s * 0.0f,	p.z+ s * 1.0f,		1.0f, 0.0f,		-0.5f, -0.5f, 0.5f,		0.2f, 0.5f, 0.6f,
		p.x+ s * 0.0f,	p.y+ s * 1.0f,	p.z+ s * 1.0f,		1.0f, 1.0f,		-0.5f, 0.5f, 0.0f,		0.2f, 0.5f, 0.6f,
		p.x+ s * 1.0f,	p.y+ s * 1.0f,	p.z+ s * 1.0f,		0.66f, 1.0f,	0.5f, 0.5f, 0.5f,		0.2f, 0.5f, 0.6f,
		p.x+ s * 1.0f,	p.y+ s * 0.0f,	p.z+ s * 1.0f,		0.66f, 0.0f,	0.5f, -0.5f, 0.5f,		0.2f, 0.5f, 0.6f
	};
	
	indPtr_cb = new unsigned int[] {1, 0, 2, 3, 7, 0, 4, 1, 5, 2, 6, 7, 5, 4 };

	nFlt_cb = 88;
	nInd_cb = 14;
}


void ObjData::fillQuadVert(glm::vec3 p, float s) {

	vPtr_q = new float []  {
		p.x + s * 0.0f,	p.y + s * 0.0f,	p.z + s * 0.0f,		0.0f, 0.0f,		0.0f, 1.0f, 0.0f,	0.2f, 0.7f, 0.8f,
		p.x + s * 1.0f,	p.y + s * 0.0f,	p.z + s * 0.0f,		1.0f, 0.0f,		0.0f, 1.0f, 0.0f,	0.2f, 0.7f, 0.8f,
		p.x + s * 1.0f,	p.y + s * 0.0f,	p.z + s * 1.0f,		1.0f, 1.0f,		0.0f, 1.0f, 0.0f,	0.2f, 0.7f, 0.8f,
		p.x + s * 0.0f,	p.y + s * 0.0f,	p.z + s * 1.0f,		0.0f, 1.0f,		0.0f, 1.0f, 0.0f,	0.2f, 0.7f, 0.8f,
	};
	indPtr_q = new unsigned int[] {1, 0, 2, 3 };

	nFlt_q = 44;
}

void ObjData::fillSphereVert(glm::vec3 p, float s) {

	std::vector<glm::vec3> positions;
	std::vector<glm::vec2> uv;
	std::vector<glm::vec3> normals;

	const unsigned int X_SEGMENTS = 16;
	const unsigned int Y_SEGMENTS = 16;
	const float PI = 3.14159265359;
	for (unsigned int y = 0; y <= Y_SEGMENTS; ++y)
	{
		for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
		{
			float xSegment = (float)x / (float)X_SEGMENTS;
			float ySegment = (float)y / (float)Y_SEGMENTS;
			float xPos = s* std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
			float yPos = s* std::cos(ySegment * PI);
			float zPos = s* std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

			positions.push_back(glm::vec3(p.x + xPos, p.y + yPos, p.z + zPos));
			uv.push_back(glm::vec2(xSegment, ySegment));
			normals.push_back(glm::vec3(xPos, yPos, zPos));
		}
	}

	bool oddRow = false;
	for (int y = 0; y < Y_SEGMENTS; ++y)
	{
		if (!oddRow) // even rows: y == 0, y == 2; and so on
		{
			for (int x = 0; x <= X_SEGMENTS; ++x)
			{
				indPtr_s.push_back(y * (X_SEGMENTS + 1) + x);
				indPtr_s.push_back((y + 1) * (X_SEGMENTS + 1) + x);
			}
		}
		else
		{
			for (int x = X_SEGMENTS; x >= 0; --x)
			{
				indPtr_s.push_back((y + 1) * (X_SEGMENTS + 1) + x);
				indPtr_s.push_back(y * (X_SEGMENTS + 1) + x);
			}
		}
		oddRow = !oddRow;
	}
	nInd_s = indPtr_s.size();
	//std::cout << "cpu sphere number of indices = " << indPtr_s.size() << std::endl;


	//std::vector<float> vPtr_s;
	for (int i = 0; i < positions.size(); ++i)
	{
		vPtr_s.push_back(positions[i].x);
		vPtr_s.push_back(positions[i].y);
		vPtr_s.push_back(positions[i].z);
		if (uv.size() > 0)
		{
			vPtr_s.push_back(uv[i].x);
			vPtr_s.push_back(uv[i].y);
		}
		if (normals.size() > 0)
		{
			vPtr_s.push_back(normals[i].x);
			vPtr_s.push_back(normals[i].y);
			vPtr_s.push_back(normals[i].z);
		}
		//color
		vPtr_s.push_back(0.7);
		vPtr_s.push_back(0.2);
		vPtr_s.push_back(0.4);
	}
	nFlt_s = vPtr_s.size();

}


ObjData::~ObjData() {

	if(vPtr_cb) free(vPtr_cb);
	if(indPtr_cb) free(indPtr_cb);
	if(vPtr_q) free(vPtr_q);
	if(indPtr_q) free(indPtr_q);

}
