#pragma once

#include <vector>
#include <string>

#include <Importer.hpp>
#include <scene.h>
#include <postprocess.h>

#include "CustomObj.h"
#include "impTexture.h"

class impModel
{
public:
	impModel();

	void LoadModel(const std::string& fileName);
	void RenderModel();
	void ClearModel();

	~impModel();

private:

	void LoadNode(aiNode* node, const aiScene* scene);
	void LoadMesh(aiMesh* mesh, const aiScene* scene);
	void LoadMaterials(const aiScene* scene);

	std::vector<CustomObj*> meshList;
	std::vector<impTexture*> textureList;
	std::vector<unsigned int> meshToTex;

};

