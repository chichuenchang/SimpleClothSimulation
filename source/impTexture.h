#pragma once

#include "util.hpp"
#include "stb_image.h"

class impTexture
{
public:
	impTexture();
	impTexture(const char* fileLoc);

	bool LoadTexture();
	bool LoadTextureA();

	void UseTexture();
	void ClearTexture();

	~impTexture();

private:
	GLuint textureID;
	int width, height, bitDepth;

	const char* fileLocation;


};

