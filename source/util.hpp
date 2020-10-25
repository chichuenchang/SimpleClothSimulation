#pragma once
#ifndef _COMMON_HPP
#define _COMMON_HPP

#include "Stdfx.h"

//init opengl
void initOpenGL(GLFWwindow* glfwWin);

//imGUI
void initGui(GLFWwindow* windowPtr);
//void drawGui(GLfloat* clearCol, bool show_demo);

//shader
void ReloadShader(GLuint &shaderID);
GLuint compileShader(GLenum type, std::string filename, std::string prepend = "");
GLuint linkProgram(std::vector<GLuint> shaders);

//delta time
float GetDeltaT(float& curT, float& lasT);

struct ClothConstant {

	float M;
	float g;
	float k;
	float rLen;	//initial length

	glm::vec3 Fw; //wind
	float WStr;
	float a;
	float stp; //time step
	float Dp;
	
	float dt; //delta time each frame
	float time; //system time



	float in_testFloat;

};

struct FixedClothConstant {

	unsigned int width;
	unsigned int height;
	unsigned int vboStrdFlt;
	unsigned int OffstPos;
	unsigned int OffstNm;
	unsigned int OffstCol;
	unsigned int OffstVel;

};


#endif