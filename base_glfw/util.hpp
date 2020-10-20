#ifndef _COMMON_HPP
#define _COMMON_HPP

#include "Stdfx.h"

//init opengl
void initOpenGL(GLFWwindow* glfwWin);

//imGUI
void initGui(GLFWwindow* windowPtr);
void drawGui(GLfloat* clearCol, bool show_demo);

//shader
void ReloadShader(GLuint &shaderID);
GLuint compileShader(GLenum type, std::string filename, std::string prepend = "");
GLuint linkProgram(std::vector<GLuint> shaders);


#endif