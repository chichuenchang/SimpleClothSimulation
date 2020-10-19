#ifndef _COMMON_HPP
#define _COMMON_HPP
//common library
#include <string>
#include <vector>
#include <iostream>
#include <cassert>
//openGL
#include <glad/glad.h>
//GLFW
#include <GLFW/glfw3.h>
//glm


#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
//gui
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
//customized header
#include "mesh.hpp"
//opengl select device
#include <windows.h>
//cuda
#include <cuda_gl_interop.h>
#include <helper_cuda.h>


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