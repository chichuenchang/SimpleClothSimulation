#ifndef _STDFX_H
#define _STDFX_H

//common library
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
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
//opengl select device
#include <windows.h>
//cuda
#include <cuda_gl_interop.h>
#include <helper_cuda.h>

#endif
