#include "testClothRender.h"
#include "util.hpp"
#include "CudaInte.cuh"

extern "C" {//force opengl run with nvidia card
	_declspec(dllexport) DWORD NvOptimusEnablement = 1;
}

/////////////////////////////////////////////////////////////////////////////
//global variables
//window
GLFWwindow* window = nullptr;
GLfloat clear_color[3] = { 0.05f, 0.1f, 0.1f };
int width = 1600, height = 1600;
//GUI
bool show_demo_window = true;
// camera
bool camRot = false;
bool pan = false;
glm::vec2 panCam = glm::vec2(0.0f);
glm::vec3 camCoords = glm::vec3(0.0, 0.0, 1.0);
glm::vec2 camOrigin;
glm::vec2 mouseOrigin;
//time
float time;

//shader ID
GLuint shaderProgram;

//cloth object
//ClothRender cloth;
testClothRender cloth;
const unsigned int clothWidth = 32;
const unsigned int clothHeight = 64;

//constants passed to cuda
ClothConstant cVar;
FixedClothConstant fxVar;

///////////////////////////////////////////////////////////////////////////////
// GLFW window callbacks--------------------------------------------------------------------
void scrollCallback(GLFWwindow* w, double x, double y);
void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mode);
void mouseButtonCallback(GLFWwindow* w, int button, int action, int mode);
void cursorPosCallback(GLFWwindow* w, double xp, double yp);
void framebufferSizeCallback(GLFWwindow* w, int width, int height);
void initGLFW(GLFWwindow** win, int winWidth, int winHeight);
void drawGui(GLfloat* clearCol, bool show_demo, ClothConstant *clothConst);
/// /////////////////////////////////////////////////////////////////////////

void ComputeTransform(glm::mat4 &returnTransform) {

	float aspect = (float)width / (float)height;
	glm::mat4 proj = glm::perspective(45.0f, aspect, 0.1f, 100.0f);
	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 10.0f), 
		glm::vec3(0.0f, -0.1f , clothHeight *0.1f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 trans = glm::translate(glm::mat4(1.0f), { panCam.x, panCam.y, -camCoords.z });
	glm::mat4 rot = glm::rotate(glm::mat4(1.0f), glm::radians(camCoords.y), { 1.0f, 0.0f, 0.0f });
	rot = glm::rotate(rot, glm::radians(camCoords.x), { 0.0f, 1.0f, 0.0f });
	glm::mat4 scaler = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));
	returnTransform = proj * view * trans * rot * scaler;
}

//void PassVarToCuda() {
//
//	cloth.passVarToCudaConst(testflt);
//
//}

void PassUniform() {
	//uniform location
	GLuint xform_uloc = 0;
	GLuint time_uloc = 1;

	//transformation
	glm::mat4 transform;
	ComputeTransform(transform);
	
	//pass the uniform  
	glUniformMatrix4fv(xform_uloc, 1, GL_FALSE, glm::value_ptr(transform));
	time = glfwGetTime();
	glUniform1f(time_uloc, time);
}

void InitGL() {
	initGLFW(&window, width, height);
	initOpenGL(window);
	initGui(window);
	ReloadShader(shaderProgram);
}

void CleanUpGL() {
	//gui clean up
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	//glfw clean up
	glfwDestroyWindow(window);
	glfwTerminate();
}

int main() {

	InitGL();

	GLuint attribLoc = 8;
	cloth.initCloth(clothWidth, clothHeight, attribLoc, cVar, fxVar);
	
	//delta time
	float lastT = 0.0f, currT =0.0f;

	//display
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(clear_color[0], clear_color[1], clear_color[2], 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(shaderProgram);
	
		PassUniform();
		
		cVar.dt = GetDeltaT(currT, lastT);
		cVar.time = currT;
		//fw from the paper
		cVar.Fw = 0.1f*glm::vec3(glm::sin(cVar.a * currT), glm::cos(cVar.a * 1.7 * currT), glm::sin(7 * cVar.a * 1.7 * currT));

		cloth.CudaUpdateCloth(cVar);
		assert(glGetError() == GL_NO_ERROR);
		cloth.DrawCloth();
		assert(glGetError() == GL_NO_ERROR);

		drawGui(clear_color, show_demo_window, &cVar);

		glUseProgram(0);
		glfwSwapBuffers(window);
	}

	CleanUpGL();
	return 0;
}

/// ///////////////////////////////////////////////////////////////////////////////
//GLFW definition
void scrollCallback(GLFWwindow* w, double x, double y) {
	float offset = (y > 0) ? 0.1f : -0.1f;
	camCoords.z = glm::clamp(camCoords.z + offset, -50.0f, 20.0f);
}

void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mode) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
		glfwSetWindowShouldClose(w, true);
	}

	if (key == GLFW_KEY_R && action == GLFW_RELEASE) {
		ReloadShader(shaderProgram);
	}
}

void mouseButtonCallback(GLFWwindow* w, int button, int action, int mode) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		// Activate rotation mode
		camRot = true;
		camOrigin = glm::vec2(camCoords);
		double xpos, ypos; //has to be double, as is the argument type of glfwGetCursorPos();
		glfwGetCursorPos(w, &xpos, &ypos);
		mouseOrigin = glm::vec2(xpos, ypos);


	} if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
		camRot = false;
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		pan = true;
		camOrigin = glm::vec2(camCoords);
		double xpos, ypos; //has to be double, as is the argument type of glfwGetCursorPos();
		glfwGetCursorPos(w, &xpos, &ypos);
		mouseOrigin = glm::vec2(xpos, ypos);

	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
		pan = false;

	}


}

void cursorPosCallback(GLFWwindow* w, double xp, double yp) {
	if (camRot) {
		float rotScale = std::fmin(width / 450.f, height / 270.f);
		glm::vec2 mouseDelta = glm::vec2(xp, yp) - mouseOrigin;
		glm::vec2 newAngle = camOrigin + mouseDelta / rotScale;
		newAngle.y = glm::clamp(newAngle.y, -90.0f, 90.0f);
		while (newAngle.x > 180.0f) newAngle.x -= 360.0f;
		while (newAngle.y < -180.0f) newAngle.y += 360.0f;
		if (glm::length(newAngle - glm::vec2(camCoords)) > std::numeric_limits<float>::epsilon()) {
			camCoords.x = newAngle.x;
			camCoords.y = newAngle.y;
		}
	}
	if (pan) {

		float rotScale = std::fmin(width / 450.f, height / 270.f);
		glm::vec2 mouseDelta = mouseOrigin - glm::vec2(xp, yp) ;
		glm::vec2 newAngle = camOrigin + mouseDelta / rotScale;
		//newAngle.y = glm::clamp(newAngle.y, -90.0f, 90.0f);
		//while (newAngle.x > 180.0f) newAngle.x -= 360.0f;
		//while (newAngle.y < -180.0f) newAngle.y += 360.0f;
		if (glm::length(newAngle - glm::vec2(camCoords)) > std::numeric_limits<float>::epsilon()) {
			panCam.x = -0.01f * newAngle.x;
			panCam.y = 0.01f * newAngle.y;

		}
	}


}

void framebufferSizeCallback(GLFWwindow* w, int width, int height) {
	::width = width;
	::height = height;
	glViewport(0, 0, width, height);
}

void initGLFW(GLFWwindow** win, int winWidth, int winHeight) {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	*win = glfwCreateWindow(winWidth, winHeight, "ClothSim", nullptr, nullptr);
	if (!*win) {
		std::cerr << "Cannot create window";
		std::exit(1);
	}
	glfwMakeContextCurrent(*win);

	glfwSetKeyCallback(*win, keyCallback);
	glfwSetMouseButtonCallback(*win, mouseButtonCallback);
	glfwSetCursorPosCallback(*win, cursorPosCallback);
	glfwSetFramebufferSizeCallback(*win, framebufferSizeCallback);
	glfwSetScrollCallback(*win, scrollCallback);
}

void drawGui(GLfloat* clearCol, bool show_demo, ClothConstant *clothConst) {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	//show demo window
	if (show_demo) ImGui::ShowDemoWindow(&show_demo);

	static int counter = 0;
	{
		ImGui::Begin("JC Zheng");                          // Create a window called "Hello, world!" and append into it.

		ImGui::ColorEdit3("clear color", clearCol); // Edit 3 floats representing a color

		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		ImGui::SliderFloat("Wind Strength", &clothConst->in_testFloat, 0.0f, 0.5f, "test float = %.3f");
		
		static int clicked = 0;
		if (ImGui::Button("Button")) {

			clicked++;
			cloth.ResetVBO();

		}
		if (clicked & 1)
		{
			ImGui::SameLine();
			ImGui::Text("Reload Cloth");
		}



		ImGui::End();
	}
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}