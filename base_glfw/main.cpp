#include "clothRender.h"
#include "testClothRender.h"

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
bool show_demo_window = false;
// camera
bool camRot = false;
glm::vec3 camCoords = glm::vec3(0.0, 0.0, 1.0);
glm::vec2 camOrigin;
glm::vec2 mouseOrigin;
//time
float time;
//shader ID
GLuint shaderProgram;

//cloth object
//ClothRender cloth;
testClothRender test;
const unsigned int clothWidth = 32;
const unsigned int clothHeight = 64;

///////////////////////////////////////////////////////////////////////////////
// GLFW window callbacks--------------------------------------------------------------------
void scrollCallback(GLFWwindow* w, double x, double y);
void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mode);
void mouseButtonCallback(GLFWwindow* w, int button, int action, int mode);
void cursorPosCallback(GLFWwindow* w, double xp, double yp);
void framebufferSizeCallback(GLFWwindow* w, int width, int height);
void initGLFW(GLFWwindow** win, int winWidth, int winHeight);
/// /////////////////////////////////////////////////////

void ComputeTransform(glm::mat4 &returnTransform) {

	float aspect = (float)width / (float)height;
	glm::mat4 proj = glm::perspective(45.0f, aspect, 0.1f, 100.0f);
	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 30.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 trans = glm::translate(glm::mat4(1.0f), { 0.0f, 0.0f, -camCoords.z });
	glm::mat4 rot = glm::rotate(glm::mat4(1.0f), glm::radians(camCoords.y), { 1.0f, 0.0f, 0.0f });
	rot = glm::rotate(rot, glm::radians(camCoords.x), { 0.0f, 1.0f, 0.0f });
	glm::mat4 scaler = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));
	returnTransform = proj * view * trans * rot * scaler;
}

int main() {
	//initialize
	initGLFW(&window, width, height);
	initOpenGL(window);
	initGui(window);
	//load shader
	ReloadShader(shaderProgram);
	//uniform location
	GLuint xform_uloc = 0;
	GLuint time_uloc = 1;

	//init mesh
	//cloth.initCloth(clothWidth, clothHeight, 8);
	test.initCloth(clothWidth, clothHeight, 8);

	//display
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(clear_color[0], clear_color[1], clear_color[2], 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//bind shader program
		glUseProgram(shaderProgram);
		////transform
		glm::mat4 transform;
		ComputeTransform(transform);
		//pass the uniform  
		glUniformMatrix4fv(xform_uloc, 1, GL_FALSE, glm::value_ptr(transform));
		time = glfwGetTime();
		glUniform1f(time_uloc, time);

		//compute with kernel function
		//cloth.CudaUpdateCloth(time);
		test.CudaUpdateCloth(time);
		assert(glGetError() == GL_NO_ERROR);

		//draw call
		//cloth.DrawCloth();
		test.DrawCloth();
		assert(glGetError() == GL_NO_ERROR);

		//draw imGui
		drawGui(clear_color, show_demo_window);
		//unbind shader
		glUseProgram(0);
		
		glfwSwapBuffers(window);
	}

	//gui clean up
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	//glfw clean up
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}



/// ///////////////////////////////////////////////////////////////////////////////
//GLFW definition
void scrollCallback(GLFWwindow* w, double x, double y) {
	float offset = (y > 0) ? 0.1f : -0.1f;
	camCoords.z = glm::clamp(camCoords.z + offset, -20.0f, 20.0f);
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