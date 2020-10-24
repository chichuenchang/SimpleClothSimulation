#include "util.hpp"
using namespace std;

float GetDeltaT(float &curT, float &lasT) {
	curT = glfwGetTime();
	float dt = curT - lasT;
	lasT = curT;
	return dt;
}

void ReloadShader(GLuint &shaderID) {

	std::vector<GLuint> shaders;
	GLuint tmp_shader_id = compileShader(GL_VERTEX_SHADER, "source/sh_v.glsl");
	shaders.push_back(tmp_shader_id);
	shaders.push_back(compileShader(GL_FRAGMENT_SHADER, "source/sh_f.glsl"));

	shaderID = linkProgram(shaders);
}

///////////////////////////////////////////////////////////////////
// Display info about the OpenGL implementation provided by the graphics driver.
void printGlInfo()
{
	std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

}

//init OpenGL
void initOpenGL(GLFWwindow* glfwWin) {
	assert(glfwWin);
	if (gladLoadGLLoader((GLADloadproc)(glfwGetProcAddress)) == 0) {
		std::cerr << "Failed to intialize OpenGL loader" << std::endl;
		std::exit(1);
	}
	glEnable(GL_DEPTH_TEST);
	assert(glGetError() == GL_NO_ERROR);
	printGlInfo();
}


/////////////////////////////////////////////////////////////////////////////
//ImGui 
void initGui(GLFWwindow* windowPtr) {
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(windowPtr, true);
	const char* GLSL_VERSION = "#version 330";
	ImGui_ImplOpenGL3_Init(GLSL_VERSION);
}



//////////////////////////////////////////////////////////////////////
//shader
GLuint compileShader(GLenum type, string filename, string prepend) {
	// Read the file
	ifstream file(filename);
	stringstream ss;
	string line;

	if (!file.is_open()) {
		ss << "Could not open " << filename << "!" << endl;
		throw runtime_error(ss.str());
	}
	stringstream buffer;
	buffer << prepend << endl;
	buffer << file.rdbuf();
	string bufStr = buffer.str();
	const char* bufCStr = bufStr.c_str();
	GLint length = bufStr.length();

	// Compile the shader
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &bufCStr, &length);
	glCompileShader(shader);
	// Make sure compilation succeeded
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		// Compilation failed, get the info log
		GLint logLength;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
		vector<GLchar> logText(logLength);
		glGetShaderInfoLog(shader, logLength, NULL, logText.data());

		// Construct an error message with the compile log
		stringstream ss;
		string typeStr = "";
		switch (type) {
		case GL_VERTEX_SHADER:
			typeStr = "vertex"; break;
		case GL_FRAGMENT_SHADER:
			typeStr = "fragment"; break;
		}
		ss << "Error compiling " + typeStr + " shader!" << endl << endl << logText.data() << endl;

		// Cleanup shader and throw an exception
		glDeleteShader(shader);
		printf(ss.str().c_str());
		throw runtime_error(ss.str());
	}

	return shader;
}

GLuint linkProgram(vector<GLuint> shaders) {
	GLuint program = glCreateProgram();

	// Attach the shaders and link the program
	for (auto it = shaders.begin(); it != shaders.end(); ++it)
		glAttachShader(program, *it);
	glLinkProgram(program);

	// Detach shaders
	for (auto it = shaders.begin(); it != shaders.end(); ++it)
		glDetachShader(program, *it);

	// Make sure link succeeded
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		// Link failed, get the info log
		GLint logLength;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
		vector<GLchar> logText(logLength);
		glGetProgramInfoLog(program, logLength, NULL, logText.data());

		// Construct an error message with the compile log
		stringstream ss;
		ss << "Error linking program!" << endl << endl << logText.data() << endl;

		// Cleanup program and throw an exception
		glDeleteProgram(program);
		throw runtime_error(ss.str());
	}

	return program;
}