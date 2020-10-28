#version 430 core
//[uniform]=========================================
layout(location = 0)uniform mat4 xform;			// Model-to-clip space transform
layout(location = 4)uniform mat4 model;
layout(location = 5)uniform mat4 view;
layout(location = 6)uniform mat4 proj;

layout (location = 1)uniform float uTime;



//[attribute]======================================
layout(location = 0) in vec3 pos;		
layout(location = 1) in vec2 uv;		
layout(location = 2) in vec3 normal;		
layout(location = 3) in vec3 pcol;		

//[out vafying]==========================================
out vec3 vsOut_pos;
out vec2 vsOut_uv;
smooth out vec3 vsOut_normal;
out vec3 vsOut_pcol;

//[local]============================================
vec3 disp = vec3(0.0, cos(-10*pos.x + 0.7*uTime), cos(10.0*pos.x + 0.3*uTime));

void main() {
	


	gl_Position = proj* view* model* vec4(pos, 1.0);


	//vsOut_normal = vec3(xform * vec4(normal, 0.0f));
	vsOut_normal =vec3( vec4(normal, 0.0f));
	vsOut_pos = pos;
	vsOut_uv = uv;
	vsOut_pcol = pcol;

}