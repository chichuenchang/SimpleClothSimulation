#version 430 core
//[uniform]=========================================
layout (location = 0)uniform mat4 xform;			// Model-to-clip space transform
layout (location = 1)uniform float uTime;

//[attribute]======================================
layout(location = 8) in vec3 pos;		
layout(location = 9) in vec2 uv;		
layout(location = 10) in vec3 normal;		
layout(location = 11) in vec3 pcol;		

//[out vafying]==========================================
out vec3 vsOut_pos;
out vec2 vsOut_uv;
smooth out vec3 vsOut_normal;
out vec3 vsOut_pcol;

//[local]============================================
vec3 disp = vec3(0.0, cos(-10*pos.x + 0.7*uTime), cos(10.0*pos.x + 0.3*uTime));
//vec3 disp = vec3(0.0, cos(pos.x + 0.7*uTime), cos(pos.x + 0.3*uTime));

void main() {
	
	//gl_Position = xform * vec4(pos + 0.0f *disp, 1.0);
	gl_Position = xform * vec4(pos, 1.0);
	//gl_Position = vec4(0.0,1.0,1.0, 1.0);


	//vsOut_normal = vec3(xform * vec4(normal, 0.0f));
	vsOut_normal =vec3( vec4(normal, 0.0f));
	vsOut_pos = pos;
	vsOut_uv = uv;
	vsOut_pcol = pcol;

}