#version 430 core
//[uniform]
uniform sampler2D texUnit;
uniform vec3 mousPck;
uniform float uTime;

//[varying]
in vec3 vsOut_pos;
in vec2 vsOut_uv;
in vec3 vsOut_normal;
in vec3 vsOut_pcol;

//[out]
out vec4 outCol;	// Final pixel color


//0.5f * sin(uTime)
void main() {
	//outCol = vec4(vsOut_pos/64.0f, 1.0f) * 0.5f + vec4(0.7f + 0.3f*sin(uTime));
	//outCol = vec4(vsOut_uv, 1.0f, 1.0f) * 0.5f + vec4(0.7f + 0.3f*sin(uTime));
	//outCol = vec4(vsOut_uv.x * 0.2f + 0.5f + 0.5f* cos(1.3f*uTime), vsOut_uv.y * 0.3f + 0.5 * sin(1.7f* uTime), 0.9f - sin(2.3f*uTime), 1.0f) * 0.5f + vec4(0.5f);
	outCol = vec4(vsOut_pos.x * 0.2f + 0.5f + 0.5f* cos(1.3f*uTime), vsOut_pos.y * 0.3f + 0.5 * sin(1.7f* uTime), vsOut_pos.z * 0.3f +0.9f - sin(2.3f*uTime), 1.0f) * 0.5f + vec4(0.5f);
	outCol = vec4(vsOut_pos, 1.0f) ;

	outCol = vec4(vsOut_uv.x, 0.0f, vsOut_uv.y, 1.0f);

	outCol = vec4(vsOut_normal, 1.0f);
	outCol = vec4(vsOut_pcol, 1.0f);
}