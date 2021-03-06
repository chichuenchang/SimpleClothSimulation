#version 430 core
//[uniform]==================================================
layout(location = 1) uniform float uTime;
layout(location = 3) uniform int ColMode;

//[in varying]==============================================
in vec3 vsOut_pos;
in vec2 vsOut_uv;
smooth in vec3 vsOut_normal;
in vec3 vsOut_pcol;

//[out]=========================================================
out vec4 outCol;	// Final pixel color

//[local]======================================================
vec3 d_Lght = vec3 (1.0f, 1.0f, 1.0f);
//vec3 col_Lght = vec3 (200.0f/300.0f, 180.0f/300.0f, 120.0f/300.0f);
vec3 col_Lght = vec3 (0.961f, 0.961f, 0.863f);

void main() {
	
	outCol = vec4(vsOut_pos.x * 0.2f + 0.5f + 0.5f* cos(1.3f*uTime), vsOut_pos.y * 0.3f + 0.5 * sin(1.7f* uTime), vsOut_pos.z * 0.3f +0.9f - sin(2.3f*uTime), 1.0f) * 0.5f + vec4(0.5f);
	outCol = vec4(vsOut_pos, 1.0f) ;

	outCol = vec4(vsOut_uv.x, 0.0f, vsOut_uv.y, 1.0f);

	vec3 diffCol = dot(vsOut_normal, d_Lght) * col_Lght;


	outCol = vec4(vsOut_normal, 1.0f);
	if (ColMode == 0) { outCol = vec4(vsOut_pcol, 1.0f);}
	else if (ColMode == 1) { outCol = vec4(vsOut_normal, 1.0f); }
	else if (ColMode == 2) { outCol = vec4(vsOut_uv.x, 0.0f, vsOut_uv.y, 1.0f); }
	else if (ColMode == 3) { outCol = vec4( 0.4f*vsOut_pcol + 0.5f*diffCol, 1.0f); }
	//since no color channel in vbo of imported models, use value defined in local
	//else if (ColMode == 3) { outCol = vec4( 0.4f*vec3(0.961f, 0.7f, 0.8f) + 0.5f*diffCol, 1.0f); }

}