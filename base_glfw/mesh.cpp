#include "mesh.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;
using namespace glm;



// Helper functions
int indexOfNumberLetter(string& str, int offset);
int lastIndexOfNumberLetter(string& str);
vector<string> split(const string &s, char delim);

// Constructor - load mesh from file
Mesh::Mesh(string filename) {
	minBB = vec3(numeric_limits<float>::max());
	maxBB = vec3(numeric_limits<float>::lowest());

	vao = 0;
	vbuf = 0;
	vcount = 0;
	load(filename);
}

// Draw the mesh
void Mesh::draw() {
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, vcount);
	glBindVertexArray(NULL);
}

// Load a wavefront OBJ file
void Mesh::load(string filename) {
	// Release resources
	release();

	ifstream file(filename);
	if (!file.is_open()) {
		stringstream ss;
		ss << "Mesh::load() - Could not open file " << filename;
		throw runtime_error(ss.str());
	}

	// Store vertex and normal data while reading
	vector<vec3> raw_vertices;
	vector<vec3> raw_normals;
	vector<unsigned int> v_elements;
	vector<unsigned int> n_elements;

	string line;
	while (getline(file, line)) {
		if (line.substr(0, 2) == "v ") {
			// Read position data
			int index1 = indexOfNumberLetter(line, 2);
			int index2 = lastIndexOfNumberLetter(line);
			vector<string> values = split(line.substr(index1, index2 - index1 + 1), ' ');
			vec3 vert(stof(values[0]), stof(values[1]), stof(values[2]));
			raw_vertices.push_back(vert);

			// Update bounding box
			minBB = glm::min(minBB, vert);
			maxBB = glm::max(maxBB, vert);
		} else if (line.substr(0, 3) == "vn ") {
			// Read normal data
			int index1 = indexOfNumberLetter(line, 2);
			int index2 = lastIndexOfNumberLetter(line);
			vector<string> values = split(line.substr(index1, index2 - index1 + 1), ' ');
			raw_normals.push_back(vec3(stof(values[0]), stof(values[1]), stof(values[2])));

		} else if (line.substr(0, 2) == "f ") {
			// Read face data
			int index1 = indexOfNumberLetter(line, 2);
			int index2 = lastIndexOfNumberLetter(line);
			vector<string> values = split(line.substr(index1, index2 - index1 + 1), ' ');
			for (unsigned int i = 0; i < values.size() - 2; i++) {
				// Split up vertex indices
				vector<string> v1 = split(values[0], '/');		// Triangle fan for ngons
				vector<string> v2 = split(values[i+1], '/');
				vector<string> v3 = split(values[i+2], '/');

				// Store position indices
				v_elements.push_back(stoul(v1[0]) - 1);
				v_elements.push_back(stoul(v2[0]) - 1);
				v_elements.push_back(stoul(v3[0]) - 1);

				// Check for normals
				if (v1.size() >= 3 && v1[2].length() > 0) {
					n_elements.push_back(stoul(v1[2]) - 1);
					n_elements.push_back(stoul(v2[2]) - 1);
					n_elements.push_back(stoul(v3[2]) - 1);
				}
			}
		}
	}
	file.close();

	// Create vertex array
	vector<Vtx> vertices(v_elements.size());
	for (unsigned int i = 0; i < v_elements.size(); i += 3) {
		// Store positions
		vertices[i+0].pos = raw_vertices[v_elements[i+0]];
		vertices[i+1].pos = raw_vertices[v_elements[i+1]];
		vertices[i+2].pos = raw_vertices[v_elements[i+2]];

		// Check for normals
		if (n_elements.size() > 0) {
			// Store normals
			vertices[i+0].norm = raw_normals[n_elements[i+0]];
			vertices[i+1].norm = raw_normals[n_elements[i+1]];
			vertices[i+2].norm = raw_normals[n_elements[i+2]];
		} else {
			// Calculate normal
			vec3 normal = normalize(cross(vertices[i+1].pos - vertices[i+0].pos,
				vertices[i+2].pos - vertices[i+0].pos));
			vertices[i+0].norm = normal;
			vertices[i+1].norm = normal;
			vertices[i+2].norm = normal;
		}
	}
	vcount = vertices.size();

	// Load vertices into OpenGL
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbuf);
	glBindBuffer(GL_ARRAY_BUFFER, vbuf);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vtx), vertices.data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), NULL);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (GLvoid*)sizeof(vec3));

	glBindVertexArray(NULL);
	glBindBuffer(GL_ARRAY_BUFFER, NULL);
}
// Release resources
void Mesh::release() {
	minBB = vec3(numeric_limits<float>::max());
	maxBB = vec3(numeric_limits<float>::lowest());

	if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
	if (vbuf) { glDeleteBuffers(1, &vbuf); vbuf = 0; }
	vcount = 0;
}

int indexOfNumberLetter(string& str, int offset) {
	for (unsigned int i = offset; i < str.length(); ++i) {
		if ((str[i] >= '0' && str[i] <= '9') || str[i] == '-' || str[i] == '.') return i;
	}
	return str.length();
}
int lastIndexOfNumberLetter(string& str) {
	for (int i = str.length() - 1; i >= 0; --i) {
		if ((str[i] >= '0' && str[i] <= '9') || str[i] == '-' || str[i] == '.') return i;
	}
	return 0;
}
vector<string> split(const string &s, char delim) {
	vector<string> elems;

	stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }

    return elems;
}