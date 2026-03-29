#version 330

uniform mat4 mvp;
uniform float depth_bias;

in vec3 in_position;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    gl_Position.z -= depth_bias * gl_Position.w;
}
