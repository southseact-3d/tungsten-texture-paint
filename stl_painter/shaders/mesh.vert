#version 330

in vec3 in_position;
in vec3 in_normal;
in vec4 in_colour;

uniform mat4 mvp;
uniform vec3 light_dir;

out vec4 v_colour;
out float v_diffuse;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_diffuse = max(dot(normalize(in_normal), normalize(light_dir)), 0.15);
    v_colour = in_colour;
}
