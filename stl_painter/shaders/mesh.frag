#version 330

in vec4 v_colour;
in float v_diffuse;

out vec4 out_colour;

void main() {
    out_colour = vec4(v_colour.rgb * v_diffuse, v_colour.a);
}
