#version 330

in vec4 v_colour;
in float v_diffuse;

out vec4 out_colour;

void main() {
    vec3 lit = v_colour.rgb * v_diffuse;
    vec3 rim = vec3(0.08);
    out_colour = vec4(min(lit + rim, vec3(1.0)), v_colour.a);
}
