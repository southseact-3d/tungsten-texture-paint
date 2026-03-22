#version 330

in vec3 in_position;
in int in_face_id;

uniform mat4 mvp;

flat out int v_face_id;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_face_id = in_face_id;
}
