#version 330

flat in int v_face_id;

out vec4 out_colour;

void main() {
    int face_id = v_face_id;
    int r = (face_id >> 16) & 255;
    int g = (face_id >> 8) & 255;
    int b = face_id & 255;
    out_colour = vec4(r / 255.0, g / 255.0, b / 255.0, 1.0);
}
