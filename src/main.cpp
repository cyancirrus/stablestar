#include <iostream>
#include <string>
#include <format>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cart_pendulum.h"
// #include "pendulum.h"
#include <btBulletDynamicsCommon.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"

void text_draw(const char* text, float x, float y, float scale = 0.005f) {
    static char vertex_buffer[1024]; // static so it persists
    int num_quads = stb_easy_font_print(0, 0, (char*)text, NULL, vertex_buffer, sizeof(vertex_buffer)*8);
    
    glPushMatrix();
    glTranslatef(x, y, 0.0f);
    glScalef(scale, -scale, 1.0f); // negative Y to flip text right-side up
    
    glColor3f(1.0f, 1.0f, 1.0f);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 16, vertex_buffer);
    glDrawArrays(GL_QUADS, 0, num_quads * 4);
    glDisableClientState(GL_VERTEX_ARRAY);
    glPopMatrix();
}
void axis_draw(int steps, float size, float x0, float y0, float x1, float y1, float x2, float y2) {
	// x0,y0 :: bottom left point
	// x1, y1 :: top left point
	// x2, y2 :: bottom right point
	float y_del = (y1-y0) / steps;
	float x_del = (x2-x0) / steps;
	// Axies
	glLineWidth(5.0f);
	glColor3f(0.3f, 0.3f, 0.35f);
	// y-axis
	glBegin(GL_LINE_LOOP);
		glVertex2f(-0.9f, -0.9f);
		glVertex2f(-0.9f, 0.9f);
	glEnd();
	// x-axis
	glBegin(GL_LINE_LOOP);
		glVertex2f(0.95f, -0.9f);
		glVertex2f(-0.95f, -0.9f);
	glEnd();

	glColor3f(0.5f, 0.5f, 0.55f);	
	// ticks for y axis are irrelevant for control
	//
	// float y = y0;
	// while (y < y1) {
	// 	glBegin(GL_LINE_LOOP);
	// 		glVertex2f(x0 - size, y);
	// 		glVertex2f(x0 + size, y);
	// 	glEnd();
	// 	// text_draw(std::to_string(y).substr(0, 4).c_str(), x0 , y - size);
	// 	y += y_del;
	// }
	float x = x0;
	while (x < x2) {
		glBegin(GL_LINE_LOOP);
			glVertex2f(x, y0 - size);
			glVertex2f(x, y0 + size);
		glEnd();
		text_draw(std::to_string(x).substr(0, 4).c_str(), x + size, y0 - size);
		x += x_del;
	}
}
void state_label_draw(float theta, float velocity) {
	glColor3f(0.9f, 0.9f, 0.95f);
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "theta: %.2f, velocity: %.2f", theta, velocity);
    text_draw(buffer, 0.15f, 0.75);
}
// void save_frame(int frame_num, int width, int height) {
//     std::vector<unsigned char> pixels(width * height * 3);
//     glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
//     // Flip vertically (OpenGL's origin is bottom-left)
//     std::vector<unsigned char> flipped(width * height * 3);
//     for (int y = 0; y < height; y++) {
//         memcpy(&flipped[y * width * 3], 
//                &pixels[(height - 1 - y) * width * 3], 
//                width * 3);
//     }
    
//     char filename[64];
//     snprintf(filename, sizeof(filename), "frames/frame_%04d.png", frame_num);
//     stbi_write_png(filename, width, height, 3, flipped.data(), width * 3);
// }

void save_frame(int frame_num, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    char filename[64];
    snprintf(filename, sizeof(filename), "frames/frame_%04d.png", frame_num);
    stbi_write_png(filename, width, height, 3, pixels.data(), width * 3);
}

int main() {
	const float SCALE = 0.25f;
	const float OFFSET = 0.10f; // initialize GLFW;
	const float VERTICAL_OFFSET = -0.75f;
	const float ASPECT = 1600.0f/800.0f;
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(1600, 800, "StableStar", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }
	glClearColor(0.1f, 0.12f, 0.15f, 1.0f);


    glfwMakeContextCurrent(window);
	PendulumCart p(3.0f, 1.0f, 1.25f);

	int frame = 0;
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);
		axis_draw(10, 0.05f, -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f);
		p.control(0.01f, 70.0f, 5.0f);

		auto [cart, pendulum_x, pendulum_y] = p.position();
		// std::cout << "theta: " << p.theta << "\n";
		// std::cout << "cart " << cart << " pendulum_x: " << pendulum_x << " pendulum_y " << pendulum_y << "\n";
		cart *= SCALE / ASPECT;
		pendulum_x *= SCALE / ASPECT;
		pendulum_y *= SCALE;
		state_label_draw(p.theta, p.x_dot);

		glLineWidth(10.0f);
		glColor3f(0.2f, 0.7f, 0.8f);
		// Box;
		glBegin(GL_QUADS);
			glVertex2f(cart + OFFSET, + OFFSET + VERTICAL_OFFSET);
			glVertex2f(cart - OFFSET, + OFFSET + VERTICAL_OFFSET);
			glVertex2f(cart - OFFSET, - OFFSET + VERTICAL_OFFSET);
			glVertex2f(cart + OFFSET, - OFFSET + VERTICAL_OFFSET);
		glEnd();
		// Pendulum
		glColor3f(0.9f, 0.6f, 0.2f);
		glBegin(GL_LINE_LOOP);
			glVertex2f(cart, VERTICAL_OFFSET);
			glVertex2f(pendulum_x, pendulum_y + VERTICAL_OFFSET);
		glEnd();
		// capture 600 frames (10 sec at 60fps)
		if (frame < 600) { save_frame(frame, 1600, 800); }
		glfwSwapBuffers(window);
		glfwPollEvents();
		frame++;
	}
	return 0;
}
