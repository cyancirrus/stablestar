#include <iostream>
#include "cart_pendulum.h"
// #include "pendulum.h"
#include <btBulletDynamicsCommon.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"

#include <random>
#include <cmath>
#include <string>


void text_draw(const char* text, float x, float y, float scale = 0.01f) {
    char buffer[1024]; // enough for ~1000 chars
    int num_quads;

    num_quads = stb_easy_font_print(x , y , (char*)text, NULL, buffer, sizeof(buffer));

    glPushMatrix();
    glScalef(scale, scale, scale);
    glColor3f(1.0f, 1.0f, 1.0f);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 16, buffer);
    glDrawArrays(GL_QUADS, 0, num_quads * 4);
    glDisableClientState(GL_VERTEX_ARRAY);
    glPopMatrix();
}


void axis_draw(int steps, float size, float x0, float y0, float x1, float y1, float x2, float y2) {
	// x0,y0 :: bottom left point
	// x1, y1 :: top left point
	// x2, y2 :: bottom right point
	float x_del = (x2-x0) / steps;
	float y_del = (y1-y0) / steps;
	
	float y = y0;
	while (y < y1) {
		glBegin(GL_LINE_LOOP);
			glVertex2f(x0 - size, y);
			glVertex2f(x0 + size, y);
		glEnd();
		text_draw(std::to_string(y).c_str(), x0 - size, y - size);
		y += y_del;
	}
	float x = x0;
	while (x < x2) {
		glBegin(GL_LINE_LOOP);
			glVertex2f(x, y1 - size);
			glVertex2f(x, y1 + size);
		glEnd();
		text_draw(std::to_string(x).c_str(), x - size, y - size);
		x += x_del;
	}
}



int main() {
	const float SCALE = 0.25;
	const float OFFSET = 0.10; // initialize GLFW;
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


    glfwMakeContextCurrent(window);
	PendulumCart p(4.0, 2.0, 3.0);


	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);
		glColor3f(1.0f, 1.0f, 0.8f);
		axis_draw(10, 0.05f, -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f);
		// Use with PendulumCart p(1.0, 1.0, 1.0);
		// p.control(0.01f, 50.0f, 10.0f);
		//
		// Use with PendulumCart p(4.0, 2.0, 3.0);
		p.control(0.01f, 100.0f, 10.0f);

		auto [cart, pendulum_x, pendulum_y] = p.position();
		cart *= SCALE;
		pendulum_x *= SCALE;
		pendulum_y *= SCALE;
		std::cout << "cart " << cart << " pendulum_x: " << pendulum_x << " pendulum_y " << pendulum_y << "\n";

		glColor3f(0.8, 1, 0.8);
		glLineWidth(10.0f);

		// Box;
		glBegin(GL_QUADS);
			glVertex2f(cart + OFFSET, + OFFSET);
			glVertex2f(cart - OFFSET, + OFFSET);
			glVertex2f(cart - OFFSET, - OFFSET);
			glVertex2f(cart + OFFSET, - OFFSET);
		glEnd();
		// Pendulum
		glBegin(GL_LINE_LOOP);
			glVertex2f(cart, 0.0f);
			glVertex2f(pendulum_x, pendulum_y);
		glEnd();
		glColor3f(1, 0, 1);
		
		// Axies
		glLineWidth(5.0f);
		// y-axis
		glBegin(GL_LINE_LOOP);
			glVertex2f(-0.95f, -0.95f);
			glVertex2f(-0.95f, 0.95f);
		glEnd();
		// x-axis
		glBegin(GL_LINE_LOOP);
			glVertex2f(0.95f, -0.95f);
			glVertex2f(-0.95f, -0.95f);
		glEnd();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	return 0;
}
