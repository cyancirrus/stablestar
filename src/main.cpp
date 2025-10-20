#include <iostream>
#include "pendulum.h"
#include <vector>
#include <btBulletDynamicsCommon.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <random>
#include <cmath>
using std::vector;

int main() {
    // initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "StableStar", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
	glClearColor(0, 0, 0, 1);
	InvPendulum p;

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);

		p.step(0.01f);
		p.control(0.01f, 20.0f, 6.0f);

		auto [x, y] = p.position();

		glColor3f(1, 0, 1);
		glLineWidth(10.0f);
		glBegin(GL_LINES);
			glVertex2f(0.0f, 0.0f); // pivot
			glVertex2f(x, y);       // bob
		glEnd();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}
