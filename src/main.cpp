#include <vector>
#include <iostream>
#include <btBulletDynamicsCommon.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
using std::vector;

struct Pendulum {
	float theta;
	float theta_dot;
};

void step(Pendulum &p, float u, float dt) {
	float g = 9.18f, l = 1.0f;
	float theta_dot = (g / l) * sin(p.theta) + u;
	p.theta += p.theta_dot * dt;
	p.theta_dot += theta_dot * dt;
}



int main(void) {
	if (!glfwInit()) return -1;
	
	GLFWwindow* window = glfwCreateWindow(800, 600, "StableStar", NULL, NULL);
	if (!window) { glfwTerminate(); return -1; }
	
	glfwMakeContextCurrent(window);
	glewInit();

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glfwDestroyWindow(window);
	glfwTerminate();

	std::cout << "hello world\n";
	return 0;
}
