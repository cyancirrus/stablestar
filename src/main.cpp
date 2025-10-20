#include <vector>
#include <iostream>
#include <btBulletDynamicsCommon.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <random>
#include <cmath>
using std::vector;

static constexpr float g = 9.18f;

struct InvPendulum {
	float theta;
	float theta_d;
	// g/l sin(theta)
	float theta_dd;

	InvPendulum() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(-0.1,0.1);

		theta = dis(gen);
		theta_d= 0.0f;
		theta_dd=0.0f;
	}
	void step(float dt) {
		float l = 1.0f;
		theta_dd = g/l * sin(theta);
		theta_d += dt * theta_dd;
		theta += dt * theta_d;
	}
	void control(float dt, float k_p, float k_d) {
		float l = 1.0f, m = 1.0f;
		theta_dd = g/l * sin(theta)
			- k_d / (m * l * l) * theta_d
			- k_p / (m * l * l) * theta
		;
		theta_d += dt * theta_dd;
		theta += dt * theta_d;
	}
};


// fn ~ c * exp(-wt) + d * exp(wt);
// consering only the unstable solution as inverted
//
// => df/dt = w * d * exp(wt) dt
// = dt * w * f

// void step(InvPendulum &p, float dt) {
// 	float g = 9.18f, l = 1.0f;
// 	p.theta_dd = g/l * sin(p.theta);
// 	p.theta_d += dt * p.theta_dd;
// 	p.theta += dt * p.theta_d;
// }

// void control_step(InvPendulum &p, float k_p, float k_d, float dt) {
// 	// from pde
// 	// d/dt d/dt f + kd/ml^2 d/dt f + f (kp/ml^2 -g/l) = 0;
// 	float g = 9.18f, l = 1.0f, m=1.0f;
// 	// don't linearize the physic but linearize control
// 	p.theta_dd = g/l * sin(p.theta)
// 			- k_d / (m * l * l) * p.theta_d
// 			- k_p/ (m * l * l) * p.theta;

// 	p.theta_d += dt * p.theta_dd;
// 	p.theta += dt * p.theta_d;
// }
int main(void) {

	InvPendulum p;
	for(int i=0; i<30; i++) {
		p.control(0.1f, 12.0f, 2.0f);
		std::cout << p.theta << " " << p.theta_d << "\n";
	}



	// if (!glfwInit()) return -1;
	
	// GLFWwindow* window = glfwCreateWindow(800, 600, "StableStar", NULL, NULL);
	// if (!window) { glfwTerminate(); return -1; }
	
	// glfwMakeContextCurrent(window);
	// if (glewInit() != GLEW_OK) {
	// 	std::cerr << "GLEW INIT FAILED \n";
	// 	return -1;
	// }

    // glMatrixMode(GL_PROJECTION);
    // glOrtho(-1, 1, -1, 1, -1, 1);

	// while (!glfwWindowShouldClose(window)) {
	// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    	// glMatrixMode(GL_PROJECTION);
    	// glLoadIdentity();

	// 	glColor3f(1.0f, 0.0f, 1.0f);
	// 		glBegin(GL_LINES);
	// 		// glVertex2f(-0.5f, -0.5f);
	// 		// glVertex2f(0.5f, 0.5f);
	// 		glVertex2f(-1.0f, -1.0f);
	// 		glVertex2f(1.0f, 1.0f);
	// 	// glEnd();

	// 	glfwSwapBuffers(window);
	// 	glfwPollEvents();
	// }
	// glfwDestroyWindow(window);
	// glfwTerminate();

	std::cout << "new world\n";
	return 0;
}
