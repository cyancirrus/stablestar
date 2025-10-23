#include <iostream>
// #include "pendulum.h"
#include <btBulletDynamicsCommon.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <random>
#include <cmath>

static constexpr float g = 9.18f;

struct State {
	float cart_x;
	float pendulum_x;
	float pendulum_y;
};

struct PendulumCart {
	float mass_cart;
	float mass_pendulum;
	float length_pendulum;
	// angle offset
	float theta;
	float theta_dot;
	// x offset
	float x;
	float x_dot;

	PendulumCart(float m_cart, float m_pendulum, float l_pendulum);
	void step(float dt);
	void control(float dt, float kp, float kd);
	State position(void);
};

PendulumCart::PendulumCart(float m_cart, float m_pendulum, float l_pendulum) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-1.0,1.0);
	mass_cart = m_cart;
	mass_pendulum = m_pendulum;
	length_pendulum = l_pendulum;
	
	theta = dis(gen);
	theta_dot = 0.0f;
	x = 0.0f;
	x_dot = 0.0f;
}

void PendulumCart::step(float dt) {
	theta_dot += dt * (mass_cart + mass_pendulum) * g /(mass_pendulum * length_pendulum) * theta;
	theta += dt * theta_dot;
	x_dot += dt * (-mass_cart * g  * theta ) / mass_pendulum;
	x += dt * x_dot;
}

// void PendulumCart::control(float dt, float kp, float kd) {
// 	float force = kp * theta +kd * theta_dot;
// 	theta_dot += dt * ((M + m) * g/(M * l) * sin(theta) - force/(M * l));
// 	theta += dt * theta_dot;
// 	x_dot += dt * (-m * g * theta + force ) / M;
// 	x += dt * x_dot;
// }

void PendulumCart::control(float dt, float kp, float kd) {
	float force = kp * theta +kd * theta_dot;
	theta_dot += dt * ((mass_cart + mass_pendulum) * g/(mass_pendulum * length_pendulum) * sin(theta) - force/(mass_cart * length_pendulum));
	theta += dt * theta_dot;
	// friction
	x_dot *= 0.99;
	x_dot += dt * (-mass_pendulum * g * theta + force ) / mass_cart;
	x += dt * x_dot;
}

State PendulumCart::position(void) {
	return State { x, x + sin(theta), cos(theta) * length_pendulum, };
}

int main() {
	const float SCALE = 0.25;
	const float OFFSET = 0.10; // initialize GLFW;
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(1600, 600, "StableStar", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
	glClearColor(0, 0, 0, 1);
	PendulumCart p(4.0, 2.0, 3.0);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);
		p.control(0.01f, 50.0f, 10.0f);

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
			glVertex2f(cart, 0.0f); // pivot
			glVertex2f(pendulum_x, pendulum_y);       // bob
		glEnd();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}
