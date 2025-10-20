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
		// std::uniform_real_distribution<float> dis(-0.1,0.1);
		std::uniform_real_distribution<float> dis(-1.0,1.0);

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
	std::tuple<float, float> position() const {
		return {sin(theta), cos(theta)};
	};
};


void simulation(void) {
	InvPendulum p;

	for(int i=0; i<100; ++i) {
		// simulate 9 natural steps
		for(int j=0; j<9; ++j)
			// approx continuous time
			p.step(0.01f);

		// apply control every 10th step new frequency
		p.control(0.1f, 20.0f, 6.0f);

		std::cout << p.theta << " " << p.theta_d << "\n";
	};
}

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
	// std::cout << p.theta << " " << p.theta_d << "\n";

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



// int main(void) {

// InvPendulum p;

// 	for(int i=0; i<100; ++i) {
// 		// simulate 9 natural steps
// 		for(int j=0; j<9; ++j)
// 			// approx continuous time
// 			p.step(0.01f);

// 		// apply control every 10th step new frequency
// 		p.control(0.1f, 20.0f, 6.0f);

// 		std::cout << p.theta << " " << p.theta_d << "\n";
// 	}

// GLFWwindow* window = glfwCreateWindow(800, 600, "StableStar", NULL, NULL);
// while (!glfwWindowShouldClose(window)) {
//     glClear(GL_COLOR_BUFFER_BIT);
    
//     glMatrixMode(GL_MODELVIEW);
//     glLoadIdentity();

//     glColor3f(1.0f, 0.0f, 1.0f);
//     glBegin(GL_LINES);
//         glVertex2f(-1.0f, -1.0f);
//         glVertex2f(1.0f, 1.0f);
//     glEnd();

//     glfwSwapBuffers(window);
//     glfwPollEvents();
// }
// 	return 0;
// }


// // obsolesced but semi working

// 	// if (!glfwInit()) return -1;
	
// 	// GLFWwindow* window = glfwCreateWindow(800, 600, "StableStar", NULL, NULL);
// 	// if (!window) { glfwTerminate(); return -1; }
	
// 	// glfwMakeContextCurrent(window);
// 	// if (glewInit() != GLEW_OK) {
// 	// 	std::cerr << "GLEW INIT FAILED \n";
// 	// 	return -1;
// 	// }

//     // glMatrixMode(GL_PROJECTION);
//     // glOrtho(-1, 1, -1, 1, -1, 1);

// 	// while (!glfwWindowShouldClose(window)) {
// 	// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//     	// glMatrixMode(GL_PROJECTION);
//     	// glLoadIdentity();

// 	// 	glColor3f(1.0f, 0.0f, 1.0f);
// 	// 		glBegin(GL_LINES);
// 	// 		// glVertex2f(-0.5f, -0.5f);
// 	// 		// glVertex2f(0.5f, 0.5f);
// 	// 		glVertex2f(-1.0f, -1.0f);
// 	// 		glVertex2f(1.0f, 1.0f);
// 	// 	// glEnd();

// 	// 	glfwSwapBuffers(window);
// 	// 	glfwPollEvents();
// 	// }
// 	// glfwDestroyWindow(window);
// 	// glfwTerminate();

// 	// std::cout << "new world\n";
