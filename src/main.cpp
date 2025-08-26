// #include <vector>
#include <iostream>
#include <btBulletDynamicsCommon.h>
// using std::vector;

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
	std::cout << "hello world\n";
}
