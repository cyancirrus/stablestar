#include <iostream>
#include <tuple>
#include <random>
#include <cmath>
static constexpr float g = 9.18f;

struct InvPendulum {
	float theta;
	float theta_d;
	// g/l sin(theta)
	float theta_dd;

	InvPendulum();
	void step(float dt);
	void control(float dt, float k_p, float k_d);
	std::tuple<float, float> position() const;
};
	
InvPendulum::InvPendulum() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-1.0,1.0);
	theta = dis(gen);
	theta_d= 0.0f;
	theta_dd=0.0f;
}

void InvPendulum::step(float dt) {
	float l = 1.0f;
	theta_dd = g/l * sin(theta);
	theta_d += dt * theta_dd;
	theta += dt * theta_d;
}
	
void InvPendulum::control(float dt, float k_p, float k_d) {
	float l = 1.0f, m = 1.0f;
	theta_dd = g/l * sin(theta)
		- k_d / (m * l * l) * theta_d
		- k_p / (m * l * l) * theta
	;
	theta_d += dt * theta_dd;
	theta += dt * theta_d;
}
	
std::tuple<float, float> InvPendulum::position() const {
	return {sin(theta), cos(theta)};
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
