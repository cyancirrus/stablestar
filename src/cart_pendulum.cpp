#include <cmath>
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

void PendulumCart::control(float dt, float kp, float kd) {
	float force = kp * theta + kd * theta_dot;
	theta_dot += dt * ((mass_cart + mass_pendulum) * g * sin(theta) - force) /(mass_cart * length_pendulum);
	theta += dt * theta_dot;
	// friction
	x_dot *= 0.99;
	x_dot += dt * (force -mass_pendulum * g * theta ) / mass_cart;
	x += dt * x_dot;
}

State PendulumCart::position(void) {
	return State { x, float(x + sin(theta)) , float(cos(theta) * length_pendulum)};
}
