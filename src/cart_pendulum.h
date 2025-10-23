#pragma once
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
void simulation(void);
