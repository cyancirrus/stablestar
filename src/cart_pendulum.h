#pragma once
struct State {
	float cart_x;
	float pendulum_x;
	float pendulum_y;
};


struct PendulumCart {
	float theta;
	float theta_d;
	float theta_dd;
	
	PendulumCart(float m_cart, float m_pendulum, float l_pendulum);
	void step(float dt);
	void control(float dt, float kp, float kd);
	State position(void);
};
void simulation(void);
