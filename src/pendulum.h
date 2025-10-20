#pragma once
#include <tuple>
struct InvPendulum {
	float theta;
	float theta_d;
	float theta_dd;
	
	InvPendulum();
	void step(float dt);
	void control(float dt, float k_p, float k_d);
	std::tuple<float, float> position() const;
};
void simulation(void);
