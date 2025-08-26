#include <vector>
#include <iostream>
#include "pipeline.h"
using std::vector;

// warps <- a collection of threads
// sharedMemBytes <- shareds

// threadIdx.x <- index of the thread inside the block
// blockIdx.x <- index of the block in the grid
// blockDim.x <- how many threads are in a block
// gridDim.x <- how many blocks in the grid


// x ~ M[i, k]
// y ~ M[k, j]
// r ~ M[i, j]

__global__ void mat_mul_kernel(
	int i, int j, int k,
	const float *x,
	const float *y,
	float *out
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int row = tid / j;
	int col = tid % j;
	
	if (row < i && col < j) {
		float sum = 0.0f;
		for (int kdx = 0; kdx < k; kdx ++ ) {
			sum += x[row * k + kdx] * y[kdx * j + col];
		}
		out[row * j + col] = sum;
	}
}

// __global__ void mat_mul_kernel(
// 	int i, int j, int k,
// 	const float *x,
// 	const float *y,
// 	float *out
// ) {
// 	int kdx = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (kdx < k ) {
// 		for (int idx = 0; idx < i; idx++) {
// 			for (int jdx = 0; jdx < j; jdx++) {
// 				out[idx * k + jdx] += x[idx * k + kdx] * y[ kdx * j + jdx];
// 			}
// 		}
// 	}
// }

vector<float> mat_mul_host(
	int i, int j, int k,
	vector<float> &x,
	vector<float> &y
) {
	float *d_x, *d_y, *d_o;
	vector<float> o(i * j, 0);
	cudaMalloc(&d_x, i * k * sizeof(float));
	cudaMalloc(&d_y, j * k *  sizeof(float));
	cudaMalloc(&d_o, i * j * sizeof(float));

	cudaMemcpy(d_x, x.data(), i * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), j * k * sizeof(float), cudaMemcpyHostToDevice);

	int blocks = k;
	dim3 threads(16, 16);
	// dim3 blocks((j+15)/16, (i+15)/16);
	mat_mul_kernel<<<blocks, threads>>>(i, j, k, d_x, d_y, d_o);
	cudaMemcpy(o.data(), d_o, i * j * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_o);

    std::cout << o[0] << ", " << o[1] << "\n";
    std::cout << o[2] << ", " << o[3] << "\n";
	return o;
}


void predict_input(void) {
	int N = 1<<20;
	float c = 3.14f;
	vector<float> x(N, 2.0f);
	vector<float> y(N, 2.71f);

	float r = pipeline(N, c, x, y );
	std::cout << "total = " << r << "\n";

	// std::cout << "y[0] = " << y[0] << "\n";
	// std::cout << "y[n-1] = " << y[N-1] << "\n";
}


vector<float> mat_mul_inner(
	int i, int j, int k,
	const vector<float> &x,
	const vector<float> &y
) {
	vector<float>  r(i * j, 0.0f);
		
	for (int idx = 0; idx < i; idx ++) {
		for (int jdx = 0; jdx < j; jdx ++ ) {
			for (int kdx = 0; kdx < k; kdx ++) {
				r[idx * j + jdx] += x[idx * k + kdx] * y[kdx * j + jdx];
			}
		}
	}
	return r;
}

vector<float> mat_mul_outer(
	int i, int j, int k,
	const vector<float> &x,
	const vector<float> &y
) {
	vector<float> r(i * j, 0.0f);
	for (int kdx = 0; kdx < k; kdx++) {
		for (int idx = 0; idx < i; idx++) {
			for (int jdx = 0; jdx < j; jdx++) {
				r[ idx * j + jdx] += x[ idx * k + kdx] * y[ kdx * j + jdx];
			}
		}
	}

	return r;
}


int main(void) {
	// predict_input();
	vector<float> x{1.0, 2.0,
                    3.0, 4.0};  // shape 2x2
    vector<float> y{5.0, 6.0,
                    7.0, 8.0};  // shape 2x2


    vector<float> r = mat_mul_host(2, 2, 2, x, y);
    // vector<float> r = mat_mul_outer(2, 2, 2, x, y);
    // vector<float> r = mat_mul_inner(2, 2, 2, x, y);

    std::cout << "matrix\n";
    std::cout << r[0] << ", " << r[1] << "\n";
    std::cout << r[2] << ", " << r[3] << "\n";

}
