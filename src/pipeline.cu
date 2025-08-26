#include "pipeline.h"
#include <vector>

// warps <- a collection of threads
// sharedMemBytes <- shareds

// threadIdx.x <- index of the thread inside the block
// blockIdx.x <- index of the block in the grid
// blockDim.x <- how many threads are in a block
// gridDim.x <- how many blocks in the grid

constexpr int BLOCKSIZE = 256;

__global__ void add(int n, float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) y[i] = x[i] +	y[i];
}

__global__ void scale(int n, float c, float *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) x[i] = x[i] * c;
}

__global__ void reduce_sum(int n, const float *in, 	float *out) {
	__shared__ float smem[BLOCKSIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	
	smem[tid] = (i < n ? in[i] : 0.0f);
	__syncthreads();

	for (int stride=blockDim.x / 2; stride > 0; stride >>=1) {
		if (tid < stride) {
			smem[tid] += smem[tid + stride];
		}
		__syncthreads();
	}; 

	if (tid == 0 ) out[blockIdx.x] = smem[0];
}

__global__ void final_reduce(int n, const float *in, float *out) {
	__shared__ float smem[BLOCKSIZE];
	int i = threadIdx.x;
	smem[i] = (i < n ? in[i]: 0.0f);
	__syncthreads();

	for (int stride=blockDim.x / 2; stride > 0; stride >>=1) {
		if (i < stride) {
			smem[i] += smem[i + stride];
		}
		__syncthreads();
	}
	if (i==0) *out = smem[0];
}

__global__ void reduce_sum_atomic(int n, float *in, float *global_sum) {
	__shared__ float smem[BLOCKSIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	smem[tid] = (i < n ? in[i] : 0.0f);
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1 ) {
		if (tid < stride) {
			smem[tid] += smem[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(global_sum, smem[0]);
	}
}


float reduce_core(std::vector<float> *in) {
	float r = 0.0f;
	for (float v: *in) r += v;
	return r;
}

float pipeline(
	int n,
	float c,
	std::vector<float>& x,
	std::vector<float>& y
) {
	int blocks = (n + BLOCKSIZE - 1 ) / BLOCKSIZE;
	float *d_x, *d_y, *d_bs;
	std::vector<float> bs(blocks);

	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, n * sizeof(float));
	cudaMalloc(&d_bs, blocks*sizeof(float));


	cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaError_t err = cudaGetLastError();

	add<<<blocks, BLOCKSIZE>>>(n, d_x, d_y);
	scale<<<blocks, BLOCKSIZE>>>(n, c, d_y);
	reduce_sum<<<blocks, BLOCKSIZE>>>(n, d_y, d_bs);
	cudaMemcpy(bs.data(), d_bs, blocks * sizeof(float), cudaMemcpyDeviceToHost);
	float r = reduce_core(&bs);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_bs);
	
	return r;
}
