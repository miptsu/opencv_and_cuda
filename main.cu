#include "main.h"

class complex{
	float r, i;
public:
	__device__ complex(float a, float b) : r(a), i(b){}
	__device__ float magnitude2() const{
		return r*r + i*i;
	}
	__device__ complex operator* (const complex& a) const{
		return {r*a.r - i*a.i, i*a.r + r*a.i};
	}
	__device__ complex operator+(const complex& a) const{
		return {r + a.r, i + a.i};
	}
};

__device__ unsigned char julia(const complex& c, const int w, const int h, const int x, const int y, const float scale, const int dx, const int dy){
	float xx = (scale*(float)(dx-x))/((float)w/2);
	float yy = (scale*(float)(dy-y))/((float)h/2);
	complex a(xx, yy);
	//
	constexpr int q = 1000;
	for (int i=1; i<q; i++) {
		a = a*a + c;
		if(a.magnitude2() > 4){
			i = i*255/q;
			return (i<254? i : 255);
		}
	}
	return 0;
}

__global__  void kernel( unsigned char *ptr, const int w, const int h, const float sin, const float cos, const float scale, const int dx, const int dy) {
	const int x = (int)blockIdx.x;
	const int y = (int)blockIdx.y;
	constexpr float r = 0.71; // 0.7885;
	complex c(r*cos, r*sin);
	ptr[x + y*gridDim.x] = julia(c, w, h, x, y, scale, dx, dy);
}

cuCalc::cuCalc(unsigned char *host_bitmap, const int w, const int h, const float scaleInit, const float angle)
				: w(w), h(h), imgSz(w*h*sizeof(unsigned char)){
	cudaMalloc((void**)&devImg, imgSz);
	dim3 grid(w, h);
	float sin = std::sin(angle), cos = std::cos(angle);
	kernel<<<grid,1>>>(devImg, w, h, sin, cos, scaleInit, w/2, h/2);
	cudaDeviceSynchronize();
	cudaMemcpy(host_bitmap, devImg, imgSz, cudaMemcpyDeviceToHost);
}

cuCalc::~cuCalc(){
	cudaFree(devImg);
}

void cuCalc::recalc(unsigned char *host_bitmap, const float scale, const int dx, const int dy, float angle) const {
	dim3 grid(w, h);
	float sin = std::sin(angle), cos = std::cos(angle);
	kernel<<<grid,1>>>(devImg, w, h, sin, cos, scale, dx, dy);
	cudaMemcpy(host_bitmap, devImg, imgSz, cudaMemcpyDeviceToHost);
}
