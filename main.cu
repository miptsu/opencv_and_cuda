#include "main.h"

class Complex{
	float r, i;
public:
	__device__ Complex(float a, float b) : r(a), i(b){}
	__device__ float magnitude2() const{
		return r*r + i*i;
	}
	__device__ Complex operator* (const Complex& a) const{
		return {r*a.r - i*a.i, i*a.r + r*a.i};
	}
	__device__ Complex operator+(const Complex& a) const{
		return {r + a.r, i + a.i};
	}
};

__device__ unsigned char julia(const Complex& c, const int w, const int h, const int x, const int y, const float scale, const int dx, const int dy){
	float xx = (scale*(float)(dx-x))/((float)w);
	float yy = (scale*(float)(dy-y))/((float)h);
	Complex a(xx, yy);
	//
	constexpr uint unroll = 4; // loop unroll and 'if' check reducing; profiling shows 4 is the best value
	constexpr uint q = 1024/unroll; // 1024 is enough and 1024/4=256 -- we are lucky!
	for (uint i=0; i<q; i++) {
		for (uint j=0; j<unroll; j++) {
			a = a*a + c;
		}
		if(a.magnitude2() > 4){
			return (unsigned char)(__fsqrt_rd(255*255*(float)i/q));
		}
	}
	return 255;
}

__global__  void kernel( unsigned char *ptr, const int w, const int h, const float r, const float sin, const float cos, const float scale, const int dx, const int dy) {
	const int x = (int)(threadIdx.x + blockIdx.x*blockDim.x);
	const int y = (int)blockIdx.y;
	const Complex c(r*sin, r*cos);
	ptr[x + y*gridDim.x*blockDim.x] = julia(c, w, h, x, y, scale, dx, dy);
}

CuCalc::CuCalc(unsigned char *host_bitmap, const int w, const int h, const float scaleInit, const int dx, const int dy, const float r, const float angle)
				: w(w), h(h), imgSz(w*h*sizeof(unsigned char)){
	cudaMalloc((void**)&devImg, imgSz);
	this->recalc(host_bitmap, scaleInit, dx, dy, r, angle);
}

CuCalc::~CuCalc(){
	cudaFree(devImg);
}

int CuCalc::recalc(unsigned char *host_bitmap, const float scale, const int dx, const int dy, const float r, const float angle) const {
	if(w%n != 0){ return 1;}
	dim3 threads(n, 1);
	dim3 grid(w/n, h);
	float sin = std::sin(angle), cos = std::cos(angle);
	kernel<<<grid,threads>>>(devImg, w, h, r, sin, cos, scale, dx, dy);
	cudaDeviceSynchronize();
	cudaMemcpy(host_bitmap, devImg, imgSz, cudaMemcpyDeviceToHost);
	return 0;
}
