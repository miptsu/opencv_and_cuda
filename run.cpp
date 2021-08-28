// Created by vit on 08.08.2021.
/** Creates Julia set using CUDA for calculation and OpenCV for visualization. Press keys to animate. */

#include <iostream>
#include <future>
#include "main.h"
#include "opencv4/opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace std;

constexpr int keyLeft = 81, keyRight = 83, keyUp = 82, keyDown = 84;
constexpr int keyPgup = 85, keyPgdown = 86;

int main(){
	constexpr int w = 64*30, h = 64*30; // 64*30=1920
	static_assert(w%64==0); // since we have 64-threaded blocks
	int dx = w/2, dy = h/2;
	float a = 3.443, r = 0.678, scaleInit = 3; // 0.7885; different r-values in [0;~1] are possible;
	float scale = scaleInit;

	enum Zoom{IN, OUT, NO};
	enum Motion{LEFT, RIGHT, UP, DOWN, STOP};
	enum Evolution{GO, BACK, PAUSE}; // since we don't use 'enum class' we need different words
	Zoom z = NO;
	Motion m = STOP;
	Evolution e = PAUSE;
	bool record = false, loop = false;
	int colormap = 21;
	// creates file always, even if no record requested - it is ok
	VideoWriter vw("./fractal.avi", CAP_FFMPEG, VideoWriter::fourcc('X','2','6','4'), 24, Size(w,h), true);

	Mat img(h, w, CV_8UC1), imgC(h, w, CV_8UC3);
	cuda::registerPageLocked(img);

	const auto cc1 = CuCalc(nullptr, w, h, scale, dx, dy, r, a); // first argument could be nullptr - this let use both Mat and GpuMat
	const auto cc2 = CuCalc(nullptr, w, h, scale, dx, dy, r, a); // to avoid copying we can either create 2 images inside CuCalc or create 2 instances of CuCalc
	cuda::GpuMat tmp1_g(h, w, CV_8UC1, cc1.getDevImg()), tmp2_g(h, w, CV_8UC1, cc2.getDevImg()), imgC_g(h, w, CV_8UC3);
	auto img_g = tmp1_g;
	namedWindow("fractal", WINDOW_OPENGL | WINDOW_AUTOSIZE); // OpenGL mode let us show GpuMat without copy to Mat

	int key=32, lastKey = 32, delay=15; // fps upper limit is approximately 1000/delay
	auto t = chrono::steady_clock::now();
	auto tt = t;
	double avrgLoop=0, dDraw, avrgDraw, avrgCalc;
	float da=1e-5, qs=0.99, dqs=0.0005;
	int d2l=16; // power of 2 preferable
	auto thr = std::thread();
	for(size_t cou=0; key!=27; cou++){
		const bool is1 = img_g.data==tmp1_g.data;
		auto recalcResult = std::async(std::launch::async, &CuCalc::recalc, is1 ? &cc2 : &cc1, nullptr, scale, dx, dy, r, a);
		img_g = is1 ? tmp2_g : tmp1_g; // shallow copy: in loop mode tmp1 is showing while tmp2 is calculating and vise versa
		// draw section
		if(!loop){
			recalcResult.get();}
		const auto t0 = chrono::steady_clock::now();
		if(colormap<22){
			img_g.download(img); // device to host copy since there are /
			applyColorMap(img, imgC, colormap); // no applyColorMap(...) in cuda:: TODO add cuda::applyColorMap
			imshow("fractal", imgC); // imshow and waitKey must be in the main thread
		} else { // bw image
			imshow("fractal", img_g);
		}

		if(record && vw.isOpened()){ vw << imgC;}
		const auto t1 = chrono::steady_clock::now();

		// fps limiting section
		if(key==32 || !loop){ // space stops all
			key = waitKey(0);
			z = NO; m = STOP; e = PAUSE;
		} else {
			const double dt = (double)chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - t).count();
			avrgLoop = avrgLoop==0 ? dt : 0.95*avrgLoop + 0.05*dt;
			key = waitKey(1+dt*1e-3>=delay? 1 : (int)(delay-dt*1e-3) ); // -1 if no key pressed
		}
		t = chrono::steady_clock::now();

		// process keypress
		if(key==keyPgup){     e = e==BACK ? PAUSE : GO; lastKey = key;}
		if(key==keyPgdown){   e = e==GO ? PAUSE : BACK; lastKey = key;}
		if( key==keyLeft){    m = m==RIGHT ? STOP : LEFT; lastKey = key;}
		if( key==keyRight){   m = m==LEFT ? STOP : RIGHT; lastKey = key;}
		if( key==keyUp){      m = m==DOWN ? STOP : UP; lastKey = key;}
		if( key==keyDown){    m = m==UP ? STOP : DOWN; lastKey = key;}
		if((char)key == 'a'){ z = z==OUT ? NO : IN; lastKey = key;}
		if((char)key == 'z'){ z = z==IN ? NO : OUT; lastKey = key;}
		// speed control
		if((char)key == 'f' && (lastKey==keyPgup || lastKey==keyPgdown)){ da *= 2;}
		if((char)key == 's' && (lastKey==keyPgup || lastKey==keyPgdown)){ da /= 2;}
		if((char)key == 'f' && (lastKey==keyLeft || lastKey==keyRight || lastKey==keyUp || lastKey==keyDown)){ d2l *= 2;}
		if((char)key == 's' && (lastKey==keyLeft || lastKey==keyRight || lastKey==keyUp || lastKey==keyDown)){ d2l = d2l>2 ? d2l/2 : 2;}
		if((char)key == 'f' && ((char)lastKey == 'a' || (char)lastKey == 'z')){ qs = qs-dqs>=0.9f  ? qs-dqs : 0.9f;}
		if((char)key == 's' && ((char)lastKey == 'a' || (char)lastKey == 'z')){ qs = qs+dqs<=0.9995f ? qs+dqs : 0.9995f;}
		// change parameters
		if(z == IN){    scale *= qs;}
		if(z == OUT){   scale /= qs;}
		if(m == LEFT){  dx += d2l;}
		if(m == RIGHT){ dx -= d2l;}
		if(m == UP){    dy += d2l;}
		if(m == DOWN){  dy -= d2l;}
		if(e == GO){    a += da;}
		if(e == BACK){  a -= da;}
		//
		if((char)key == 'c'){ colormap = colormap<=0 ? 0 : colormap-1;}
		if((char)key == 'd'){ colormap = colormap>=22 ? 22 : colormap+1;}
		if((char)key == 'r'){ record = !record; cout<<"record state: "<<record<<endl;}
		if((char)key == 'p'){
			stringstream ss; ss << "picture-" << cou << ".png";
			imwrite(ss.str(), img);
			cout<<"image writed to "<<ss.str()<<endl;
		}
		if((char)key == '['){ r -= 0.001; cout<<"r="<<r<<endl;}
		if((char)key == ']'){ r += 0.001; cout<<"r="<<r<<endl;}
		// /process keypress

		const auto t2 = chrono::steady_clock::now();
		if(loop){
			recalcResult.get(); // cc1-2.recalc(img.data, scale, dx, dy, a);
		}
		const auto t3 = chrono::steady_clock::now(); // expect ~zero time here

		//
		if( e==PAUSE && m==STOP && z == NO){ loop = false; lastKey = key;}
		else { loop = true;}
		// stats
		dDraw = (double)chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
		const double tCalc = (double)chrono::duration_cast<chrono::microseconds>(t3-t2).count();
		avrgDraw = cou<2 ? dDraw : 0.95*avrgDraw + 0.05*dDraw;
		avrgCalc = cou<2 ? tCalc : 0.95*avrgCalc + 0.05*tCalc; // calc time over draw time
		if(cou>0 && cou%(size_t)(50)==49){
			auto tnow = chrono::steady_clock::now();
			const double fps = 50*1000/(double)chrono::duration_cast<chrono::microseconds>(tnow - tt).count();
			cout<<avrgLoop*1e-3<<" ("<<avrgDraw*1e-3<<"; "<<avrgCalc*1e-3<<") ms;  fps="<<1e3*fps<<endl;
			tt = tnow;
		}
	}

	return 0;
}

