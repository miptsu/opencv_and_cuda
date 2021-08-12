// Created by vit on 08.08.2021.
/** Creates Julia set using CUDA for calculation and OpenCV for visualization. Press keys to animate. */

#include <iostream>
#include <future>
#include "main.h"
#include "opencv4/opencv2/opencv.hpp"

using namespace cv;
using namespace std;

constexpr int keyLeft = 81, keyRight = 83, keyUp = 82, keyDown = 84;
constexpr int keyPgup = 85, keyPgdown = 86;

int main(){
	constexpr int w = 64*30, h = 64*30; // 64*30=1920
	static_assert(w%64==0); // since we have 64-threaded blocks
	int dx = w/2, dy = h/2;
	Mat tmp1(h, w, CV_8UC1, Scalar(0));
	Mat tmp2(h, w, CV_8UC1, Scalar(0)); // tmp1 is showing while tmp2 is calculating and vise versa
	Mat img, imgC(h, w, CV_8UC3);
	float a = 1.62, scaleInit = 3;
	float scale = scaleInit;

	enum Zoom{IN, OUT, NO};
	enum Motion{LEFT, RIGHT, UP, DOWN, STOP};
	enum Evolution{GO, BACK, PAUSE}; // since we don't use 'enum class' we need different words
	Zoom z = NO;
	Motion m = STOP;
	Evolution e = PAUSE;
	bool record = false, loop = false;
	// creates file always, even if no record requested - it is ok
	VideoWriter vw("./fractal.avi", CAP_FFMPEG, VideoWriter::fourcc('X','2','6','4'), 24, Size(w,h), true);

	auto cc = CuCalc(tmp1.data, w, h, scale, dx, dy, a);
	img = tmp2;

	int key=32, lastKey = 32, delay=20; // fps upper limit is 1000/delay
	auto t = chrono::steady_clock::now();
	double avrgLoop=0, dDraw, avrgDraw, avrgCalc;
	float da=1e-5, qs=0.99, dqs=0.001;
	int d2l=16; // power of 2 preferable
	auto thr = std::thread();
	for(size_t cou=0; key!=27; cou++){
		auto recalcResult = std::async(std::launch::async, &CuCalc::recalc, &cc, img.data, scale, dx, dy, a);
		img = cou%2 ? tmp2 : tmp1; // shallow copy: tmp1 is showing while tmp2 is calculating and vise versa

		// draw section
		const auto t0 = chrono::steady_clock::now();
		applyColorMap(img, imgC, COLORMAP_MAGMA); // takes a lot of resources, use UMat or draw bw-image 'img' in case of freezing
		imshow("fractal", imgC); // imshow and waitKey must be in the main thread
		if(record && vw.isOpened()){ vw << imgC;}
		const auto t1 = chrono::steady_clock::now();

		// fps limiting section
		if(key==32 || !loop){ // space stops all
			key = waitKey(0);
			z = NO; m = STOP; e = PAUSE;
		} else {
			const double dt = (double)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t).count();
			avrgLoop = avrgLoop==0 ? dt : 0.95*avrgLoop + 0.05*dt;
			key = waitKey(dt>=delay? 1 : (int)(1+delay-dt) ); // -1 if no key pressed
		}
		t = chrono::steady_clock::now();

		const auto t2 = chrono::steady_clock::now();
		recalcResult.get(); // cc.recalc(img.data, scale, dx, dy, a);
		const auto t3 = chrono::steady_clock::now(); // expect ~zero time here

		// process keypress
		if(key==keyPgup){     e = e==BACK ? PAUSE : GO; lastKey = key;}
		if(key==keyPgdown){   e = e==GO ? PAUSE : BACK; lastKey = key;}
		if( key==keyLeft){    m = m==RIGHT ? STOP : LEFT; lastKey = key;}
		if( key==keyRight){   m = m==LEFT ? STOP : RIGHT; lastKey = key;}
		if( key==keyUp){      m = m==DOWN ? STOP : UP; lastKey = key;}
		if( key==keyDown){    m = m==UP ? STOP : DOWN; lastKey = key;}
		if((char)key == 'a'){ z = z==OUT ? NO : IN; lastKey = key;}
		if((char)key == 'z'){ z = z==IN ? NO : OUT; lastKey = key;}
		if( e==PAUSE && m==STOP && z == NO){ loop = false; lastKey = 32;}
		else { loop = true;}
		// speed control
		if((char)key == 'f' && (lastKey==keyPgup || lastKey==keyPgdown)){ da *= 2;}
		if((char)key == 's' && (lastKey==keyPgup || lastKey==keyPgdown)){ da /= 2;}
		if((char)key == 'f' && (lastKey==keyLeft || lastKey==keyRight || lastKey==keyUp || lastKey==keyDown)){ d2l *= 2;}
		if((char)key == 's' && (lastKey==keyLeft || lastKey==keyRight || lastKey==keyUp || lastKey==keyDown)){ d2l = d2l>2 ? d2l/2 : 2;}
		if((char)key == 'f' && ((char)lastKey == 'a' || (char)lastKey == 'z')){ qs = qs-dqs>=0.9f  ? qs-dqs : 0.9f;}
		if((char)key == 's' && ((char)lastKey == 'a' || (char)lastKey == 'z')){ qs = qs+dqs<=0.999f ? qs+dqs : 0.999f;}
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
		if((char)key == 'r'){ record = !record; cout<<"record state: "<<record<<endl;}
		// /process keypress

		dDraw = (double)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
		const double tCalc = (double)chrono::duration_cast<chrono::milliseconds>(t3-t2).count();
		avrgDraw = cou<2 ? dDraw : 0.95*avrgDraw + 0.05*dDraw;
		avrgCalc = cou<2 ? tCalc : 0.95*avrgCalc + 0.05*tCalc; // calc time over draw time
		if(cou>0 && cou%(size_t)(50)==49){ cout<<avrgLoop<<" ("<<avrgDraw<<"; "<<avrgCalc<<") ms"<<endl; }
	}

	return 0;
}

