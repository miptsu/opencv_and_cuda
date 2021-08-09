// Created by vit on 08.08.2021.
/** Creates Julia set using CUDA for calculation and OpenCV for visualization. Press keys to animate. */

#include <iostream>
#include "main.h"
#include "opencv4/opencv2/opencv.hpp"

using namespace cv;
using namespace std;

constexpr int keyLeft = 81, keyRight = 83, keyUp = 82, keyDown = 84;
constexpr int keyPgup = 85, keyPgdown = 86;

Mat& colorize(const Mat& src, Mat& dst){
	applyColorMap(src, dst, COLORMAP_MAGMA);
	return dst;
}

int main(){
	constexpr int w = 64*30, h=1080; // 64*30=1920
	static_assert(w%64==0); // since we have 64-threaded blocks
	int dx = w/2, dy = h/2;
	float a = 1.62, scaleInit = 3;
	float scale = scaleInit;
	Mat img(h, w, CV_8UC1, Scalar(0));
	Mat imgC(h, w, CV_8UC3);
	//
	VideoWriter vw("./fractal.avi", CAP_FFMPEG, VideoWriter::fourcc('X','2','6','4'), 24, Size(w,h), true);
	bool record = false;
	//
	auto cc = cuCalc(img.data, w, h, scale, dx, dy, a);
	int key=32, lastKey=0, delay=20;
	double avrgDt=0;
	float da = 1e-5;
	for(size_t cou=0; key!=27; cou++){
		const auto t0 = chrono::steady_clock::now();
		imshow("fractal", colorize(img, imgC));
		if(record && vw.isOpened()){ vw << imgC;}
		//
		if((char)lastKey == 'a'){       scale *= 0.99;}
		else if((char)lastKey == 'z'){  scale *= (1/0.99);}
		else if(lastKey == keyLeft){    dx +=10;}
		else if(lastKey == keyRight){   dx -= 10;}
		else if(lastKey == keyUp){      dy +=10;}
		else if(lastKey == keyDown){    dy -= 10;}
		else if(lastKey == keyPgup){    a += da;}
		else if(lastKey == keyPgdown){  a -= da;}
		if((char)key == 'f'){           da *= 2;}
		else if((char)key == 's'){      da *= 1./2;}
		else if((char)key == 'r'){      record = !record; cout<<"record state: "<<record<<endl;}
		//
		cc.recalc(img.data, scale, dx, dy, a);
		//
		if(key==32){ key = waitKey(0);}
		else {
			const double dt = (double)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-t0).count();
			if(cou==0){ avrgDt = dt;
			} else { avrgDt = 0.95*avrgDt + 0.05*dt;}
			key = waitKey(dt>=delay? 1 : (int)(1 + delay-dt) );}
		//
		if(key>0 && (char)key !='f' && (char)key !='s'){ // -1 if no key pressed
			lastKey = key; cout<<"key = "<<key<<endl;
		}
		if(cou>0 && cou%(int)(1000/delay)==0){ cout<<"dt = "<<avrgDt<<" ms"<<endl; }
	}

	return 0;
}

