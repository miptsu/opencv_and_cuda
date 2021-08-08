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
	applyColorMap(src, dst, COLORMAP_JET);
	return dst;
}

int main(){
	constexpr int w = 1920, h=1080;
	int dx = w/2, dy = h/2;
	float a = 1.62, scaleInit = 1.5;
	float scale = scaleInit;
	Mat img(h, w, CV_8UC1, Scalar(0));
	Mat imgC(h, w, CV_8UC3);
	//
	VideoWriter vw("./fractal.avi", CAP_FFMPEG, VideoWriter::fourcc('X','2','6','4'), 24, Size(w,h), true);
	bool record = false;
	//
	auto cc = cuCalc( img.data, w, h, scale, a);
	int key=32, lastKey=0;
	for(size_t cou=0; key!=27 && cou<1000; cou++){
		imshow("fractal", colorize(img, imgC));
		if(record && vw.isOpened()){ vw << imgC;}
		//
		if((char)lastKey == 'a'){       scale *= 0.99;} // cout << scale<<endl;
		else if((char)lastKey == 'z'){  scale *= (1/0.99);}
		else if((char)key == 'r'){      record = !record; cout<<"record state: "<<record<<endl;}
		else if(lastKey == keyLeft){    dx +=10;}
		else if(lastKey == keyRight){   dx -= 10;}
		else if(lastKey == keyUp){      dy +=10;}
		else if(lastKey == keyDown){    dy -= 10;}
		else if(lastKey == keyPgup){    a += 1e-5;}
		else if(lastKey == keyPgdown){  a -= 1e-5;}
		cc.recalc(img.data, scale, dx, dy, a);
		//
		if(key==32){ key = waitKey(0);}
		else { key = waitKey(10);}
		if(key>0){ lastKey = key; cout<<"key = "<<key<<endl;}
	}

	return 1;
}

