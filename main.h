// Created by vit on 07.08.2021.

#ifndef MAIN_H
#define MAIN_H

struct cuCalc {
	unsigned char *devImg = nullptr;
	const unsigned long imgSz;
	const int w,h;

	cuCalc(unsigned char *host_bitmap,int w, int h, float scaleInit, float angle);
	~cuCalc();
	void recalc(unsigned char *host_bitmap, float scale, int dx, int dy, float angle) const;
};

#endif //MAIN_H
