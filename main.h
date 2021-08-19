// Created by vit on 07.08.2021.

#ifndef MAIN_H
#define MAIN_H

class CuCalc {
	const unsigned long imgSz;
	const int w,h;
	unsigned char *devImg = nullptr;
	const unsigned int n = 64; // threads per block, w%n==0
public:

	CuCalc(unsigned char *bitmap, int w, int h, float scaleInit, int dx, int dy, float r, float angle);
	~CuCalc();
	unsigned char *getDevImg() const;

	/**@param host_bitmap -- pointer to host bitmap, if nullptr then no copy from device to host
	 * @return pointer to gpu memory bitmap */
	unsigned char* recalc(unsigned char *host_bitmap, float scale, int dx, int dy, float r, float angle) const;
};

#endif //MAIN_H
