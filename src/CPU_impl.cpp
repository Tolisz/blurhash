#include "CPU_impl.h"

#include "blurhash/encode.h"
//#include "blurhash/encode.c"
#include <iostream>

void computeCPU(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow)
{
	const char* hash = blurHashForPixels(xComponents, yComponents, width, height, rgb, width * 3);

	std::cout << hash << std::endl;
}