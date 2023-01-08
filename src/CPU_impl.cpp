#include "CPU_impl.h"

#include <iostream>
#include <chrono>
using namespace std::chrono;

#include "blurhash/encode.h"


void computeCPU(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow)
{
	std::cout << "Version [CPU] \n-----------------\n\n";

	auto start = high_resolution_clock::now();

	const char* hash = blurHashForPixels(xComponents, yComponents, width, height, rgb, width * 3);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	std::cout << hash << std::endl;

	std::cout << "\nTime = " << duration.count() << "\n\n";
}