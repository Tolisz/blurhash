#include "CPU_impl.h"

#include <iostream>
using namespace std::chrono;

#include "blurhash/encode.h"


microseconds computeCPU(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow)
{
	// This is an original algorithm taken from official blurhash GitHub
	// with minor modifications
	// -----------------------------------------------------------------
	

	std::cout << "\n-----------------\n Version [CPU] \n-----------------\n";

	auto start = high_resolution_clock::now();

	const char* hash = blurHashForPixels(xComponents, yComponents, width, height, rgb, width * 3);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	std::cout << hash << std::endl;

	std::cout << "Time = " << duration.count() << "\n\n";

	return duration;
}