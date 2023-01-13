#pragma once

#include <chrono>

std::chrono::microseconds computeCPU(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow);