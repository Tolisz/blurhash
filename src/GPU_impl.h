#pragma once

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <chrono>

std::chrono::microseconds computeGPU(int xComponents, int yComponents, int width, int height, unsigned char* rgb);

const char* BigFactors(cl_device_id& device, cl_context& context, cl_command_queue& queue,
	int img_W, int img_H, unsigned char* img, int xComp, int yComp);