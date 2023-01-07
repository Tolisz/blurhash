#pragma once

#include <CL/cl.h>

void BigFactors(cl_device_id& device, cl_context& context, cl_command_queue& queue,
	int img_W, int img_H, unsigned char* img, int xComp, int yComp);