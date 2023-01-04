#pragma once

#include <CL/cl.h>

char* read_file(const char* filename, size_t* size);
cl_program create_cl_program(cl_device_id& device, cl_context& context,
	const char* program_file);
cl_kernel create_cl_kelner(cl_program& program,
	const char* kelner_name);