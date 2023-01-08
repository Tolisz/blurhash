#pragma once

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

char* read_file(const char* filename, size_t* size);

cl_program create_cl_program(cl_device_id& device, cl_context& context,
	const char* program_file);

cl_kernel create_cl_kelner(cl_program& program,
	const char* kelner_name);

cl_mem create_buffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr);

void set_argument(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);