#include "utils.h"

#include <iostream>

char* read_file(const char* filename, size_t* size)
{
    FILE* handle;
    char* buffer;

    /* Read program file and place content into buffer */
    handle = fopen(filename, "rb");
    if (handle == NULL) {
        perror("Couldn't find the file");
        exit(1);
    }
    fseek(handle, 0, SEEK_END);
    *size = (size_t)ftell(handle);
    rewind(handle);
    buffer = (char*)malloc(*size + 1);
    buffer[*size] = '\0';
    fread(buffer, sizeof(char), *size, handle);
    fclose(handle);

    return buffer;
}

cl_program create_cl_program(cl_device_id& device, cl_context& context,
    const char* program_file)
{
    cl_int err;

    cl_program program;
    char* program_buffer, * program_log;
    size_t program_size, log_size;

    program_buffer = read_file(program_file, &program_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        std::cout << "Couldn't create the program from file" << program_file << ": err = " << err << std::endl;
        exit(1);
    }

    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

cl_kernel create_cl_kelner(cl_program& program,
    const char* kelner_name)
{
    cl_int err;

    cl_kernel kernel;
    kernel = clCreateKernel(program, kelner_name, &err);
    if (err < 0) {
        std::cout << "Coudn't create a kelner for " << kelner_name << ": err = " << err << std::endl;
        exit(1);
    };

    return kernel;
}