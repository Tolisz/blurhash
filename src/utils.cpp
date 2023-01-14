#include "utils.h"

#include "error.h"

char* read_file(const char* filename, size_t* size)
{
    FILE* handle;
    char* buffer;

    /* Read program file and place content into buffer */
    //handle = fopen(filename, "rb");
    fopen_s(&handle, filename, "rb");
    if (handle == NULL) {
        ERROR("Couldn't find the file");
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
    if (err < 0) {
        ERROR("Couldn't create the program from file" << program_file << ": err = " << err);
    }

    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    printf("----------------------------------------------------------------------------\n [%s] Errors and Warnings \n----------------------------------------------------------------------------\n", program_file);

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    program_log = (char*)malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
    printf("%s", program_log);
    free(program_log);
    //exit(1);

    printf("\n----------------------------------------------------------------------------\n\n");

    return program;
}

cl_kernel create_cl_kelner(cl_program& program,
    const char* kelner_name)
{
    cl_int err;

    cl_kernel kernel;
    kernel = clCreateKernel(program, kelner_name, &err);
    if (err < 0) {
        ERROR("Coudn't create a kelner for " << kelner_name << ": err = " << err); 
    }

    return kernel;
}

cl_mem create_buffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr)
{
    cl_int err;

    cl_mem buffor;
    buffor = clCreateBuffer(context, flags, size, host_ptr, &err);
    if (err) {
        ERROR("Coudn't create a buffor for a cl_BigFactors on GPU: err = " << err);
    }

    return buffor;
}

void set_argument(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value)
{
    cl_int err;

    err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    if (err < 0) {
        char kernel_name[32];
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kernel_name), kernel_name, NULL);
        ERROR("Couldn't set the kernel " << arg_index << " argument: err = " << err << "Kelner name: " << kernel_name);
    }
}