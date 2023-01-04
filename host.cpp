#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <CL/cl.h>

#include "BigFactors.h"

int main()
{
    // blurhash components number
    int xComp = 8;
    int yComp = 8;

    if (xComp < 1 || xComp > 9 || yComp < 1 || yComp > 9)
    {
        std::cout << "Number of components must be between 1 and 8" << std::endl;
        return -1;
    }

    // image loading
    int width, height, nrChannels;
    unsigned char* img_data = stbi_load("img/restaurant.jpg", &width, &height, &nrChannels, 3);
    if (!img_data)
    {
        std::cout << "Can not read your image file, try to use another one" << std::endl;
        return -1;
    }


    /// ---------------------------------------------------
    /// ---------------------------------------------------
    ///
    ///             OPEN CL CONFIGURATION
    ///
    /// ---------------------------------------------------
    /// ---------------------------------------------------

    cl_int err;

    /// ------------------------------------
    ///             platform 
    /// ------------------------------------
    cl_platform_id platform;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't find any platforms");
        exit(1);
    }

    /// ------------------------------------
    ///               device 
    /// ------------------------------------
    cl_device_id device;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err < 0) {
        perror("Couldn't find any devices");
        exit(1);
    }

    /// ------------------------------------
    ///               context 
    /// ------------------------------------
    cl_context context;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    /// ------------------------------------
    ///               queue 
    /// ------------------------------------
    cl_command_queue queue;

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    };


    BigFactors(device, context, queue, width, height, img_data, xComp, yComp);


    ///// ------------------------------------
    /////               program 
    ///// ------------------------------------

    //cl_program program;
    //char* program_buffer, * program_log;
    //size_t program_size, log_size;

    //program_buffer = read_file(PROGRAM_FILE, &program_size);
    //program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, &err);
    //if (err < 0)
    //{
    //    perror("Couldn't create the program");
    //    exit(1);
    //}

    //free(program_buffer);

    //err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    //if (err < 0) {

    //    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    //    program_log = (char*)malloc(log_size + 1);
    //    program_log[log_size] = '\0';
    //    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
    //    printf("%s\n", program_log);
    //    free(program_log);
    //    exit(1);
    //}



    /// ------------------------------------
    ///               kernel 
    /// ------------------------------------
    //cl_kernel kernel;

    //kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    //if (err < 0) {
    //    printf("Couldn't create a kernel: %d", err);
    //    exit(1);
    //};
    


    stbi_image_free(img_data);


	std::cout << "Hello World" << std::endl;
}

