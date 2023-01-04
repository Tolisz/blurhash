#include "BigFactors.h"

#include <iostream>

#include "utils.h"

void BigFactors(cl_device_id& device, cl_context& context, cl_command_queue& queue,
    int img_W, int img_H, unsigned char* img, int xComp, int yComp)
{
    cl_int err;

    cl_program program = create_cl_program(device, context, "BigFactors.cl");

    // Table of images of size xComp on yComp
    // 
    unsigned int sizeBigFactors = xComp * yComp * (img_W * img_H * 3);
    unsigned char* BigFactors = new unsigned char[sizeBigFactors];
    if (!BigFactors)
    {
        std::cout << "BigFactors table allocation was unsuccessful: err = " << err << std::endl;
        exit(1);
    }

    // Memory creation on GPU

    cl_mem cl_BigFactors;
    cl_BigFactors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(char) * sizeBigFactors, BigFactors, &err);
    if (err)
    {
        std::cout << "Coudn't create a buffor for a cl_BigFactors on GPU: err = " << err << std::endl;
        exit(1);
    }

    // kelner creation
    cl_kernel kernel = create_cl_kelner(program, "BigFactors");
    
    // set kelner arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_BigFactors);
    if (err < 0) {
        printf("Couldn't set a kernel argument: cl_Sphere_X");
        exit(1);
    }

    // execute kelner
    cl_event kernel_event;

    size_t global_size[2];
    global_size[0] = 1;
    global_size[1] = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &kernel_event);
    if (err < 0)
    {
        perror("clEnqueueNDRangeKernel an error occured:");
        std::cout << "err = " << err << std::endl;
        exit(1);
    }

    /* Wait for kernel execution to complete */
    err = clWaitForEvents(1, &kernel_event);
    if (err < 0) {
        perror("Couldn't wait for events");
        exit(1);
    }

    clReleaseEvent(kernel_event);


    //err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &cl_buffor);
    //if (err < 0) {
    //    printf("Couldn't set a kernel argument: cl_Sphere_X");
    //    exit(1);
    //}
}