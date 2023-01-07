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
        std::cout << "BigFactors table allocation was unsuccessful" << std::endl;
        exit(1);
    }

    // Memory creation on GPU
    cl_mem cl_BigFactors = create_buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(char) * sizeBigFactors, BigFactors);
    cl_mem cl_img = create_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * img_W * img_H * 3, img);

    // kelner creation
    cl_kernel kernel = create_cl_kelner(program, "BigFactors");
    
    // set kelner arguments
    set_argument(kernel, 0, sizeof(cl_mem), &cl_BigFactors);
    set_argument(kernel, 1, sizeof(cl_mem), &cl_img);
    
    set_argument(kernel, 2, sizeof(int), &img_W);
    set_argument(kernel, 3, sizeof(int), &img_H);
    set_argument(kernel, 4, sizeof(int), &xComp);
    set_argument(kernel, 5, sizeof(int), &yComp);

    // execute kelner
    cl_event kernel_event;

    size_t global_size[2];
    global_size[0] = xComp * img_W * 3;
    global_size[1] = yComp* img_H;

    //size_t local_size[2];
    //local_size[0] = 2; //img_W * 3;
    //local_size[1] = 2; //img_H;

    //std::cout << "x = " << global_size[0] << "y = " << global_size[1] << std::endl;

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &kernel_event);
    if (err < 0)
    {
        std::cout << "in BigFactors in clEnqueueNDRangeKernel an error occured: " << "err = " << err << std::endl;
        exit(1);
    }

    /* Wait for kernel execution to complete */
    err = clWaitForEvents(1, &kernel_event);
    if (err < 0) {
        perror("Couldn't wait for events in BigFactors");
        exit(1);
    }

    clReleaseEvent(kernel_event);
}