#include "GPU_impl.h"

#include "error.h"
#include "utils.h"
#include <iostream>
#include <cmath>

void computeGPU(int xComponents, int yComponents, int width, int height, unsigned char* img_data)
{
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
        ERROR("Couldn't find any platforms");
    }

    /// ------------------------------------
    ///               device 
    /// ------------------------------------
    cl_device_id device;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err < 0) {
        ERROR("Couldn't find any devices");
    }

    /// ------------------------------------
    ///               context 
    /// ------------------------------------
    cl_context context;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        ERROR("Couldn't create a context");
    }

    /// ------------------------------------
    ///               queue 
    /// ------------------------------------
    cl_command_queue queue;

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        ERROR("Couldn't create a command queue");
    };


    BigFactors(device, context, queue, width, height, img_data, xComponents, yComponents);
}

void BigFactors(cl_device_id& device, cl_context& context, cl_command_queue& queue,
    int img_W, int img_H, unsigned char* img, int xComponents, int yComponents)
{
    cl_int err;

    cl_program program = create_cl_program(device, context, "kernels/BigFactors.cl");

    // Table of images of size xComp on yComp
    // 
    unsigned int sizeBigFactors = img_W * img_H * 3;
    float* BigFactors = new float[sizeBigFactors];
    if (!BigFactors) {
        ERROR("BigFactors table allocation was unsuccessful");
    }

    // Memory creation on GPU
    cl_mem cl_BigFactors = create_buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * sizeBigFactors, BigFactors);
    cl_mem cl_img = create_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * img_W * img_H * 3, img);

    // kelner creation
    cl_kernel kernel = create_cl_kelner(program, "BigFactors");

    // set kelner arguments
    set_argument(kernel, 0, sizeof(cl_mem), &cl_BigFactors);
    set_argument(kernel, 1, sizeof(cl_mem), &cl_img);

    set_argument(kernel, 2, sizeof(int), &img_W);
    set_argument(kernel, 3, sizeof(int), &img_H);
    set_argument(kernel, 4, sizeof(int), &xComponents);
    set_argument(kernel, 5, sizeof(int), &yComponents);


    // kelner 2 
    cl_kernel kernel2 = create_cl_kelner(program, "sumVertically");

    // set kelner 2 arguments
    set_argument(kernel2, 0, sizeof(cl_mem), &cl_BigFactors);
    set_argument(kernel2, 1, sizeof(int), &img_W);
    set_argument(kernel2, 2, sizeof(int), &img_H);


    // Maksymalna liczba wątków które mogą być w jednym work-group (CUDA: w jednym blocku)
    // 
    size_t maxWorkGroupSize;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
    //size_t dimSize = (size_t)std::sqrt((double)maxWorkGroupSize);
    
    std::cout << "Maximal work-group size = " << maxWorkGroupSize << std::endl;



    size_t global_size[2];
    global_size[0] = img_W + (maxWorkGroupSize - (img_W) % maxWorkGroupSize);
    global_size[1] = img_H;
    
    size_t local_size[2];
    local_size[0] = maxWorkGroupSize;
    local_size[1] = 1;



    size_t global_size_kelner2[2]; 
    global_size_kelner2[0] = 1;
    global_size_kelner2[1] = img_H + (maxWorkGroupSize - (img_H) % maxWorkGroupSize);

    size_t local_size_kelner2[2];
    local_size_kelner2[0] = 1;
    local_size_kelner2[1] = maxWorkGroupSize;


    for (int y = 3; y < 4/*yComponents*/; y++)
    {
        for (int x = 4; x < 5/*xComponents*/; x++)
        {
            set_argument(kernel, 6, sizeof(int), &x);
            set_argument(kernel, 7, sizeof(int), &y);

            std::cout << "x = " << x << "; y = " << y << std::endl;

            // execute kelner
            cl_event kernel_event;

            err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &kernel_event);
            if (err < 0) {
                ERROR("in BigFactors in clEnqueueNDRangeKernel an error occured: err = " << err);
            }

            /* Wait for kernel execution to complete */
            err = clWaitForEvents(1, &kernel_event);
            if (err < 0) {
                ERROR("Couldn't wait for events in BigFactors: err = " << err);
            }


            set_argument(kernel2, 3, sizeof(int), &x);
            set_argument(kernel2, 4, sizeof(int), &y);

            err = clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, global_size_kelner2, local_size_kelner2, 0, NULL, &kernel_event);
            if (err < 0) {
                ERROR("in BigFactors in clEnqueueNDRangeKernel an error occured: err = " << err);
            }

            /* Wait for kernel execution to complete */
            err = clWaitForEvents(1, &kernel_event);
            if (err < 0) {
                ERROR("Couldn't wait for events in kernel2: err = " << err);
            }



            clReleaseEvent(kernel_event);
        }
    }

}