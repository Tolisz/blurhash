#include "GPU_impl.h"

#include "error.h"
#include "utils.h"

#include <iostream>
#include <cmath>


void computeGPU(int xComponents, int yComponents, int width, int height, unsigned char* img_data)
{
    std::cout << "Version [CPU] \n-----------------\n\n";

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
    float* factors = (float*)malloc(sizeof(float) * xComponents * yComponents * 3);
    if (!factors)
    {
        printf("[%s, %d] Table allocation error \n", __FILE__, __LINE__);
        exit(-1);
    }
    memset(factors, 0, sizeof(float) * xComponents * yComponents * 3);

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


    for (int y = 0; y < yComponents; y++)
    {
        for (int x = 0; x < xComponents; x++)
        {
            set_argument(kernel, 6, sizeof(int), &x);
            set_argument(kernel, 7, sizeof(int), &y);

            //std::cout << "x = " << x << "; y = " << y << std::endl;

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

            err = clEnqueueReadBuffer(queue, cl_BigFactors, CL_TRUE, 0, sizeBigFactors * sizeof(float), BigFactors, 0, NULL, NULL);
            if (err < 0)
            {
                ERROR("Couldn't read from cl_BigFactors");
            }

            //std::cout << BigFactors[0] << std::endl;
            
            // MUSIMY TUTAJ JESZCZE ZSUMOWAĆ DLA ZDJĘĆ O WYSOKOŚCI > maxWorkGroupSize

            float normalisation = (x == 0 && y == 0) ? 1 : 2;
            float scale = normalisation / (img_W * img_H);

            *(factors + (y * yComponents + x) * 3 + 0) = scale * BigFactors[0];
            *(factors + (y * yComponents + x) * 3 + 1) = scale * BigFactors[1];
            *(factors + (y * yComponents + x) * 3 + 2) = scale * BigFactors[2];
        }
    }


    //for (int y = 0; y < yComponents; y++)
    //{
    //    std::cout << "y = " << y << std::endl;

    //    for (int x = 0; x < xComponents; x++)
    //    {
    //        std::cout << "[" << factors[3 * (y * yComponents + x) + 0] << ", ";
    //        std::cout << factors[3 * (y * yComponents + x) + 1] << ", ";
    //        std::cout << factors[3 * (y * yComponents + x) + 2] << "] \n";
    //    }

    //    std::cout << "\n";
    //}

    char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

    char characters[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

    // lambdy
    auto encode_int = [&characters](int value, int length, char* destination) -> char*
    {
        int divisor = 1;
        for (int i = 0; i < length - 1; i++) divisor *= 83;

        for (int i = 0; i < length; i++) {
            int digit = (value / divisor) % 83;
            divisor /= 83;
            *destination++ = characters[digit];
        }
        return destination;
    };

    auto linearTosRGB = [](float value) -> int
    {
        float v = fmaxf(0, fminf(1, value));
        if (v <= 0.0031308f) return (int)(v * 12.92f * 255.0f + 0.5f);
        else return (int)((1.055f * powf(v, 1.0f / 2.4f) - 0.055f) * 255.0f + 0.5f);
    };

    auto encodeDC = [&linearTosRGB](float r, float g, float b) -> int
    {
        int roundedR = linearTosRGB(r);
        int roundedG = linearTosRGB(g);
        int roundedB = linearTosRGB(b);
        return (roundedR << 16) + (roundedG << 8) + roundedB;
    };

    auto signPow = [](float value, float exp) -> float 
    {
        return copysignf(powf(fabsf(value), exp), value);
    };


    auto encodeAC = [&signPow](float r, float g, float b, float maximumValue) -> int
    {
        int quantR = (int)fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5f) * 9.0f + 9.5f)));
        int quantG = (int)fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5f) * 9.0f + 9.5f)));
        int quantB = (int)fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5f) * 9.0f + 9.5f)));

        return quantR * 19 * 19 + quantG * 19 + quantB;
    };

    // obliczenia

    float* dc = factors;
    float* ac = dc + 3;
    int acCount = xComponents * yComponents - 1;
    char* ptr = buffer;

    int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
    ptr = encode_int(sizeFlag, 1, ptr);

    float maximumValue;
    if (acCount > 0) {
        float actualMaximumValue = 0;
        for (int i = 0; i < acCount * 3; i++) {
            actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
        }

        int quantisedMaximumValue = (int)fmaxf(0, fminf(82, floorf(actualMaximumValue * 166.0f - 0.5f)));
        maximumValue = ((float)quantisedMaximumValue + 1) / 166;
        ptr = encode_int(quantisedMaximumValue, 1, ptr);
    }
    else {
        maximumValue = 1;
        ptr = encode_int(0, 1, ptr);
    }

    ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

    for (int i = 0; i < acCount; i++) {
        ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
    }

    *ptr = 0;

    std::cout << "hash = " << buffer << std::endl;

    free(factors);
}