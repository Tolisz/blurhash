#include "GPU_impl.h"

#include "error.h"
#include "utils.h"
#include "blurhash/common.h"

#include <iostream>
#include <cmath>

#include <chrono>
using namespace std::chrono;



void computeGPU(int xComponents, int yComponents, int width, int height, unsigned char* img_data)
{
    std::cout << "\n-----------------\n\n Version [GPU] \n-----------------\n\n";

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

    auto start = high_resolution_clock::now();
    
    BigFactors(device, context, queue, width, height, img_data, xComponents, yComponents);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "\nTime = " << duration.count() << "\n\n";
}

void BigFactors(cl_device_id& device, cl_context& context, cl_command_queue& queue,
    int img_W, int img_H, unsigned char* img, int xComponents, int yComponents)
{
    // Tablica factors do przychowywania wyników działania kelnerów
    // ------------------------------------------------------------

    float* factors = (float*)malloc(sizeof(float) * xComponents * yComponents * 3);
    if (!factors)
    {
        printf("[%s, %d] Table allocation error \n", __FILE__, __LINE__);
        exit(-1);
    }
    memset(factors, 0, sizeof(float) * xComponents * yComponents * 3);


    // Tworzenie programu oraz kerneli
    // -------------------------------

    cl_int err;
    cl_program program = create_cl_program(device, context, "kernels/blurhash.cl");

    cl_kernel kernel_rows = create_cl_kelner(program, "factors_rows");
    cl_kernel kernel_column = create_cl_kelner(program, "factors_column");


    // Maksymalna liczba wątków które mogą być w jednym work-group (CUDA: w jednym blocku)
    // na danym urządzeniu
    // -----------------------------------------------------------------------------------

    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    std::cout << "Maximal work-group size = " << max_work_group_size << std::endl;

    // Tworzenie tablicy do przechowywania wyników pośrednich
    // ------------------------------------------------------

    unsigned int sizeBigFactors = img_W * img_H * 3;
    float* BigFactors = new float[sizeBigFactors];
    if (!BigFactors) 
    {
        ERROR("BigFactors table allocation was unsuccessful");
    }


    // Alokacja pamięci na GPU
    // -----------------------

    cl_mem cl_BigFactors = create_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * sizeBigFactors, NULL);
    cl_mem cl_img = create_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * img_W * img_H * 3, img);

    // Ustawienie parametrów kerneli
    // -----------------------------

    set_argument(kernel_rows, 0, sizeof(cl_mem), &cl_BigFactors);
    set_argument(kernel_rows, 1, sizeof(cl_mem), &cl_img);
    set_argument(kernel_rows, 2, sizeof(int), &img_W);
    set_argument(kernel_rows, 3, sizeof(int), &img_H);
    set_argument(kernel_rows, 4, sizeof(int), &xComponents);
    set_argument(kernel_rows, 5, sizeof(int), &yComponents);

    set_argument(kernel_column, 0, sizeof(cl_mem), &cl_BigFactors);
    set_argument(kernel_column, 1, sizeof(int), &img_W);
    set_argument(kernel_column, 2, sizeof(int), &img_H);


    // Ustawienie liczby wątków oraz rozmiaru pojedynczego work-group (CUDA: rozmiaru bloku)
    // -------------------------------------------------------------------------------------

    size_t rows_global_size[2];
    rows_global_size[0] = img_W + (max_work_group_size - (img_W) % max_work_group_size);
    rows_global_size[1] = img_H;
    
    size_t rows_local_size[2];
    rows_local_size[0] = max_work_group_size;
    rows_local_size[1] = 1;


    size_t column_global_size[2];
    column_global_size[0] = 1;
    column_global_size[1] = img_H + (max_work_group_size - (img_H) % max_work_group_size);

    size_t column_local_size[2];
    column_local_size[0] = 1;
    column_local_size[1] = max_work_group_size;


    // Pętla po komponentach 
    // ---------------------

    for (int y = 0; y < yComponents; y++)
    {
        for (int x = 0; x < xComponents; x++)
        {
            set_argument(kernel_rows, 6, sizeof(int), &x);
            set_argument(kernel_rows, 7, sizeof(int), &y);

            cl_event kernel_event;

            err = clEnqueueNDRangeKernel(queue, kernel_rows, 2, NULL, rows_global_size, rows_local_size, 0, NULL, &kernel_event);
            if (err < 0) {
                ERROR("in BigFactors in clEnqueueNDRangeKernel an error occured: err = " << err);
            }

            /* Wait for kernel execution to complete */
            err = clWaitForEvents(1, &kernel_event);
            if (err < 0) {
                ERROR("Couldn't wait for events in BigFactors: err = " << err);
            }


            set_argument(kernel_column, 3, sizeof(int), &x);
            set_argument(kernel_column, 4, sizeof(int), &y);

            err = clEnqueueNDRangeKernel(queue, kernel_column, 2, NULL, column_global_size, column_local_size, 0, NULL, &kernel_event);
            if (err < 0) {
                ERROR("in BigFactors in clEnqueueNDRangeKernel an error occured: err = " << err);
            }

            /* Wait for kernel execution to complete */
            err = clWaitForEvents(1, &kernel_event);
            if (err < 0) {
                ERROR("Couldn't wait for events in kernel_column: err = " << err);
            }
            clReleaseEvent(kernel_event);

            err = clEnqueueReadBuffer(queue, cl_BigFactors, CL_TRUE, 0, 3 * sizeof(float), BigFactors, 0, NULL, NULL);
            if (err < 0)
            {
                ERROR("Couldn't read from cl_BigFactors");
            }

            //std::cout << BigFactors[0] << std::endl;
            
            // MUSIMY TUTAJ JESZCZE ZSUMOWAĆ DLA ZDJĘĆ O WYSOKOŚCI > maxWorkGroupSize

            float normalisation = (x == 0 && y == 0) ? 1.0f : 2.0f;

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

    printf("dc[0, 1, 2] = [%.10f, %.10f, %.10f]\n", dc[0], dc[1], dc[2]);

    ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

    for (int i = 0; i < acCount; i++) {
        ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
    }

    *ptr = 0;


    // Sprzątanie 
    // ----------

    free(factors);
   

    std::cout << buffer << std::endl;
}