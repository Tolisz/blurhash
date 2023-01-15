#include "GPU_impl.h"

#include "error.h"
#include "utils.h"
#include "blurhash/common.h"

#include <iostream>
#include <cmath>

using namespace std::chrono;

//#define DEBUG_TABLE


microseconds computeGPU(int xComponents, int yComponents, int width, int height, unsigned char* img_data)
{
    std::cout << "\n-----------------\n Version [GPU] \n-----------------\n\n";

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
    
    const char* hash = BigFactors(device, context, queue, width, height, img_data, xComponents, yComponents);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << hash;
    std::cout << "\nTime = " << duration.count() << "\n\n";

    // Sprzątanie
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);

    return duration;
}

const char* BigFactors(cl_device_id& device, cl_context& context, cl_command_queue& queue,
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
    //std::cout << "Maximal work-group size = " << max_work_group_size << std::endl;


    // Tworzenie tablicy do przechowywania wyników pośrednich
    // ------------------------------------------------------

    size_t left_to_sum = (size_t)std::ceilf(img_H / (float)max_work_group_size);
    size_t table_size = ((size_t)img_W) * ((size_t)img_H);
    
    float* factor_result = new float[left_to_sum * 3];
    if (!factor_result)
    {
        ERROR("factor_result table allocation was unsuccessful");
    }

    // Alokacja pamięci na GPU
    // -----------------------

    //cl_mem cl_BigFactors = create_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * table_size, NULL);
    cl_mem cl_R = create_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * table_size, NULL);
    cl_mem cl_G = create_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * table_size, NULL);
    cl_mem cl_B = create_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * table_size, NULL);

    cl_mem cl_factor_result = create_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * left_to_sum * 3, NULL);
    cl_mem cl_img = create_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * img_W * img_H * 3, img);

    // Ustawienie parametrów kerneli
    // -----------------------------

    set_argument(kernel_rows, 0, sizeof(cl_mem), &cl_R);
    set_argument(kernel_rows, 1, sizeof(cl_mem), &cl_G);
    set_argument(kernel_rows, 2, sizeof(cl_mem), &cl_B);
    set_argument(kernel_rows, 3, sizeof(cl_mem), &cl_img);
    set_argument(kernel_rows, 4, sizeof(int), &img_W);
    set_argument(kernel_rows, 5, sizeof(int), &img_H);
    set_argument(kernel_rows, 6, sizeof(int), &xComponents);
    set_argument(kernel_rows, 7, sizeof(int), &yComponents);

    set_argument(kernel_column, 0, sizeof(cl_mem), &cl_R);
    set_argument(kernel_column, 1, sizeof(cl_mem), &cl_G);
    set_argument(kernel_column, 2, sizeof(cl_mem), &cl_B);
    set_argument(kernel_column, 3, sizeof(cl_mem), &cl_factor_result);
    set_argument(kernel_column, 4, sizeof(int), &img_W);
    set_argument(kernel_column, 5, sizeof(int), &img_H);


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
            cl_event kernel_event;

            // Uruchamiamy pierwszy kernel, w którym liczymy sumy w poszczególnych rzędach
            // ----------------------------------------------------------------------------

            set_argument(kernel_rows, 8, sizeof(int), &x);
            set_argument(kernel_rows, 9, sizeof(int), &y);

            err = clEnqueueNDRangeKernel(queue, kernel_rows, 2, NULL, rows_global_size, rows_local_size, 0, NULL, &kernel_event);
            if (err < 0) 
            {
                ERROR("in kernel_rows in clEnqueueNDRangeKernel an error occured: err = " << err << "\n");
            }

            err = clWaitForEvents(1, &kernel_event);
            if (err < 0) 
            {
                ERROR("Couldn't wait for events in BigFactors: err = " << err << "\n");
            }


            // Uruchamiamy drugi kernel, w którym liczymy sumę w pierwszej kolumnie
            // ----------------------------------------------------------------------------

            set_argument(kernel_column, 6, sizeof(int), &x);
            set_argument(kernel_column, 7, sizeof(int), &y);

            err = clEnqueueNDRangeKernel(queue, kernel_column, 2, NULL, column_global_size, column_local_size, 0, NULL, &kernel_event);
            if (err < 0) 
            {
                ERROR("in BigFactors in clEnqueueNDRangeKernel an error occured: err = " << err);
            }

            err = clWaitForEvents(1, &kernel_event);
            if (err < 0) 
            {
                ERROR("Couldn't wait for events in kernel_column: err = " << err);
            }
            

            // Czytamy informację z bufora na GPU
            // ----------------------------------------------------------------------------

            err = clEnqueueReadBuffer(queue, cl_factor_result, CL_TRUE, 0, left_to_sum * 3 * sizeof(float), factor_result, 0, NULL, &kernel_event);
            if (err < 0)
            {
                ERROR("Couldn't read from cl_BigFactors");
            }

            err = clWaitForEvents(1, &kernel_event);
            if (err < 0) 
            {
                ERROR("Couldn't wait for events in BigFactors: err = " << err);
            }

            clReleaseEvent(kernel_event);
            
            float r = 0;
            float g = 0;
            float b = 0;
            
            for (size_t i = 0; i < left_to_sum; i++)
            {
                r += factor_result[i * 3 + 0];
                g += factor_result[i * 3 + 1];
                b += factor_result[i * 3 + 2];
            }

            float normalisation = (x == 0 && y == 0) ? 1.0f : 2.0f;
            float scale = normalisation / (img_W * img_H);

            *(factors + (y * xComponents + x) * 3 + 0) = scale * r;
            *(factors + (y * xComponents + x) * 3 + 1) = scale * g;
            *(factors + (y * xComponents + x) * 3 + 2) = scale * b;
        }
    }


#ifdef DEBUG_TABLE
    for (int y = 0; y < yComponents; y++)
    {
        std::cout << "y = " << y << std::endl;

        for (int x = 0; x < xComponents; x++)
        {
            printf("[%.20f, %.20f, %.20f]\n", factors[3 * (y * yComponents + x) + 0], factors[3 * (y * yComponents + x) + 1], factors[3 * (y * yComponents + x) + 2]);
        }

        std::cout << "\n";
    }
#endif

    static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];
    
    // Obliczenie hashu na CPU
    // -----------------------
    
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
    
    
    // Sprzątanie 
    // ----------
    
    free(factors);
    delete[] factor_result;
    
    clReleaseProgram(program);
    clReleaseKernel(kernel_column);
    clReleaseKernel(kernel_rows);
    clReleaseMemObject(cl_factor_result);
    clReleaseMemObject(cl_img);

    return buffer;
}