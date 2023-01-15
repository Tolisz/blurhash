#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "CPU_impl.h"
#include "GPU_impl.h"

int main(int argc, char** argv)
{

    if (argc < 4)
    {
        std::cout << "Usage: xComp yComp image\n";
        std::cout << "\txComp - number from 1 to 9\n";
        std::cout << "\tyComp - number from 1 to 9\n";
        std::cout << "\timage - relative path to image\n";
        std::cout << "\t[Optional] --GPUonly -> Computes hash only on GPU\n";
    
        return -1;
    }

    // blurhash components number
    int xComp = std::atoi(argv[1]);
    int yComp = std::atoi(argv[2]);

    char* image = argv[3];

    bool GPUonly = false;
    if (argc == 5 && std::strcmp(argv[4], "--GPUonly") == 0)
    {   
        GPUonly = true;
    }

    if (xComp < 1 || xComp > 9 || yComp < 1 || yComp > 9)
    {
        std::cout << "Number of components must be between 1 and 8" << std::endl;
        return -1;
    }

    // image loading
    int width, height, nrChannels;
    unsigned char* img_data = stbi_load(image, &width, &height, &nrChannels, 3);
    if (!img_data)
    {
        std::cout << "Can not read your image file, try to use another one" << std::endl;
        return -1;
    }

    std::chrono::microseconds cpu_duration;
    std::chrono::microseconds gpu_duration;

    if (!GPUonly) cpu_duration = computeCPU(xComp, yComp, width, height, img_data, width * 3);
    gpu_duration = computeGPU(xComp, yComp, width, height, img_data);

    if (!GPUonly) std::cout << "SpeedUP = " << cpu_duration.count() / (float) gpu_duration.count() << std::endl;

    stbi_image_free(img_data);
}