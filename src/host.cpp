#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "CPU_impl.h"
#include "GPU_impl.h"

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
    unsigned char* img_data = stbi_load("img/pioro-large.jpg", &width, &height, &nrChannels, 3);
    if (!img_data)
    {
        std::cout << "Can not read your image file, try to use another one" << std::endl;
        return -1;
    }

    std::chrono::microseconds cpu_duration = computeCPU(xComp, yComp, width, height, img_data, width * 3);
    std::chrono::microseconds gpu_duration = computeGPU(xComp, yComp, width, height, img_data);

    std::cout << "SpeedUP = " << cpu_duration.count() / (float) gpu_duration.count() << std::endl;

    stbi_image_free(img_data);
}

