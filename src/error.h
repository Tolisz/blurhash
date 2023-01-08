#pragma once

#include <cstdlib>
#include <iostream>

#define ERROR(message)                                                          \
    do                                                                          \
    {                                                                           \
        std::cout << "[" << __FILE__  << "," << __LINE__ << "] "<< message;     \
        std::exit(-1);                                                          \
    } while (0)                                                 