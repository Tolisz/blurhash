#ifndef __BLURHASH_ENCODE_H__
#define __BLURHASH_ENCODE_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

const char *blurHashForPixels(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow);

#endif
