#include "encode.h"
#include "common.h"

#include <string.h>

static float *multiplyBasisFunction(int xComponent, int yComponent, int width, int height, unsigned char* rgb, size_t bytesPerRow);
static char *encode_int(int value, int length, char *destination);

static int encodeDC(float r, float g, float b);
static int encodeAC(float r, float g, float b, float maximumValue);

const char *blurHashForPixels(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow) {
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	if(xComponents < 1 || xComponents > 9) return NULL;
	if(yComponents < 1 || yComponents > 9) return NULL;

	float* newFactor = (float*)malloc(sizeof(float) * xComponents * yComponents * 3);
	if (!newFactor)
	{
		printf("[%s, %d] Table allocation error \n", __FILE__, __LINE__);
		exit(-1);
	}

	for(int y = 0; y < yComponents; y++) {
		for(int x = 0; x < xComponents; x++) {
			float *factor = multiplyBasisFunction(x, y, width, height, rgb, bytesPerRow);

			*(newFactor + (y * yComponents + x) * 3 + 0) = factor[0];
			*(newFactor + (y * yComponents + x) * 3 + 1) = factor[1];
			*(newFactor + (y * yComponents + x) * 3 + 2) = factor[2];
		}
	}

	float* dc = newFactor; //factors[0][0];
	float *ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char *ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	if(acCount > 0) {
		float actualMaximumValue = 0;
		for(int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = (int)fmaxf(0, fminf(82, floorf(actualMaximumValue * 166.0f - 0.5f)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	} else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for(int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;

	free(newFactor);
	return buffer;
}

static float *multiplyBasisFunction(int xComponent, int yComponent, int width, int height, unsigned char* rgb, size_t bytesPerRow) {
	float r = 0, g = 0, b = 0;
	float normalisation = (xComponent == 0 && yComponent == 0) ? 1.0f : 2.0f;

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			float basis = cosf(M_PI * xComponent * x / width) * cosf(M_PI * yComponent * y / height);
			r += basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
			g += basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
			b += basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);
		}
	}

	float scale = normalisation / (width * height);

	static float result[3];
	result[0] = r * scale;
	result[1] = g * scale;
	result[2] = b * scale;

	return result;
}



static int encodeDC(float r, float g, float b) {
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

static int encodeAC(float r, float g, float b, float maximumValue) {
	int quantR = (int)fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5f) * 9.0f + 9.5f)));
	int quantG = (int)fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5f) * 9.0f + 9.5f)));
	int quantB = (int)fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5f) * 9.0f + 9.5f)));

	return quantR * 19 * 19 + quantG * 19 + quantB;
}

static char characters[]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

static char *encode_int(int value, int length, char *destination) {
	int divisor = 1;
	for(int i = 0; i < length - 1; i++) divisor *= 83;

	for(int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}
