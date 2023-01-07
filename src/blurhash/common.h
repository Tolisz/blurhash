#ifndef __BLURHASH_COMMON_H__
#define __BLURHASH_COMMON_H__

#include<math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static inline int linearTosRGB(float value) {
	float v = fmaxf(0, fminf(1, value));
	if(v <= 0.0031308f) return (int)(v * 12.92f * 255.0f + 0.5f);
	else return (int)((1.055f * powf(v, 1.0f / 2.4f) - 0.055f) * 255.0f + 0.5f);
}

static inline float sRGBToLinear(int value) {
	float v = (float)value / 255;
	if(v <= 0.04045) return v / 12.92f;
	else return powf((v + 0.055f) / 1.055f, 2.4f);
}

static inline float signPow(float value, float exp) {
	return copysignf(powf(fabsf(value), exp), value);
}

#endif
