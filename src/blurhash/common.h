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

static int encodeDC(float r, float g, float b)
{
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

static char characters[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

static char* encode_int(int value, int length, char* destination) {
	int divisor = 1;
	for (int i = 0; i < length - 1; i++) divisor *= 83;

	for (int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}

#endif
