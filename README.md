# blurhash GPU implementation with OpenCL

This project implements blurhash algorithm on GPU using OpenCL. The original algorithm written in C language was used as a prototype for GPU algorithm (you can find it on the blurhash official github [here](https://github.com/woltapp/blurhash)).

This GPU algorithm works for images of differet sizes.

## Compilation and usage




## GPU algorithm description

In this algorithm was optimized a loop over all image pixels, namely

```C++
static float *multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t *rgb, size_t bytesPerRow) 
{
	float r = 0, g = 0, b = 0;
	float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;

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
```

where we sum `width * height` numbers. 

The first thing we do on GPU is sum numbers in each row. To do this we devide our 2 dimantional image on work-groups (blocks), and sum elements in each work-group as shown on image below. The sum of each work-group elements are written in the most left table cell. All of this do first kelner `factors_rows`

![Sum elements in row](/documentation/sum_rows.svg)

