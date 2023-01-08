// TODO: Add OpenCL kernel code here.

float sRGBToLinear(int value);

//#define M_PI 3.14159265358979323846

__kernel void BigFactors(__global float* BigFactors, __global unsigned char* img, int img_W, int img_H, int xComponents, int yComponents, int ComponentX, int ComponentY)
{
	int x = get_global_id(0);
    int y = get_global_id(1);

	int size = get_local_size(0);
	int n = x / size;

	bool shouldWork = true;

	if (x >= img_W || y >= img_H)
	{
		 shouldWork = false;
	}
	
	if (shouldWork) 
	{
		float basis = cos( (M_PI_F * ComponentX * x) / (float) img_W) * cos( (M_PI_F * ComponentY * y) / (float) img_H);
	
		BigFactors[3 * (img_W * y + x) + 0] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 0]);
		BigFactors[3 * (img_W * y + x) + 1] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 1]);
		BigFactors[3 * (img_W * y + x) + 2] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 2]);
	}
	
	
	//printf("czekam przy barierze \n");
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	//printf("Jestem po barierze\n");

	//if (ComponentX == 4 && ComponentY == 3 && x == 185 && y == 1)
	//{
	//	printf("[x, y] = [%d, %d]\n", x, y);
	//	printf("Component [X, Y] [%d, %d]\n", ComponentX, ComponentY);
	//	printf("Image [X, Y] [%d, %d]\n", x, y);
	//	printf("basis = %f\n", basis);
	//	printf("BigFactors = [%f, %f, %f]\n", 
	//					BigFactors[3 * img_W * y + 3 * x + 0], 
	//					BigFactors[3 * img_W * y + 3 * x + 1], 
	//					BigFactors[3 * img_W * y + 3 * x + 2]);
	//	//printf("BigFactors = [%f, %f, %f]\n", 
	//	//				basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 0]), 
	//	//				basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 1]), 
	//	//				basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 2]));
	//
	//	printf("n = %d\n", n);
	//}
	//

	for (int d = 1; d < size; d *= 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
		if (shouldWork)
		{
			if ( x % (2 * d) != 0 || x + d >= img_W || x + d >= (n + 1) * size)
			{
				continue;
			}

			BigFactors[3 * (y * img_W  + x) + 0] += BigFactors[3 * (y * img_W  + x + d) + 0];
			BigFactors[3 * (y * img_W  + x) + 1] += BigFactors[3 * (y * img_W  + x + d) + 1];
			BigFactors[3 * (y * img_W  + x) + 2] += BigFactors[3 * (y * img_W  + x + d) + 2];
		}
	}
	
	//barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//if (x == 0 && shouldWork)
	//{
	//	int left = img_W / size;
	//
	//	for (int i = 0; i < left; i++)
	//	{
	//		BigFactors[3 * (y * img_W) + 0] += BigFactors[3 * (y * img_W + (i + 1) * size) + 0];
	//		BigFactors[3 * (y * img_W) + 1] += BigFactors[3 * (y * img_W + (i + 1) * size) + 1];
	//		BigFactors[3 * (y * img_W) + 2] += BigFactors[3 * (y * img_W + (i + 1) * size) + 2];
	//	}
	//}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if (x != 0)
	{
		return;
	}

	printf("%.10f\n", BigFactors[3 * (y * img_W + x) + 0]);
}

float sRGBToLinear(int value) 
{
	float v = (float)value / 255.0f;
	if(v <= 0.04045f) return v / 12.92f;
	else return pow((v + 0.055f) / 1.055f, 2.4f);
}


__kernel void sumVertically(__global float* BigFactors, int img_W, int img_H, int ComponentX, int ComponentY)
{
	int x = get_global_id(0);
    int y = get_global_id(1);

	int size = get_local_size(1);

	float normalisation = (ComponentX == 0 && ComponentY == 0) ? 1.0f : 2.0f;
	float scale = normalisation / (float)(img_W * img_H);

	if (y == 0)
	{
		printf("BigFactors[0][0] = %f\n", BigFactors[3 * (y * img_W + x) + 0] );
	}

	for (int d = 1; d < size; d *= 2)
	{
		mem_fence(CLK_GLOBAL_MEM_FENCE);

		if ( y % (2 * d) != 0 || y + d >= img_H)
		{
			if (y == 0)
			{
				printf("Oj kurwa niedobrze\n");
			}
			return;
		}

		BigFactors[3 * (y * img_W + x) + 0] += BigFactors[3 * ((y + d) * img_W + x) + 0];
		BigFactors[3 * (y * img_W + x) + 1] += BigFactors[3 * ((y + d) * img_W + x) + 1];
		BigFactors[3 * (y * img_W + x) + 2] += BigFactors[3 * ((y + d) * img_W + x) + 2];
	}

	if (y == 0)
	{
		int left = img_H / size;

		for (int i = 0; i < left; i++)
		{
			BigFactors[3 * (y * img_W + x) + 0] += BigFactors[3 * ((y + (i+1) * size) * img_W + x) + 0];
			BigFactors[3 * (y * img_W + x) + 1] += BigFactors[3 * ((y + (i+1) * size) * img_W + x) + 1];
			BigFactors[3 * (y * img_W + x) + 2] += BigFactors[3 * ((y + (i+1) * size) * img_W + x) + 2];
		}
	}

	if (y == 0)
	{
		printf("Koniec BigFactors[0][0] = %f\n", scale * BigFactors[3 * (y * img_W + x) + 1] );
	}
}