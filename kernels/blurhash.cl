// TODO: Add OpenCL kernel code here.

float sRGBToLinear(int value);

//#define M_PI 3.14159265358979323846

__kernel void factors_rows(__global float* BigFactors, __global unsigned char* img, int img_W, int img_H, int xComponents, int yComponents, int ComponentX, int ComponentY)
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
		float cosValue1 = M_PI_F * (ComponentX * x / (float) img_W);
		float COS1 = cos(cosValue1);

		float cosValue2 =  M_PI_F * (ComponentY * y / (float) img_H);
		float COS2 = cos(cosValue2);

		float basis = COS1 * COS2;

		//float basis = cos( M_PI_F * ((float)ComponentX * (float)x) / (float) img_W) * cos( M_PI_F * ((float)ComponentY * (float)y) / (float) img_H);

		BigFactors[3 * (img_W * y + x) + 0] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 0]);
		BigFactors[3 * (img_W * y + x) + 1] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 1]);
		BigFactors[3 * (img_W * y + x) + 2] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 2]);
	}
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

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
}

float sRGBToLinear(int value) 
{
	float v = (float)value / 255.0f;
	if(v <= 0.04045f) return v / 12.92f;
	else return pow((v + 0.055f) / 1.055f, 2.4f);
}

__kernel void factors_column(__global float* BigFactors, int img_W, int img_H, int ComponentX, int ComponentY)
{
    int y = get_global_id(1);

	int size = get_local_size(1);
	int n = y / size;

	bool shouldWork = true;
	if (y >= img_H)
	{
		 shouldWork = false;
	}

	if (shouldWork)
	{
		int left = img_W / size;
		for (int i = 0; i < left; i++)
		{
			BigFactors[3 * (y * img_W) + 0] += BigFactors[3 * (y * img_W + (i + 1) * size) + 0];
			BigFactors[3 * (y * img_W) + 1] += BigFactors[3 * (y * img_W + (i + 1) * size) + 1];
			BigFactors[3 * (y * img_W) + 2] += BigFactors[3 * (y * img_W + (i + 1) * size) + 2];
		}
	}

	for (int d = 1; d < size; d *= 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
		if (shouldWork)
		{
			if ( y % (2 * d) != 0 || y + d >= img_H || y + d >= (n+1) * size)
			{
				continue;
			}
	
			BigFactors[3 * (y * img_W) + 0] += BigFactors[3 * ((y + d) * img_W) + 0];
			BigFactors[3 * (y * img_W) + 1] += BigFactors[3 * ((y + d) * img_W) + 1];
			BigFactors[3 * (y * img_W) + 2] += BigFactors[3 * ((y + d) * img_W) + 2];
		}
	}

	if (y % size == 0)
	{
		BigFactors[3 * (img_H * img_W + n) + 0] = BigFactors[3 * (y * img_W) + 0];
		BigFactors[3 * (img_H * img_W + n) + 1] = BigFactors[3 * (y * img_W) + 1];
		BigFactors[3 * (img_H * img_W + n) + 2] = BigFactors[3 * (y * img_W) + 2];
	}
}