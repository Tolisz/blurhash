// TODO: Add OpenCL kernel code here.


float sRGBToLinear(int value);

//#define M_PI 3.14159265358979323846

__kernel void factors_rows(__global float* R, __global float* G, __global float* B, __global unsigned char* img, int img_W, int img_H, int xComponents, int yComponents, int ComponentX, int ComponentY)
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
		float basis = cos( M_PI_F * ((float)ComponentX * (float)x) / (float) img_W) * cos( M_PI_F * ((float)ComponentY * (float)y) / (float) img_H);

		R[img_W * y + x] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 0]);
		G[img_W * y + x] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 1]);
		B[img_W * y + x] = basis * sRGBToLinear((int)img[3 * (img_W * y + x) + 2]);
	}
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// Compute sum in each row in each work-group (block)
	// --------------------------------------------------

	for (int d = 1; d < size; d *= 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
		if (shouldWork)
		{
			if ( x % (2 * d) != 0 || x + d >= img_W || x + d >= (n + 1) * size)
			{
				continue;
			}

			R[y * img_W  + x] += R[y * img_W  + x + d];
			G[y * img_W  + x] += G[y * img_W  + x + d];
			B[y * img_W  + x] += B[y * img_W  + x + d];
		}
	}

}

float sRGBToLinear(int value) 
{
	float v = (float)value / 255.0f;
	if(v <= 0.04045f) return v / 12.92f;
	else return pow((v + 0.055f) / 1.055f, 2.4f);
}

__kernel void factors_column(__global float* R, __global float* G, __global float* B, __global float* result, int img_W, int img_H, int ComponentX, int ComponentY)
{
    int y = get_global_id(1);
	
	int size = get_local_size(1);
	int n = y / size;
	
	bool shouldWork = true;
	if (y >= img_H)
	{
		 shouldWork = false;
	}
	
	// Compute sum of the remaining elements in each row
	// and write it to the first column
	// --------------------------------------------------

	int left = ceil(img_W / (float) size);
	for (int i = 1; i < left; i++)
	{
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (shouldWork)
		{
			R[y * img_W] += R[y * img_W + i * size];
			G[y * img_W] += G[y * img_W + i * size];
			B[y * img_W] += B[y * img_W + i * size];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// Compute sum in first column in each work-group (block)
	// -------------------------------------------------------

	for (int d = 1; d < size; d *= 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
		if (shouldWork)
		{
			if ( y % (2 * d) != 0 || y + d >= img_H || y + d >= (n+1) * size)
			{
				continue;
			}
	
			R[y * img_W] += R[(y + d) * img_W];
			G[y * img_W] += G[(y + d) * img_W];
			B[y * img_W] += B[(y + d) * img_W];
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// Write result of sum in each work-group to the 
	// additional table
	// -------------------------------------------------------

	if (shouldWork)
	{
		if (y % size == 0)
		{
			result[3 * n + 0] = R[y * img_W];
			result[3 * n + 1] = G[y * img_W];
			result[3 * n + 2] = B[y * img_W];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}