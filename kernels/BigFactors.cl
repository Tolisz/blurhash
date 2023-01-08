// TODO: Add OpenCL kernel code here.

float sRGBToLinear(int value);

//#define M_PI 3.14159265358979323846

__kernel void BigFactors(__global float* BigFactors, __global unsigned char* img, int img_W, int img_H, int xComponents, int yComponents, int ComponentX, int ComponentY)
{
	int x = get_global_id(0);
    int y = get_global_id(1);

	int size = get_local_size(0);

	if (x >= img_W || y >= img_H)
	{
		return;
	}

	float normalisation = (ComponentX == 0 && ComponentY == 0) ? 1.0f : 2.0f;
	float basis = cos( (M_PI_F * ComponentX * x) / (float) img_W) * cos( (M_PI_F * ComponentY * y) / (float) img_H);

	BigFactors[3 * img_W * y + 3 * x + 0] = basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 0]);
	BigFactors[3 * img_W * y + 3 * x + 1] = basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 1]);
	BigFactors[3 * img_W * y + 3 * x + 2] = basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 2]);

	if (ComponentX == 4 && ComponentY == 3 && x == 185 && y == 1)
	{
		printf("[x, y] = [%d, %d]\n", x, y);
		printf("Component [X, Y] [%d, %d]\n", ComponentX, ComponentY);
		printf("Image [X, Y] [%d, %d]\n", x, y);
		printf("basis = %f\n", basis);
		printf("BigFactors = [%f, %f, %f]\n", 
						BigFactors[3 * img_W * y + 3 * x + 0], 
						BigFactors[3 * img_W * y + 3 * x + 1], 
						BigFactors[3 * img_W * y + 3 * x + 2]);
		//printf("BigFactors = [%f, %f, %f]\n", 
		//				basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 0]), 
		//				basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 1]), 
		//				basis * sRGBToLinear((int)img[3 * (y * img_W + x) + 2]));
	}

	mem_fence(CLK_GLOBAL_MEM_FENCE);


}

float sRGBToLinear(int value) 
{
	float v = (float)value / 255.0f;
	if(v <= 0.04045f) return v / 12.92f;
	else return pow((v + 0.055f) / 1.055f, 2.4f);
}