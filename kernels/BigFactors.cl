// TODO: Add OpenCL kernel code here.

float sRGBToLinear(int value);

//#define M_PI 3.14159265358979323846

__kernel void BigFactors(__global float* BigFactors, __global unsigned char* img, int img_W, int img_H, int xComponents, int yComponents)
{
	//printf("czesc\n");
	//BigFactors[0] = 'a';
	//printf("%d\n", BigFactors[0]);
	//printf("[%d, %d, %d]", img[0], img[1], img[2]);

	int x = get_global_id(0);
    int y = get_global_id(1);

	int ComponentX = x / img_W;
	int ComponentY = y / img_H;

	int ImageX = x % img_W;
	int ImageY = y % img_H;

	float normalisation = (ComponentX == 0 && ComponentY == 0) ? 1.0f : 2.0f;
	float basis = cos( (M_PI_F * ComponentX * ImageX) / (float) img_W) * cos( (M_PI_F * ComponentY * ImageY) / (float) img_H);

	BigFactors[3 * img_W * xComponents * y + 3 * x + 0] = basis * sRGBToLinear((int)img[3 * (ImageY * img_W + ImageX) + 0]);
	BigFactors[3 * img_W * xComponents * y + 3 * x + 1] = basis * sRGBToLinear((int)img[3 * (ImageY * img_W + ImageX) + 1]);
	BigFactors[3 * img_W * xComponents * y + 3 * x + 2] = basis * sRGBToLinear((int)img[3 * (ImageY * img_W + ImageX) + 2]);


	if (ComponentX == 4 && ComponentY == 3 && ImageX == 185 && ImageY == 1)
	{
		printf("Component [X, Y] [%d, %d]\n", ComponentX, ComponentY);
		printf("Image [X, Y] [%d, %d]\n", ImageX, ImageY);
		printf("basis = %f\n", basis);
		printf("BigFactors = [%f, %f, %f]\n", 
						BigFactors[3 * img_W * xComponents * y + 3 * x + 0], 
						BigFactors[3 * img_W * xComponents * y + 3 * x + 1], 
						BigFactors[3 * img_W * xComponents * y + 3 * x + 2]);
	}


}

float sRGBToLinear(int value) 
{
	float v = (float)value / 255.0f;
	if(v <= 0.04045f) return v / 12.92f;
	else return pow((v + 0.055f) / 1.055f, 2.4f);
}