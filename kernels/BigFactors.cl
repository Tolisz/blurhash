// TODO: Add OpenCL kernel code here.

__kernel void BigFactors(__global unsigned char* BigFactors, __global unsigned char* img, int img_W, int img_H, int xComp, int yComp)
{
	//printf("czesc\n");
	//BigFactors[0] = 'a';
	//printf("%d\n", BigFactors[0]);
	//printf("[%d, %d, %d]", img[0], img[1], img[2]);

	int x = get_global_id(0);
    int y = get_global_id(1);

	//int lx = get_local_id(0);
	//int ly = get_local_id(1);

	//printf("[%d, %d]\n", lx, ly);
	//if (lx == 300 && ly == 193)
	//{
	//}

	//if (x == 7223 && y == 1543)
	//{
	//	printf("[%d, %d]\n", x, y);
	//}
}