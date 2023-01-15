# blurhash GPU implementation with OpenCL

This project implements blurhash algorithm on GPU using OpenCL. The original algorithm written in C language was used as a prototype for GPU algorithm (you can find it on the blurhash official github [here](https://github.com/woltapp/blurhash)).

This GPU algorithm works for images of differet sizes.

## Compilation and usage

After compilation you can use a program from terminal:
```
$ program
Usage: xComp yComp image
        xComp - number from 1 to 9
        yComp - number from 1 to 9
        image - relative path to image
        [Optional] --GPUonly -> Computes hash only on GPU
```
The result looks like this:
```  
$ program 8 8 .\img\piesek.jpg

-----------------
 Version [CPU]
-----------------
:MG[vSxu4U9Zjb-;x]Rj0KR*D%t7xuofj[Rj9Fxut7Mx%MofM{j[ofaz-pWBRjt7M{j@M{RjxuxuM{M{j]ayM{Rjof%MM{M{t7j[Ioaxt7t7ayRjt7j[WBoft7RjWBofj[of
Time = 9252205


-----------------
 Version [GPU]
-----------------

----------------------------------------------------------------------------
 [kernels/blurhash.cl] Errors and Warnings
----------------------------------------------------------------------------

----------------------------------------------------------------------------

:MG[vSxu4U9Zjb-;x]Rj0KR*D%t7xuofj[Rj9Fxut7Mx%MofM{j[ofaz-pWBRjt7M{j@M{RjxuxuM{M{j]ayM{Rjof%MM{M{t7j[Ioaxt7t7ayRjt7j[WBoft7RjWBofj[of
Time = 484689

SpeedUP = 19.089

```


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

![Sum elements in rows](/documentation/sum_rows.svg)

When the first kelner ends its computation, we execute a second kelner `factors_column`. At the beginning of this kelner we sum remained elements in each row and write result to the first table column. The last thing we need to do is sum elements in the first column. Elements in each work-groups as summed on GPU, and results of each work-group are summed up on CPU.

![Sum elements in column](/documentation/sum_column.svg)

The remaining part of algorith is implemented on CPU.