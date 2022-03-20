#include "blowfish.h"
#include "const.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

void version(void)
{
	printf("By Darshan D V(16CO216) and (16CO232) and Piyush Hanchate | Sequential version\n");
}

void blowfish_encryptptr(blowfish_context_t *context, uint32_t *ptr, size_t numblocks, double *runtime, double *rate)
{
	size_t pos = 0;
	clock_t start, end;
	double local_runtime;

	start = clock();
	for (pos = 0; pos < numblocks; pos+=2)
	{
		blowfish_encryptblock(context, ptr+pos, ptr+pos+1);
	}
	end = clock();

	local_runtime = ((double) (end - start)) /(double)CLOCKS_PER_SEC;

	if (runtime)
		*runtime = local_runtime;

	if (rate)
		*rate = ((double)numblocks*sizeof(uint32_t))/(local_runtime);
}

void blowfish_decryptptr(blowfish_context_t *context, uint32_t *ptr, size_t numblocks, double *runtime, double *rate)
{
	size_t pos = 0;
	clock_t start, end;
	double local_runtime;
	start = clock();
	for (pos = 0; pos < numblocks; pos+=2)
	{
		blowfish_decryptblock(context, ptr+pos, ptr+pos+1);
	}
	end = clock();

	local_runtime = ((double) (end - start)) /(double)CLOCKS_PER_SEC;

	if (runtime)
		*runtime = local_runtime;

	if (rate)
		*rate = ((double)numblocks*sizeof(uint32_t))/(local_runtime);
}
