#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MEGABYTE 1<<20
#define GIGABYTE 1<<30

int main(int argc, char const *argv[])
{
	if (argc != 2)
	{
		printf("Usage: filegen <size in MB>\n");
		exit(-1);
	}
	size_t i = 0, filesize = atoi(argv[1]) * MEGABYTE;
	printf("Generating file of size %zu bytes\n", filesize);
	FILE* output = fopen("../testfile", "w");
	srand(1632);
	for (i = 0; i < filesize; ++i)
		fputc(rand() & 0xFF, output);
	return 0;
}
