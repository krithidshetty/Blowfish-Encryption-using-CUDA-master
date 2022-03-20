#ifndef BLOWFISH_H_
#define BLOWFISH_H_

#include "const.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

typedef struct blowfish_context_t_ {
	uint32_t pbox[256];
	uint32_t sbox[4][256];
} blowfish_context_t;

void blowfish_encryptblock 	(blowfish_context_t *ctx, uint32_t  *hi,  uint32_t *lo);
void blowfish_decryptblock 	(blowfish_context_t *ctx, uint32_t  *hi,  uint32_t *lo);
int  blowfish_init     		(blowfish_context_t *ctx, char *key, size_t keybytes);
void blowfish_clean        	(blowfish_context_t *ctx);
void* readfile				(size_t *size, char* filepath);
void splashscreen			(void);
uint64_t hash				(void* mem, size_t numblocks);

// Blowfish Macro
#define BLOWFISH_F(x) \
	(((ctx->sbox[0][x >> 24] + ctx->sbox[1][(x >> 16) & 0xFF]) \
	^ ctx->sbox[2][(x >> 8) & 0xFF]) + ctx->sbox[3][x & 0xFF])

uint64_t hash (void* mem, size_t numblocks)
{
	uint32_t* x = (uint32_t*) mem;
	uint32_t hi = 0x1, lo = 0x0;
	uint64_t ret = 0x0;
	size_t pos;
	for (pos = 0; pos < numblocks; pos+=2)
	{
		hi ^= *(x+pos);
		lo ^= *(x+pos+1);
	}
	ret = ((uint64_t)hi)<<32 | lo;
	return ret;
}

void* readfile(size_t *size, const char* filepath)
{
	FILE* input = fopen(filepath, "r");
	size_t filesize;
	if (!input)
	{
		perror("Read file error!\n");
		exit(-1);
	}
	// Find file size
	fseek(input, 0, SEEK_END);
	filesize = ftell(input);
	fseek(input, 0, SEEK_SET);
	// Round filesize up if neccessary
	if (filesize%sizeof(uint32_t)) 
	{
		filesize = (filesize/sizeof(uint32_t)+1)*sizeof(uint32_t);
	}
	// Read file into buffer
	char *buffer;

	#ifdef __CUDACC__ // If using CUDA / nvcc
	cudaMallocHost((void**)&buffer, sizeof(char)*filesize);
	#else // else use standard C
	buffer = (char*)malloc(sizeof(char)*filesize);
	#endif

	memset(buffer, 0, sizeof(char)*filesize);
	size_t bytes_read = fread(buffer, sizeof(char), filesize, input);
	fclose(input);
	// Return
	*size = bytes_read;
	return buffer;
}
void blowfish_encryptblock(blowfish_context_t *ctx, uint32_t *hi, uint32_t *lo)
{
	uint32_t i, temp;

	for(i = 0; i < 16; i++) {
		*hi ^= ctx->pbox[i];
		*lo ^= BLOWFISH_F(*hi);
		temp = *hi, *hi = *lo, *lo = temp;
	}
	temp = *hi, *hi = *lo, *lo = temp;

	*lo ^= ctx->pbox[16];
	*hi ^= ctx->pbox[17];
}

void blowfish_decryptblock(blowfish_context_t *ctx, uint32_t *hi, uint32_t *lo)
{
	uint32_t i, temp;

	for(i = 17; i > 1; i--) {
		*hi ^= ctx->pbox[i];
		*lo ^= BLOWFISH_F(*hi);
		temp = *hi, *hi = *lo, *lo = temp;
	}
	temp = *hi, *hi = *lo, *lo = temp;

	*lo ^= ctx->pbox[1];
	*hi ^= ctx->pbox[0];
}

int blowfish_init(blowfish_context_t *ctx, const char *key, size_t keybytes)
{
	// Key can't be longer than 448 bits
	if(keybytes > 56) 
		return -1;

	int i, j, k;
	uint32_t calc;

	// Copy original S-boxes to context
	for(i = 0; i < 4; i++)
		memcpy(ctx->sbox[i], ORIG_S[i], 256 * sizeof(uint32_t));

	// Copy original P-boxes to context
	memcpy(ctx->pbox, ORIG_P, 18 * sizeof(uint32_t));

	// Blowfish key schedule
	if (keybytes) {
		// Do 18 times for 18 P-boxes
		for(i = 0, j = 0; i < 18; i++) {
			// Init counters
			calc = 0;
			for(k = 0; k < 4; k++) {
				// Cycle key around to fill P-boxes. 
				// CALC is the cycled key used to XOR with P-box
				// Move current key over 1 byte
				calc <<= 8;
				// Get next byte by using OR
				calc |= key[j++];
				// If at the end of key then wrap around
				if (j == keybytes)
					j = 0;
			}
			// XOR with content of P-box
			ctx->pbox[i] ^= calc;
		}
	}

	// Encrypt a 64-bit all zeroes block to use as subkey
	// Blowfish uses 64-bit blocks so here two 32-bit ints is used.
	// Using two ints also makes it easier to replace boxes
	// And don't have to split for left and right data later.
	uint32_t hi = 0, lo = 0;

	// Replace P-boxes with new subkeys
	for(i = 0; i < 18; i += 2) {
		blowfish_encryptblock(ctx, &hi, &lo);
		ctx->pbox[i] = hi;
		ctx->pbox[i + 1] = lo;
	}

	// Replace S-boxes with new subkeys
	for(i = 0; i < 4; i++) {
		for(j = 0; j < 256; j += 2) {
			blowfish_encryptblock(ctx, &hi, &lo);
			ctx->sbox[i][j] = hi;
			ctx->sbox[i][j + 1] = lo;
		}
	}

	// No errors
	return 0;
}

void blowfish_clean(blowfish_context_t *ctx)
{
	memset(ctx, 0, sizeof(blowfish_context_t));
	free(ctx);
}

void splashscreen(void)
{
	printf("%s",ascii_art);
}

#endif