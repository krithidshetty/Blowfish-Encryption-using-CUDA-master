#ifndef IMPLEMENT_H_
#define IMPLEMENT_H_

#include "blowfish.h"
#include "const.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define NUM_STREAMS 256

void version(void)
{
	printf("\n");
}

__device__ __forceinline__ void cudaBlowfishEncryptBlock(blowfish_context_t *ctx, uint32_t *hi, uint32_t *lo)
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

__global__ void cudaBlowfishEncryptPtr(blowfish_context_t *globalCtx, uint32_t *ptr, size_t numblocks)
{
	size_t tid = threadIdx.x;
	size_t pos = (threadIdx.x + blockDim.x * blockIdx.x)<<1;
	__shared__ blowfish_context_t localCtx;

	localCtx.sbox[tid >> 8][tid & 0xFF] = globalCtx->sbox[tid >> 8][tid & 0xFF];
	localCtx.pbox[tid & 0xFF] = globalCtx->pbox[tid & 0xFF];

	__syncthreads();

	if (pos < numblocks)
	{
		uint32_t lo = ptr[pos+1];
		uint32_t hi = ptr[pos];
		cudaBlowfishEncryptBlock(&localCtx, &hi, &lo);
		ptr[pos+1] = lo;
		ptr[pos] = hi;
	}
}

__device__ __forceinline__ void cudaBlowfishDecryptBlock(blowfish_context_t *ctx, uint32_t *hi, uint32_t *lo)
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

__global__ void cudaBlowfishDecryptPtr(blowfish_context_t *globalCtx, uint32_t *ptr, size_t numblocks)
{
	size_t tid = threadIdx.x;
	size_t pos = (threadIdx.x + blockDim.x * blockIdx.x)<<1;
	__shared__ blowfish_context_t localCtx;

	localCtx.sbox[tid >> 8][tid & 0xFF] = globalCtx->sbox[tid >> 8][tid & 0xFF];
	localCtx.pbox[tid & 0xFF] = globalCtx->pbox[tid & 0xFF];

	__syncthreads();

	if (pos < numblocks)
	{
		uint32_t lo = ptr[pos+1];
		uint32_t hi = ptr[pos];
		cudaBlowfishDecryptBlock(&localCtx, &hi, &lo);
		ptr[pos+1] = lo;
		ptr[pos] = hi;
	}
}

#endif