#include "blowfish.h"
#include "implement.h"
#include "const.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char *argv[])
{
	//splashscreen();
	version();
	// Misc variables
	int status = 0;
	uint64_t hash_original, hash_encrypted, hash_decrypted;
	float cudaRuntime, rate;
	// File variables
	size_t filesize;
	const char *filepath = "../testfile";
	uint32_t *file = (uint32_t*) readfile(&filesize, filepath);
	size_t numblocks = filesize/sizeof(uint32_t);
	printf("File size = %zu bytes, numblocks = %zu\n", filesize, numblocks/2);
	// Encryption key
	const char *key = "TESTKEY";
	printf("Key = %s, length = %zu\n", key, strlen(key));
	// Create Blowfish context
	blowfish_context_t *context = (blowfish_context_t*) malloc(sizeof(blowfish_context_t));
	if(!context) 
	{
		printf("Could not allocate enough memory!\n");
		return -1;
	}

	// Initialize key schedule
	status = blowfish_init(context, key, strlen(key));
	if (status)
	{
		printf("Error initiating key\n");
		return -1;
	} else printf("Key schedule complete!\n");

	// Hash original file
	hash_original = hash(file, numblocks);
	printf("Original hash = %llx\n", (unsigned long long)hash_original);

	// CUDA Starts
	printf("CUDA Starts!\n");

	int i;
	const int numStreams = NUM_STREAMS;
	const int streamSize = filesize/numStreams;
	const int numBlocksPerStream = numblocks/numStreams;

	cudaStream_t stream[numStreams];

	for (i = 0; i < numStreams; ++i)
		cudaStreamCreate(&stream[i]);

	uint32_t *filegpu;
	cudaMalloc(&filegpu, filesize);
	// cudaMemcpy(filegpu, file, filesize, cudaMemcpyHostToDevice);

	blowfish_context_t *ctxgpu;
	cudaMalloc(&ctxgpu, sizeof(blowfish_context_t));
	cudaMemcpy(ctxgpu, context, sizeof(blowfish_context_t), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int blocksize = 1024;
	int gridsize = (numblocks/(blocksize<<1)+1)/numStreams;

	//__________ENCRYPTION__________
	printf("Encryption starts...\n");

	cudaEventRecord(start);

	for (i = 0; i < numStreams; ++i)
	{
		int offset = i * numBlocksPerStream;
		cudaMemcpyAsync(&filegpu[offset], &file[offset], streamSize, cudaMemcpyHostToDevice, stream[i]);
		cudaBlowfishEncryptPtr<<<gridsize, blocksize, 0, stream[i]>>>(ctxgpu, &filegpu[offset], numBlocksPerStream);
		cudaMemcpyAsync(&file[offset], &filegpu[offset], streamSize, cudaMemcpyDeviceToHost, stream[i]);
	}

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	hash_encrypted = hash(file, numblocks);

	cudaEventElapsedTime(&cudaRuntime, start, stop);
	rate = (filesize*1e3)/(cudaRuntime);

	printf("Encryption done!\n");
	printf("Time taken: %lf milliseconds\n", cudaRuntime);
	printf("Average speed: %lf MB/s\n", rate/MEGABYTE);
	printf("Encrypted hash = %llx\n", (unsigned long long)hash_encrypted);

	//__________DECRYPTION__________
	printf("Decryption starts...\n");

	cudaEventRecord(start);
	for (i = 0; i < numStreams; ++i)
	{
		int offset = i * numBlocksPerStream;
		cudaMemcpyAsync(&filegpu[offset], &file[offset], streamSize, cudaMemcpyHostToDevice, stream[i]);
		cudaBlowfishDecryptPtr<<<gridsize, blocksize, 0, stream[i]>>>(ctxgpu, &filegpu[offset], numBlocksPerStream);
		cudaMemcpyAsync(&file[offset], &filegpu[offset], streamSize, cudaMemcpyDeviceToHost, stream[i]);
	}

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	hash_decrypted = hash(file, numblocks);

	cudaEventElapsedTime(&cudaRuntime, start, stop);
	rate = (filesize*1e3)/(cudaRuntime);

	printf("Decryption done!\n");
	printf("Time taken: %lf milliseconds\n", cudaRuntime);
	printf("Average speed: %lf MB/s\n", rate/MEGABYTE);
	printf("Decrypted hash = %llx\n", (unsigned long long)hash_decrypted);

	// Check
	if (hash_decrypted == hash_original)
		printf("Hashes match! PASSED!\n");
	else
		printf("Hashes mismatch! FAILED!\n");

	//__________DONE__________
	blowfish_clean(context);
	cudaFreeHost(file);
	cudaFree(filegpu);
	return 0;
}
