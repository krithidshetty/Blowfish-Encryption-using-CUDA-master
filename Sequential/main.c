#include "blowfish.h"
#include "implement.h"
#include "const.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char *argv[])
{
	version();
	// Misc variables
	int status = 0;
	uint64_t hash_original, hash_encrypted, hash_decrypted;
	double runtime, rate;
	// File variables
	size_t filesize;
	char *filepath = "../testfile";
	uint32_t *file = readfile(&filesize, filepath);
	size_t numblocks = filesize/sizeof(uint32_t);
	printf("File size = %zu bytes, numblocks = %zu\n", filesize, numblocks/2);
	// Encryption key
	char *key = "TESTKEY";
	printf("Key = %s, length = %zu\n", key, strlen(key));
	// Create Blowfish context
	blowfish_context_t *context = malloc(sizeof(blowfish_context_t));
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

	//__________ENCRYPTION__________
	printf("Encryption starts...\n");

	blowfish_encryptptr(context, file, numblocks, &runtime, &rate);
	hash_encrypted = hash(file, numblocks);

	printf("Encryption done!\n");
	printf("Time taken: %lf milliseconds\n", runtime*1e3);
	printf("Average speed: %lf MB/s\n", rate/MEGABYTE);
	printf("Encrypted hash = %llx\n", (unsigned long long)hash_encrypted);

	//__________DECRYPTION__________
	printf("Decryption starts...\n");

	blowfish_decryptptr(context, file, numblocks, &runtime, &rate);
	hash_decrypted = hash(file, numblocks);

	printf("Decryption done!\n");
	printf("Time taken: %lf milliseconds\n", runtime*1e3);
	printf("Average speed: %lf MB/s\n", rate/MEGABYTE);
	printf("Decrypted hash = %llx\n", (unsigned long long)hash_decrypted);

	// Check
	if (hash_decrypted == hash_original)
		printf("Hashes match! PASSED!\n");
	else
		printf("Hashes mismatch! FAILED!\n");

	//__________DONE__________
	blowfish_clean(context);
	free(file);
	return 0;
}
