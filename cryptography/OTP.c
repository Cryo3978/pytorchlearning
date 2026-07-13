#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

int main() {
	
	char message[] = "ATT8AM";
	int len = strlen(message);
	
	printf("%d\n", len);
	
	uint8_t key[] = {0b00110110, 0b01100110, 0b10010110, 0b10110101, 0b10010011, 0b01110110};
	uint8_t *cipher = malloc(len);
	
	for (int i = 0; i < len; i++)
	{
		cipher[i] = message[i] ^ key[i];
	}
	
	uint8_t *decrypted_message = malloc(len+1);
	
	for (int i = 0; i <len; i++)
	{
		decrypted_message[i] = cipher[i] ^ key[i];
	}
	
	decrypted_message[len] = '\0';
	
	printf("%s\n", message);
	
	printf("%s\n", decrypted_message);
	
	printf("Cipher: ");

	for(int i =0; i<len; i++)
	{
		printf("%02X ", cipher[i]);
	}
	
	printf("\n");
	
	free(cipher);
	free(decrypted_message);
	
    return 0;
}