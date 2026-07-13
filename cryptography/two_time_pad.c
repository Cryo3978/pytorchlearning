#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>


int main()
{
    char message1[] = "ATT8PM";
    char message2[] = "1KENEMS";
    
    printf("%s\n", message1);
	printf("%s\n", message2);
	
	int len = strlen(message1);
	
	uint8_t key[] = {0b10000000, 0b10000100, 0b10001000, 0b10000011, 0b10110000, 0b10000111};
	
	uint8_t *cipher1 = malloc(len);
	uint8_t *cipher2 = malloc(len);
	
	for(int i =0; i<len; i++){
		cipher1[i] = message1[i] ^ key[i];
		cipher2[i] = message2[i] ^ key[i];
	}
	
	for(int i =0;i<len;i++)
		printf("%02X ", cipher1[i]);
	printf("\n");
	
	for(int i =0;i<len;i++)
		printf("%02X ", cipher2[i]);	
	printf("\n");

	char *CXOR = malloc(len+1);
	
	for(int i =0; i<len; i++){
		CXOR[i] = cipher1[i] ^ cipher2[i];
	}
	
	for(int i =0; i<len; i++){
		printf("%02X ", CXOR[i]);
	}
	
	char *potential_message = malloc(len+1);
	char guess[] = "ATT";
	
	for(int i =0; i<strlen(guess); i++){
		potential_message[i] = CXOR[i] ^ guess[i];
	}
	
	printf("%s\n", potential_message);
}