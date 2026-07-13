#include <stdio.h>
#include <stdint.h>
#include <string.h>


int main()
{
    char *messages[] = {
        "MEET AT THE STATION",
        "SEND ME THE FILE NOW",
        "ATTACK AT MIDNIGHT!!",
        "THE PACKAGE ARRIVED",
        "WAIT FOR FURTHER ORDERS"
    };


    int num_messages = 5;


    uint8_t key[23] = {
        0x12,0x45,0xAA,0x91,0x33,
        0xFE,0x10,0x88,0x73,0x44,
        0x56,0x99,0x01,0xAB,0xCD,
        0x77,0x88,0x99,0x11,0x22,
        0x33,0x44,0x55
    };


    uint8_t ciphertext[5][23];


    for(int j=0;j<num_messages;j++)
    {
        for(int i=0;i<23;i++)
        {
            ciphertext[j][i]
                = messages[j][i]^key[i];
        }
    }


    printf("Ciphertexts:\n");


    for(int j=0;j<num_messages;j++)
    {
        for(int i=0;i<23;i++)
        {
            printf("%02X ", ciphertext[j][i]);
        }

        printf("\n");
    }


    
}