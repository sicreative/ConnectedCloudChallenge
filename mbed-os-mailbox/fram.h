/*
 * fram.h
 *
 * Work for fram access
 *
 *  Created on: 23 Apr 2020
 *      Author: slee
 */

#ifndef FRAM_H_
#define FRAM_H_

#include "mbed.h"

#define SOUND_OUT_OF_RANGE -1;

/*MEMORY MAPPING*/
#define RECORD_SOUND_ADDRESS_START 0x1FFFF
#define RECORD_SOUND_DATA_PRE_BLOCK 256
#define RECORD_SOUND_TOTAL_BLOCK 1536

#define STATUS_ADDRESS 0xFFFF
typedef struct {
	int32_t guard = 0xFEFD;
	int16_t sound_start;
	int16_t sound_end;
	uint8_t sound_volume;
	char pass_hash[64];

} Status;

extern Status status;

int fram_init();

int fram_read_status(void);
int fram_write_status(void);

int fram_write_sound(int pos,int size, void* data);
int fram_read_sound(int pos,void* data);
int fram_read_sound(int pos,int size,void* data);


#endif /* FRAM_H_ */


