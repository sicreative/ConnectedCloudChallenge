/*
 * fram.cpp
 *
 * for FRAM access and sound read/write
 *
 * Created on: 23 Apr 2020
 *      Author: slee
 */

#include "fram.h"
#include "lcd_ui.h"
#include "serial_fram_api.h"

Status status;

/* QSPI HAL object */
cyhal_qspi_t qspi_host_fram_obj;

fram_bus_type_t access_mode;


void fram_reset_to_factory_default (void)
{
	uint8_t tx[ONE_BYTE_ACCESS];
	cy_rslt_t result;

	/* Write to CR2 to set the interface mode to SPI */
	tx[0] = 0x0;

	/* Try in DPI mode, in case device access mode is DPI after power up */
	result = fram_opcode_only_cmd(&qspi_host_fram_obj, MEM_CMD_WREN, FRAM_BUS_TYPE_DPI);


	result = fram_wrar_cmd(&qspi_host_fram_obj, CR2_ADDR_VOLATILE, tx, FRAM_BUS_TYPE_DPI);


	/* Try in QPI mode, in case device access mode is QPI after power up */
	result = fram_opcode_only_cmd(&qspi_host_fram_obj, MEM_CMD_WREN, FRAM_BUS_TYPE_QPI);


	result = fram_wrar_cmd(&qspi_host_fram_obj, CR2_ADDR_VOLATILE, tx, FRAM_BUS_TYPE_QPI);


	/* At this point device is confugred to a known (SPI) access mode.
	 * Write to CR1 to clear memory latency setting.
	 */
	result = fram_opcode_only_cmd(&qspi_host_fram_obj, MEM_CMD_WREN, FRAM_BUS_TYPE_SPI);


	tx[0] = 0x0;
	result = fram_wrar_cmd(&qspi_host_fram_obj, CR1_ADDR_VOLATILE, tx, FRAM_BUS_TYPE_SPI); /* Set the access mode */


	/* Write to CR5 to clear register latency setting */
	result = fram_opcode_only_cmd(&qspi_host_fram_obj, MEM_CMD_WREN, FRAM_BUS_TYPE_SPI);


	tx[0] = 0x0;
	result = fram_wrar_cmd(&qspi_host_fram_obj, CR5_ADDR_VOLATILE, tx, FRAM_BUS_TYPE_SPI); /* Set the access mode */

}

int fram_init(){

	uint8_t rx[DEV_ID_LENGTH_BYTE];
		uint8_t tx[DEV_ID_LENGTH_BYTE];

			cy_rslt_t result;
			uint32_t index;
			size_t length = DEV_ID_LENGTH_BYTE;

	/* Initialize the QSPI control and pins */
		result = cyhal_qspi_init(&qspi_host_fram_obj, CYBSP_QSPI_D0, CYBSP_QSPI_D1, CYBSP_QSPI_D2, CYBSP_QSPI_D3, NC, NC,
								NC, NC, CYBSP_QSPI_SCK, CYBSP_FRAM_CS, QSPI_BUS_FREQUENCY_HZ, 0);



		/* This function brings the device from DPI or QPI (if set) to
		 * factory default status and configuration register settings.
		 */
		fram_reset_to_factory_default();
		access_mode = FRAM_BUS_TYPE_SPI; /* Sets the access mode for the host */

		/* This sets the device access mode configuration in volatile register. To make this setting
		 * persistent (nonvolatile), you must write into nonvolatile register address. Refer to
		 * Excelon(TM) Ultra CY15x104QSN datasheet for register address details.
		 */
		result = fram_opcode_only_cmd(&qspi_host_fram_obj, MEM_CMD_WREN, access_mode); /* Send the WREN opcode prior to register write */


		/* DPI_EN - to set the DPI protocol enable bit '1' in F-RAM CR2 (CR2[4]).
		 * QPI_EN - to set the QPI protocol enable bit '1' in F-RAM CR2 (CR2[6]).
		 */
		tx[0] = QPI_EN;

		/* Write to CR2 to set the F-RAM access mode */
		result = fram_wrar_cmd(&qspi_host_fram_obj, CR2_ADDR_VOLATILE, tx, access_mode);


		/* Sets the new access mode for the host, as per the F-RAM access mode setting */
		access_mode = FRAM_BUS_TYPE_QPI;

		/*******************************************************************************
		* Set the F-RAM memory and register access latency
		********************************************************************************/

		/* Send the WREN opcode prior to write into config reg 1 (CR1) for the memory latency setting */
		result = fram_opcode_only_cmd(&qspi_host_fram_obj, MEM_CMD_WREN, access_mode);


		tx[0] = (MEM_LATENCY << MEM_LATENCY_POS);
		result = fram_wrar_cmd(&qspi_host_fram_obj, CR1_ADDR_VOLATILE, tx, access_mode); /* Sets the memory latency (MLC) */


		/* Send the WREN opcode prior to write into config reg 5 (CR5) for the register latency setting */
		result = fram_opcode_only_cmd(&qspi_host_fram_obj, MEM_CMD_WREN, access_mode);


		tx[0] = (REG_LATENCY << REG_LATENCY_POS);
		result = fram_wrar_cmd(&qspi_host_fram_obj, CR5_ADDR_VOLATILE, tx, access_mode); /* Sets the register latency (RLC) */


		/*******************************************************************************
		* Read and Display Device ID and Unique ID of QSPI F-RAM
		********************************************************************************/

		APP_INFO(( "Read QSPI F-RAM 8-byte DID - "));
		result = fram_read_id_cmd (&qspi_host_fram_obj, MEM_CMD_RDID, rx, &length, access_mode, REG_LATENCY);


		for(index = 0; index < length; index++)
		{
			APP_INFO(( "0x%02X ", rx[index]));
		}
		APP_INFO(( "\r\n"));

		APP_INFO(( "Read QSPI F-RAM 8-byte UDID - "));
		result = fram_read_id_cmd (&qspi_host_fram_obj, MEM_CMD_RUID, rx, &length, access_mode, REG_LATENCY);


		for(index = 0; index < length; index++)
		{
			APP_INFO(( "0x%02X ", rx[index] ));

		}
		APP_INFO(( "\r\n\r\n" ));



}

/* read program status */
int fram_read_status(){
	size_t length = sizeof(status);

	fram_read_cmd(&qspi_host_fram_obj, MEM_CMD_READ, STATUS_ADDRESS, 0, (uint8_t*) &status , &length , access_mode,5);

	/* Init if no status save before */
	if (status.guard != 0xFEFD){
		status.guard = 0xFEFD;
		status.sound_start = 0;
		status.sound_end = 0;
		status.sound_volume = 0;

	}
}

/* write program status */
int fram_write_status(){
	size_t length = sizeof(status);

	return fram_write_cmd(&qspi_host_fram_obj, MEM_CMD_WRITE, STATUS_ADDRESS, 0 , (uint8_t*) &status , &length , access_mode);
}

/* write sound to fram
 *
 * @param pos The pos of sound block
 * @param size no of block for send
 * @param data pointer of data
 *
 * */
int fram_write_sound(int pos,int size, void* data){

	if (pos<0 || pos>=RECORD_SOUND_TOTAL_BLOCK)
		return SOUND_OUT_OF_RANGE;

	int address = RECORD_SOUND_ADDRESS_START + ( pos * RECORD_SOUND_DATA_PRE_BLOCK);

	size_t length = RECORD_SOUND_DATA_PRE_BLOCK*size;

	return fram_write_cmd(&qspi_host_fram_obj, MEM_CMD_WRITE, address, 0 , (uint8_t*)data, &length , access_mode);

}

/* read sound block from fram
 *
 * @param pos The pos of sound block
 * @param size no of block for send
 * @param data
 *
 * @return result
 * */
int fram_read_sound(int pos,int size,void* data){
	if (pos<0 || pos>=RECORD_SOUND_TOTAL_BLOCK)
			return SOUND_OUT_OF_RANGE;

	int address = RECORD_SOUND_ADDRESS_START + ( pos * RECORD_SOUND_DATA_PRE_BLOCK);

	size_t length = RECORD_SOUND_DATA_PRE_BLOCK*size;

	return fram_read_cmd(&qspi_host_fram_obj, MEM_CMD_READ, address, 0, (uint8_t*) data, &length , access_mode,5);



}

/* read sound block from fram, each time read one block only
 *
 * @param pos The pos of sound block
 * @param size no of block for send
 * @param data
 *
 * @return result
 * */
int fram_read_sound(int pos,void* data){


	size_t length = RECORD_SOUND_DATA_PRE_BLOCK;

	return fram_read_sound(pos,1,data);

}



