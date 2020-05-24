/*
* serial_fram_api.c
*
* Description - Provides API for interfacing with Cypress Exelon-Ultra (QSPI F-RAM).
* This API is a wrapper around the QSPI HAL API to make it specific for QSPI F-RAM access.
*
******************************************************************************
* Copyright (2019), Cypress Semiconductor Corporation.
******************************************************************************
* This software, including source code, documentation and related materials
* (“Software”), is owned by Cypress Semiconductor Corporation or one of its
* subsidiaries (“Cypress”) and is protected by and subject to worldwide patent
* protection (United States and foreign), United States copyright laws and
* international treaty provisions. Therefore, you may use this Software only
* as provided in the license agreement accompanying the software package from
* which you obtained this Software (“EULA”).
*
* If no EULA applies, Cypress hereby grants you a personal, nonexclusive,
* non-transferable license to copy, modify, and compile the Software source
* code solely for use in connection with Cypress’s integrated circuit products.
* Any reproduction, modification, translation, compilation, or representation
* of this Software except as specified above is prohibited without the express
* written permission of Cypress.
*
* Disclaimer: THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, NONINFRINGEMENT, IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. Cypress
* reserves the right to make changes to the Software without notice. Cypress
* does not assume any liability arising out of the application or use of the
* Software or any product or circuit described in the Software. Cypress does
* not authorize its products for use in any products where a malfunction or
* failure of the Cypress product may reasonably be expected to result in
* significant property damage, injury or death (“High Risk Product”). By
* including Cypress’s product in a High Risk Product, the manufacturer of such
* system or application assumes all risk of such use and in doing so agrees to
* indemnify Cypress against all liability.
*****************************************​**************************************/

/******************************************************************************
* Header file (s)
******************************************************************************/
#include "serial_fram_api.h"

#ifdef CY_IP_MXSMIF

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

cyhal_qspi_command_t command_frame;

/*******************************************************************************
* Function Name: fram_opcode_only_cmd
********************************************************************************
*
* Summary:
*  This function executes opcode only command in F-RAM.
*
* Parameters:
*  cyhal_qspi_t *qspi_host_fram -  quad spi configuration structure for HAL function
*
*  uint8_t cmd - fram opcode to be executed via this function
*
*  fram_bus_type_t bus_width - qspi bus width for access - single, dual, or quad
*
* Return:
*  cy_rslt_t - command transfer status over quad spi bus
*
*******************************************************************************/

cy_rslt_t fram_opcode_only_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t cmd, fram_bus_type_t bus_width)
{
	uint8_t tx[ONE_BYTE_ACCESS];
	size_t tx_length = 0;
	
	command_frame.instruction.bus_width = bus_width;
	command_frame.instruction.value = cmd;
	command_frame.instruction.disabled = false;

	command_frame.address.disabled = true;
	command_frame.mode_bits.disabled = true;
	command_frame.dummy_count = 0;
	command_frame.data.bus_width = bus_width;

	return cyhal_qspi_transfer(qspi_host_fram, &command_frame, tx, tx_length, tx, tx_length);
}

/*******************************************************************************
* Function Name: fram_read_SRxCRx_cmd
********************************************************************************
*
* Summary:
*  This function executes status and configuration register read opcodes.
*
* Parameters:
*  cyhal_qspi_t *qspi_host_fram -  Quad SPI configuration structure for HAL function
*
*  uint8_t opcode - F-RAM register read opcode to be executed via this function
*
*  uint8_t *reg_rdata - stores register read data (byte)
*
*  fram_bus_type_t bus_width - QSPI bus width for access - single, dual, or quad
*
*  uint8_t latency_code - clock cycle latency during register read (RLC)
*
* Return:
*  cy_rslt_t - command transfer status over Quad SPI bus
*
*******************************************************************************/

cy_rslt_t fram_read_SRxCRx_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t opcode, uint8_t *reg_rdata, fram_bus_type_t bus_width, uint8_t latency_code)
{
	size_t reg_length = ONE_BYTE_ACCESS;

	command_frame.instruction.bus_width = bus_width;
	command_frame.instruction.value = opcode;
	command_frame.instruction.disabled = false;
	
	command_frame.address.disabled = true;
	command_frame.mode_bits.disabled = true;
	command_frame.dummy_count = latency_code;

	command_frame.data.bus_width = bus_width;

	return cyhal_qspi_read(qspi_host_fram, &command_frame, reg_rdata, &reg_length);
}

/*******************************************************************************
* Function Name: fram_wrar_cmd
********************************************************************************
*
* Summary:
*  This function executes write any register (WRAR) opcode.
*
* Parameters:
*  cyhal_qspi_t *qspi_host_fram - Quad SPI configuration structure for HAL function
*
*  uint32_t reg_address - register address for write using WRAR
*
*  uint8_t *reg_wdata - register write data (byte)
*
*  fram_bus_type_t bus_width - QSPI bus width for access - single, dual, or quad
*
* Return:
*  cy_rslt_t - command transfer status over Quad SPI bus
*
*******************************************************************************/
cy_rslt_t fram_wrar_cmd (cyhal_qspi_t *qspi_host_fram, uint32_t reg_address, uint8_t *reg_wdata, fram_bus_type_t bus_width)
{
	size_t tx_length = ONE_BYTE_ACCESS;

	command_frame.instruction.bus_width = bus_width;
	command_frame.instruction.value = MEM_CMD_WRAR;
	command_frame.instruction.disabled = false;

	command_frame.address.bus_width = bus_width;
	command_frame.address.size = CYHAL_QSPI_CFG_SIZE_24;
	command_frame.address.value = reg_address;
	command_frame.address.disabled = false;
	command_frame.mode_bits.disabled = true;
	command_frame.dummy_count = 0;

	command_frame.data.bus_width = bus_width;

	return cyhal_qspi_write(qspi_host_fram, &command_frame, reg_wdata, &tx_length);
}

/*******************************************************************************
* Function Name: fram_rdar_cmd
********************************************************************************
*
* Summary:
*  This function executes read any register (RDAR) opcode.
*
* Parameters:
*  cyhal_qspi_t *qspi_host_fram -  Quad SPI configuration structure for HAL function
*
*  uint32_t reg_address - register address for read using RDAR
*
*  uint8_t *reg_rdata - register read data (byte)
*
*  fram_bus_type_t bus_width - SPI bus width for access - single, dual, or quad
*
*  uint8_t latency_code - clock cycle latency during register read (RLC)
*
* Return:
*  cy_rslt_t - command transfer status over Quad SPI bus
*
*******************************************************************************/
cy_rslt_t fram_rdar_cmd (cyhal_qspi_t *qspi_host_fram, uint32_t reg_address, uint8_t *reg_rdata, fram_bus_type_t bus_width, uint8_t latency_code)
{
	size_t reg_length = ONE_BYTE_ACCESS;

	command_frame.instruction.bus_width = bus_width;
	command_frame.instruction.value = MEM_CMD_RDAR;
	command_frame.instruction.disabled = false;

	command_frame.address.bus_width = bus_width;
	command_frame.address.size = CYHAL_QSPI_CFG_SIZE_24;
	command_frame.address.value = reg_address;
	command_frame.address.disabled = false;

	command_frame.mode_bits.disabled = true;
	command_frame.dummy_count = latency_code;

	command_frame.data.bus_width = bus_width;

	return cyhal_qspi_read(qspi_host_fram, &command_frame, reg_rdata, &reg_length);
}

/*******************************************************************************
* Function Name: fram_read_id_cmd
********************************************************************************
*
* Summary:
*  This function executes various device ID read commands.
*
* Parameters:
*  cyhal_qspi_t *qspi_host_fram -  Quad SPI configuration structure for HAL function
*
*  uint8_t opcode - F-RAM ID read command to be executed via this function
*
*  uint8_t *id_rdata - read device id
*
*  size_t *id_length - ID length (8 bytes)
*
*  fram_bus_type_t bus_width - SPI bus width for access - single, dual, or quad
*
*  uint8_t latency_code - clock cycle latency during register read (RLC)
*
* Return:
*  cy_rslt_t - command transfer status over Quad SPI bus
*
*******************************************************************************/
cy_rslt_t fram_read_id_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t opcode, uint8_t *id_rdata, size_t *id_length, fram_bus_type_t bus_width, uint8_t latency_code)
{
	command_frame.instruction.bus_width = bus_width;
	command_frame.instruction.value = opcode;
	command_frame.instruction.disabled = false;

	command_frame.address.disabled = true;
	command_frame.mode_bits.disabled = true;
	command_frame.dummy_count = latency_code;

	command_frame.data.bus_width = bus_width;

	return cyhal_qspi_read(qspi_host_fram, &command_frame, id_rdata, id_length);
}

/*******************************************************************************
* Function Name: fram_read_cmd
********************************************************************************
*
* Summary:
*  This function executes F-RAM memory read commands.
*
* Parameters:
*  cyhal_qspi_t *qspi_host_fram -  Quad SPI configuration structure for HAL function
*
*  uint8_t opcode - F-RAM read opcode to be executed via this function
*
*  uint32_t mem_address - F-RAM memory read start address
*
*  uint8_t mode_byte - set the xip mode for commands supporting xip
*
*  uint8_t *mem_rdata - read buffer to store read data from memory
*
*  size_t *data_length - read data length (65536 bytes max)
*
*  fram_bus_type_t bus_width - SPI bus width for access - single, dual, or quad
*
*  uint8_t latency_code - clock cycle latency during memry read (MLC)
*
* Return:
*  cy_rslt_t - command transfer status over Quad SPI bus
*
*******************************************************************************/
cy_rslt_t fram_read_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t opcode, uint32_t mem_address, uint8_t mode_byte, uint8_t *mem_rdata, size_t *data_length, fram_bus_type_t bus_width, uint8_t latency_code)
{
	command_frame.instruction.bus_width = bus_width;
	command_frame.instruction.value = opcode;
	command_frame.instruction.disabled = false;

	command_frame.address.bus_width = bus_width;
	command_frame.address.size = CYHAL_QSPI_CFG_SIZE_24;
	command_frame.address.value = mem_address;
	command_frame.address.disabled = false;

	if (opcode == MEM_CMD_FAST_READ)
	{
		command_frame.mode_bits.disabled = false;
		command_frame.mode_bits.bus_width = bus_width;
		command_frame.mode_bits.size = CYHAL_QSPI_CFG_SIZE_8;
		command_frame.mode_bits.value = mode_byte;
	}
	else
	{
		command_frame.mode_bits.disabled = true;
	}
	command_frame.dummy_count = latency_code;
	command_frame.data.bus_width = bus_width;

	return cyhal_qspi_read(qspi_host_fram, &command_frame, mem_rdata, data_length);
}

/*******************************************************************************
* Function Name: fram_write_cmd
********************************************************************************
*
* Summary:
*  This function executes F-RAM write commands.
*
* Parameters:
*  cyhal_qspi_t *qspi_host_fram -  Quad SPI configuration structure for HAL function
*
*  uint8_t opcode - F-RAM write opcode to be executed via this function
*
*  uint32_t mem_address - F-RAM memory write start address
*
*  uint8_t mode_byte - set the XIP mode for commands supporting xip
*
*  uint8_t *mem_wdata - buffer to hold data to be written into F-RAM memory
*
*  size_t *data_length - write data length (65536 bytes max)
*
*  fram_bus_type_t bus_width - SPI bus width for access - single, dual, or quad
*
* Return:
*  cy_rslt_t - command transfer status over Quad SPI bus
*
*******************************************************************************/
cy_rslt_t fram_write_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t opcode, uint32_t mem_address, uint8_t mode_byte, uint8_t *mem_wdata, size_t *data_length, fram_bus_type_t bus_width)
{
	cy_rslt_t result;
	
	result = fram_opcode_only_cmd (qspi_host_fram, MEM_CMD_WREN, bus_width);
	
	if(result == CY_RSLT_SUCCESS)
	{
		command_frame.instruction.bus_width = bus_width;
		command_frame.instruction.value = opcode;
		command_frame.instruction.disabled = false;

		command_frame.address.bus_width = bus_width;
		command_frame.address.size = CYHAL_QSPI_CFG_SIZE_24;
		command_frame.address.value = mem_address;
		command_frame.address.disabled = false;

		if (opcode == MEM_CMD_FAST_WRITE)
		{
			command_frame.mode_bits.bus_width = bus_width;
			command_frame.mode_bits.size = CYHAL_QSPI_CFG_SIZE_8;
			command_frame.mode_bits.value = mode_byte;
			command_frame.mode_bits.disabled = false;
		}
		else
		{
			command_frame.mode_bits.disabled = true;
		}	
		command_frame.dummy_count = 0;
		command_frame.data.bus_width = bus_width;





		result = cyhal_qspi_write(qspi_host_fram, &command_frame, mem_wdata, data_length);
	}
	
	return result;
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* CY_IP_MXSMIF */


