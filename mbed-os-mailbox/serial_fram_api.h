
/***************************************************************************//**
* \serial_fram_api.h
*
* \brief
* Defines function prototypes and macros for F-RAM access in SPI/DPI/QPI modes.
* See CY15x104QSN datasheet for details on the command macros defined in this file.
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

#pragma once

#if defined(CY_USING_HAL)
#include "cyhal_qspi.h"
#include "cyhal.h"
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#define CYBSP_FRAM_CS           (P11_0)/* F-RAM chip select control pin on the board */
#define QSPI_BUS_FREQUENCY_HZ   (50000000lu) /* QSPI clock frequency */

typedef enum
{
	FRAM_BUS_TYPE_SPI = CYHAL_QSPI_CFG_BUS_SINGLE,
	FRAM_BUS_TYPE_DPI = CYHAL_QSPI_CFG_BUS_DUAL,
	FRAM_BUS_TYPE_QPI = CYHAL_QSPI_CFG_BUS_QUAD
}fram_bus_type_t;

/***************************************
*  QSPI F-RAM specific Opcodes
***************************************/

/* QSPI F-RAM Read Opcodes*/
#define MEM_CMD_READ              (0x03)    /* Memory READ opcode */
#define MEM_CMD_FAST_READ         (0x0B)    /* Memory FastRead opcode */
#define MEM_CMD_DIOR              (0xBB)    /* Memory read opcode - Dual IO Read (DIOR) */
#define MEM_CMD_QIOR              (0xEB)    /* Memory read opcode - Quad IO Read (QIOR) */
#define MEM_CMD_DOR               (0x3B)    /* Memory read opcode - Dual output Read (DOR) */
#define MEM_CMD_QOR               (0x6B)    /* Memory read opcode - Quad output Read (QOR) */
#define MEM_CMD_SSRD              (0x4B) 	/* Special sector -256 byte read */

/* QSPI F-RAM Write Opcodes*/
#define MEM_CMD_WRITE             (0x02) 	/* Memory WRITE opcode */
#define MEM_CMD_WREN              (0x06) 	/* Write Enable */
#define MEM_CMD_SSWR              (0x42) 	/* Special sector -256 byte write */
#define MEM_CMD_FAST_WRITE        (0xDA)    /* Fast Write opcode */
#define MEM_CMD_DIW               (0xA2)    /* Dual input write - data on dual line */
#define MEM_CMD_DIOW              (0xA1)    /* Dual input input write - address and data on dual line */
#define MEM_CMD_QIW               (0x32)    /* Quad input write - data on quad line */
#define MEM_CMD_QIOW              (0xD2)    /* Quad input input write - address and data on quad line */

/* Status and Configuration Register Opcodes */
#define MEM_CMD_WRSR              (0x01) 	/* Write the status and configuration registers (SR1, CR1, CR2, CR4, CR5) */
#define MEM_CMD_WRDI              (0x04) 	/* Write disable the write enable latch */
#define MEM_CMD_RDSR1             (0x05) 	/* Read Status Register 1 (SR1) */
#define MEM_CMD_RDSR2             (0x07) 	/* Read Status Register 2 (SR2) */
#define MEM_CMD_RDCR1             (0x35) 	/* Read Configuration Register 1 (CR1) */
#define MEM_CMD_RDCR2             (0x3F) 	/* Read Configuration Register 2 (CR2) */
#define MEM_CMD_RDCR4             (0x45) 	/* Read Configuration Register 4 (CR4) */
#define MEM_CMD_RDCR5             (0x5E) 	/* Read Configuration Register 5 (CR5) */
#define MEM_CMD_WRAR              (0x71) 	/* Write to any one register (SR1, CR1, CR2, CR4, CR5), one byte at a time */
#define MEM_CMD_RDAR              (0x65) 	/* Read from any one register (SR1, CR1, CR2, CR4, CR5), one byte at a time */

/* Device ID Access Opcodes*/	
#define MEM_CMD_RDID              (0x9F) 	/* Command to read 8-byte device ID */
#define MEM_CMD_RDSN              (0xC3) 	/* Command to read 8-byte Serial Number (SN) */
#define MEM_CMD_RUID              (0x4C) 	/* Command to read 8-byte unique ID */
#define MEM_CMD_WRSN              (0xC2) 	/* Command to write 8-byte Serial Number (SN) */

/* Low Power Mode Commands*/
#define MEM_CMD_ENTDPD            (0xB9) 	/* Enter DPD*/
#define MEM_CMD_ENTHBN            (0xBA) 	/* Enter Hibernate*/

/* Status and Configuration Register Address */
#define SR0_ADDR                  (0x000000) /* Status Register 0 address for read using RDAR command */
#define SR1_ADDR                  (0x000001) /* Status Register 1 address for read using RDAR command */
#define CR1_ADDR_NV               (0x000002) /* Configuration Register 1 address for nonvolatile write using WRAR command */
#define CR2_ADDR_NV               (0x000003) /* Configuration Register 2 address for nonvolatile write using WRAR command */
#define CR4_ADDR_NV               (0x000005) /* Configuration Register 4 address for nonvolatile write using WRAR command */
#define CR5_ADDR_NV               (0x000006) /* Configuration Register 5 address for nonvolatile write using WRAR command */
#define CR1_ADDR_VOLATILE         (0x070002) /* Configuration Register 1 address for volatile write using WRAR command */
#define CR2_ADDR_VOLATILE         (0x070003) /* Configuration Register 2 address for volatile write using WRAR command */
#define CR4_ADDR_VOLATILE         (0x070005) /* Configuration Register 4 address for volatile write using WRAR command */
#define CR5_ADDR_VOLATILE         (0x070006) /* Configuration Register 5 address for volatile write using WRAR command */

/* Memory and Register Latency Settings */
#define MEM_LATENCY               (0x05)    /* For 80 MHz or below SPI clock */
                                            /* DPI & QPI mode accesses need minimum latency of 2, i.e. 0x02 */    
#define REG_LATENCY               (0x01)    /* For the SPI clock above 50 MHz */
#define MEM_LATENCY_POS           (0x04)    /* Shifts MEM_LATENCY value to MLC bits in CR2 (CR2[7:4]) */
#define REG_LATENCY_POS           (0x06)    /* Shifts REG_LATENCY value to RLC bits in CR5 (CR[7:6] */

/* Register Length */
#define ONE_BYTE_ACCESS           (0x01)    /* Buffer size for opcode only commands and register access */
#define DEV_ID_LENGTH_BYTE        (0x08)    /* Access length for device ID and Serial Number Registers */
#define DPI_EN                    (0x10)    /* Sets the DPI bit in CR2[4] to enable DPI protocol */
#define QPI_EN                    (0x40)    /* Sets the QPI bit in CR2[6] to enable QPI protocol */
  

/******************************************************************************
* Function prototypes
******************************************************************************/
cy_rslt_t fram_opcode_only_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t cmd, fram_bus_type_t bus_width);
cy_rslt_t fram_read_id_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t opcode, uint8_t *id_rdata, size_t *id_length, fram_bus_type_t bus_width, uint8_t latency_code);
cy_rslt_t fram_read_SRxCRx_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t opcode, uint8_t *reg_rdata, fram_bus_type_t bus_width, uint8_t latency_code);
cy_rslt_t fram_wrar_cmd (cyhal_qspi_t *qspi_host_fram, uint32_t reg_address, uint8_t *reg_wdata, fram_bus_type_t bus_width);
cy_rslt_t fram_rdar_cmd (cyhal_qspi_t *qspi_host_fram, uint32_t reg_address, uint8_t *reg_rdata, fram_bus_type_t bus_width, uint8_t latency_code);
cy_rslt_t fram_read_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t opcode, uint32_t mem_address, uint8_t mode_byte, uint8_t *mem_rdata, size_t *data_length, fram_bus_type_t bus_width, uint8_t latency_code);
cy_rslt_t fram_write_cmd (cyhal_qspi_t *qspi_host_fram, uint8_t opcode, uint32_t mem_address, uint8_t mode_byte, uint8_t *mem_wdata, size_t *data_length, fram_bus_type_t bus_width);

#if defined(__cplusplus)
}
#endif
