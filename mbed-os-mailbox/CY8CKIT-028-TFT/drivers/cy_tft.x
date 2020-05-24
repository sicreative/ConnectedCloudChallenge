/***************************************************************************//**
* \file cy_tft.c
*
* \brief
*    This is display software i8080 interface source file
*
********************************************************************************
* \copyright
* Copyright 2018-2019 Cypress Semiconductor Corporation
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#ifndef CY8CKIT_028_TFT_CUSTOM_CONFIG
    #include "../cy8ckit_028_tft_config.h"
#endif

/***************************************************************************//**
* \file cy_tft.c
*
* \brief
*    This is display software i8080 interface source file
*
********************************************************************************
* \copyright
* Copyright 2018-2019 Cypress Semiconductor Corporation
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/







#include "cy_tft.h"
#include "mbed.h"


static BusInOut *DB;

/*
static DigitalInOut *DB8;
static DigitalInOut *DB9;
static DigitalInOut *DB10;
static DigitalInOut *DB11;
static DigitalInOut *DB12;
static DigitalInOut *DB13;
static DigitalInOut *DB14;
static DigitalInOut *DB15;*/
static DigitalInOut *NWR;
static DigitalInOut *DC;
static DigitalInOut *RST;
static DigitalInOut *NRD;



/*******************************************************************************
 * Changes data bus GPIO pins drive mode to digital Hi-Z with enabled input buffer.
 *******************************************************************************/
__STATIC_INLINE void data_io_set_input(void)
{
    /* enable input */

	//DB->mode(CYHAL_GPIO_DRIVE_NONE);
	//DB->input();
	/*
	DB8->mode(CYHAL_GPIO_DRIVE_NONE);
	DB9->mode(CYHAL_GPIO_DRIVE_NONE);
	DB10->mode(CYHAL_GPIO_DRIVE_NONE);
	DB11->mode(CYHAL_GPIO_DRIVE_NONE);
	DB12->mode(CYHAL_GPIO_DRIVE_NONE);
	DB13->mode(CYHAL_GPIO_DRIVE_NONE);
	DB14->mode(CYHAL_GPIO_DRIVE_NONE);
	DB15->mode(CYHAL_GPIO_DRIVE_NONE);*/


	/*
    cyhal_gpio_configure(CY_TFT_DB8, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_NONE);
    cyhal_gpio_configure(CY_TFT_DB9, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_NONE);
    cyhal_gpio_configure(CY_TFT_DB10, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_NONE);
    cyhal_gpio_configure(CY_TFT_DB11, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_NONE);
    cyhal_gpio_configure(CY_TFT_DB12, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_NONE);
    cyhal_gpio_configure(CY_TFT_DB13, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_NONE);
    cyhal_gpio_configure(CY_TFT_DB14, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_NONE);
    cyhal_gpio_configure(CY_TFT_DB15, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_NONE);*/
}

/*******************************************************************************
 * Changes data bus GPIO pins drive mode to strong drive with disabled input buffer.
 *******************************************************************************/
__STATIC_INLINE void data_io_set_output(void)
{
    /* enable output */
	//DB->mode(CYHAL_GPIO_DRIVE_STRONG);
//	DB->output();


 /*

	DB8->mode(CYHAL_GPIO_DRIVE_STRONG);
	DB9->mode(CYHAL_GPIO_DRIVE_STRONG);
	DB10->mode(CYHAL_GPIO_DRIVE_STRONG);
	DB11->mode(CYHAL_GPIO_DRIVE_STRONG);
	DB12->mode(CYHAL_GPIO_DRIVE_STRONG);
	DB13->mode(CYHAL_GPIO_DRIVE_STRONG);
	DB14->mode(CYHAL_GPIO_DRIVE_STRONG);
	DB15->mode(CYHAL_GPIO_DRIVE_STRONG);*/
	/*

    cyhal_gpio_configure(CY_TFT_DB8, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG);
    cyhal_gpio_configure(CY_TFT_DB9, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG);
    cyhal_gpio_configure(CY_TFT_DB10, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG);
    cyhal_gpio_configure(CY_TFT_DB11, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG);
    cyhal_gpio_configure(CY_TFT_DB12, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG);
    cyhal_gpio_configure(CY_TFT_DB13, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG);
    cyhal_gpio_configure(CY_TFT_DB14, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG);
    cyhal_gpio_configure(CY_TFT_DB15, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG);
*/

}

/*******************************************************************************
 * Writes one byte of data to the software i8080 interface.
 *******************************************************************************/
__STATIC_INLINE void write_data(uint8_t data)
{

	/*
	DB8->write( data     & 0x01);
	DB9->write( (data>>1)     & 0x01);
	DB10->write( (data>>2)     & 0x01);
	DB11->write( (data>>3)     & 0x01);
	DB12->write( (data>>4)     & 0x01);
	DB13->write( (data>>5)     & 0x01);
	DB14->write( (data>>6)     & 0x01);
	DB15->write( (data>>7)     & 0x01);*/

	DB->write(data);

	NWR->write(0);
	NWR->write(1);

	/*
    cyhal_gpio_write(CY_TFT_DB8,  data     & 0x01);
    cyhal_gpio_write(CY_TFT_DB9, (data>>1) & 0x01);
    cyhal_gpio_write(CY_TFT_DB10, (data>>2) & 0x01);
    cyhal_gpio_write(CY_TFT_DB11, (data>>3) & 0x01);
    cyhal_gpio_write(CY_TFT_DB12, (data>>4) & 0x01);
    cyhal_gpio_write(CY_TFT_DB13, (data>>5) & 0x01);
    cyhal_gpio_write(CY_TFT_DB14, (data>>6) & 0x01);
    cyhal_gpio_write(CY_TFT_DB15, (data>>7) & 0x01);

    cyhal_gpio_write(CY_TFT_NWR, 0u);
    cyhal_gpio_write(CY_TFT_NWR, 1u);*/
}

/*******************************************************************************
 * Reads one byte of data from the software i8080 interface.
 *******************************************************************************/
__STATIC_INLINE uint8_t read_data(void)
{

	 uint8_t data = 0u;

	// NRD->write(0);
	 /*
	 	data |= DB8->read();
	    data |= DB9->read()<<1;
	    data |= DB10->read()<<2;
	    data |= DB11->read()<<3;
	    data |= DB12->read()<<4;
	    data |= DB13->read()<<5;
	    data |= DB14->read()<<6;
	    data |= DB15->read()<<7;*/

//	 data = DB->read();

//	 NRD->write(1);
	 /*

    cyhal_gpio_write(CY_TFT_NRD, 0u);

    data |= cyhal_gpio_read(CY_TFT_DB8);
    data |= cyhal_gpio_read(CY_TFT_DB9)<<1;
    data |= cyhal_gpio_read(CY_TFT_DB10)<<2;
    data |= cyhal_gpio_read(CY_TFT_DB11)<<3;
    data |= cyhal_gpio_read(CY_TFT_DB12)<<4;
    data |= cyhal_gpio_read(CY_TFT_DB13)<<5;
    data |= cyhal_gpio_read(CY_TFT_DB14)<<6;
    data |= cyhal_gpio_read(CY_TFT_DB15)<<7;

    cyhal_gpio_write(CY_TFT_NRD, 1u);
    */


    return data;
}



cy_rslt_t cy_tft_io_init(void)
{

	DB = new BusInOut(CY_TFT_DB8,CY_TFT_DB9,CY_TFT_DB10,CY_TFT_DB11,CY_TFT_DB12,CY_TFT_DB13,CY_TFT_DB14,CY_TFT_DB15);
//	DB->mode(CYHAL_GPIO_DRIVE_STRONG);
	DB->output();

	/*

	DB8 =  new DigitalInOut(CY_TFT_DB8, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	DB9 =  new DigitalInOut(CY_TFT_DB9, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	DB10 =  new DigitalInOut(CY_TFT_DB10, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	DB11 =  new DigitalInOut(CY_TFT_DB11, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	DB12 =  new DigitalInOut(CY_TFT_DB12, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	DB13 =  new DigitalInOut(CY_TFT_DB13, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	DB14 =  new DigitalInOut(CY_TFT_DB14, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	DB15 =  new DigitalInOut(CY_TFT_DB15, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);*/

	NWR =  new DigitalInOut(CY_TFT_NWR, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 1);
	DC =  new DigitalInOut(CY_TFT_DC, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	RST =  new DigitalInOut(CY_TFT_RST, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 1);
	NRD =  new DigitalInOut(CY_TFT_NRD, PIN_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 1);


/*
	cy_rslt_t rslt;
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_DB9, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_DB10, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_DB11, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_DB12, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_DB13, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_DB14, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_DB15, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_NWR, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 1u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_DC, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_RST, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 1u);
    if (CY_RSLT_SUCCESS == rslt)
        rslt = cyhal_gpio_init(CY_TFT_NRD, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 1u);
    return rslt;
*/

	/*
	 *
	 DB8 = DigitalInOut CY_TFT_DB8 ;
	DB9 = DigitalInOut(CY_TFT_DB9);
	DB10 = DigitalInOut(CY_TFT_DB10);
	DB11 = DigitalInOut(CY_TFT_DB11);
	DB12 = DigitalInOut(CY_TFT_DB12);
	DB13 = DigitalInOut(CY_TFT_DB13);
	DB14 = DigitalInOut(CY_TFT_DB14);
	DB15 = DigitalInOut(CY_TFT_DB15);
	NWR = DigitalInOut(CY_TFT_NWR);
	DC = DigitalInOut(CY_TFT_DC);
	RST = DigitalInOut(CY_TFT_RST);
	NRD = DigitalInOut(CY_TFT_NRD);*/

	return CY_RSLT_SUCCESS;

}

void cy_tft_write_reset_pin(uint8_t value)
{
	RST->write(value);
   // cyhal_gpio_write(CY_TFT_RST, value);
}

void cy_tft_write_command(uint8_t data)
{
	DC->write(0);
   // cyhal_gpio_write(CY_TFT_DC, 0u);
    write_data(data);
}

void cy_tft_write_data(uint8_t data)
{
	DC->write(1);
 //   cyhal_gpio_write(CY_TFT_DC, 1u);
    write_data(data);
}

void cy_tft_write_command_stream(uint8_t data[], int num)
{
    int i = 0;
    DC->write(0);
   //cyhal_gpio_write(CY_TFT_DC, 0u);

    for(i = 0; i < num; i++)
    {
        write_data(data[i]);
    }
}

void cy_tft_write_data_stream(uint8_t data[], int num)
{
    int i = 0;
    DC->write(1);
   // cyhal_gpio_write(CY_TFT_DC, 1u);

    for(i = 0; i < num; i++)
    {
        write_data(data[i]);
    }
}

uint8_t cy_tft_read_data(void)
{
    uint8_t data;
    DC->write(1);
   // cyhal_gpio_write(CY_TFT_DC, 1u);

    data_io_set_input();
    data = read_data();
    data_io_set_output();

    return data;
}

void cy_tft_read_data_stream(uint8_t data[], int num)
{
   // cyhal_gpio_write(CY_TFT_DC, 1u);
	DC->write(1);

    data_io_set_input();

    for(int i = 0; i < num; i++)
    {
        data[i] = read_data();
    }

    data_io_set_output();
}

void cy_tft_io_free(void)
{
	delete DB;
	/*
	delete DB8;
	delete DB9;
	delete DB10;
	delete DB11;
	delete DB12;
	delete DB13;
	delete DB14;
	delete DB15;*/
	delete NWR;
	delete DC;
	delete RST;
	delete NRD;

	/*cyhal_gpio_free(CY_TFT_DB8);
    cyhal_gpio_free(CY_TFT_DB9);
    cyhal_gpio_free(CY_TFT_DB10);
    cyhal_gpio_free(CY_TFT_DB11);
    cyhal_gpio_free(CY_TFT_DB12);
    cyhal_gpio_free(CY_TFT_DB13);
    cyhal_gpio_free(CY_TFT_DB14);
    cyhal_gpio_free(CY_TFT_DB15);
    cyhal_gpio_free(CY_TFT_NWR);
    cyhal_gpio_free(CY_TFT_DC);
    cyhal_gpio_free(CY_TFT_RST);
    cyhal_gpio_free(CY_TFT_NRD);*/
}

/* [] END OF FILE */
