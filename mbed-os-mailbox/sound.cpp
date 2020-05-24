/*
 * sound.cpp
 *
 *  Created on: 21 Apr 2020
 *      Author: slee
 *      
 *      
 *      For handling pcm mic and i2s sound
 */

#include "sound.h"
#include "mbed.h"
#include "cy_pdl.h"
#include "lcd_ui.h"
#include "cyhal.h"
#include "ak4954a.h"
#include "hw.h"
#include "hal/static_pinmap.h"
#include "fram.h"


extern serial_t stdio_uart;
extern int stdio_uart_inited;

#define ioss_0_port_10_pin_4_HSIOM P10_4_AUDIOSS_PDM_CLK
#define ioss_0_port_10_pin_5_HSIOM P10_5_AUDIOSS_PDM_DATA
#define ioss_0_port_10_pin_4_ENABLED 1U
#define ioss_0_port_10_pin_4_PORT GPIO_PRT10
#define ioss_0_port_10_pin_4_PORT_NUM 10U
#define ioss_0_port_10_pin_4_PIN 4U
#define ioss_0_port_10_pin_4_NUM 4U
#define ioss_0_port_10_pin_4_DRIVEMODE CY_GPIO_DM_STRONG_IN_OFF
#define ioss_0_port_10_pin_4_INIT_DRIVESTATE 1
#ifndef ioss_0_port_10_pin_4_HSIOM
	#define ioss_0_port_10_pin_4_HSIOM HSIOM_SEL_GPIO
#endif
#define ioss_0_port_10_pin_4_IRQ ioss_interrupts_gpio_10_IRQn
#if defined (CY_USING_HAL)
	#define ioss_0_port_10_pin_4_HAL_PORT_PIN P10_4
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_10_pin_4_HAL_IRQ CYHAL_GPIO_IRQ_NONE
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_10_pin_4_HAL_DIR CYHAL_GPIO_DIR_OUTPUT
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_10_pin_4_HAL_DRIVEMODE CYHAL_GPIO_DRIVE_STRONG
#endif //defined (CY_USING_HAL)
#define ioss_0_port_10_pin_5_ENABLED 1U
#define ioss_0_port_10_pin_5_PORT GPIO_PRT10
#define ioss_0_port_10_pin_5_PORT_NUM 10U
#define ioss_0_port_10_pin_5_PIN 5U
#define ioss_0_port_10_pin_5_NUM 5U
#define ioss_0_port_10_pin_5_DRIVEMODE CY_GPIO_DM_HIGHZ
#define ioss_0_port_10_pin_5_INIT_DRIVESTATE 1
#ifndef ioss_0_port_10_pin_5_HSIOM
	#define ioss_0_port_10_pin_5_HSIOM HSIOM_SEL_GPIO
#endif
#define ioss_0_port_10_pin_5_IRQ ioss_interrupts_gpio_10_IRQn
#if defined (CY_USING_HAL)
	#define ioss_0_port_10_pin_5_HAL_PORT_PIN P10_5
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_10_pin_5_HAL_IRQ CYHAL_GPIO_IRQ_NONE
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_10_pin_5_HAL_DIR CYHAL_GPIO_DIR_INPUT
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_10_pin_5_HAL_DRIVEMODE CYHAL_GPIO_DRIVE_NONE
#endif //defined (CY_USING_HAL)

#define CYBSP_I2S_ENABLED 1U
#define CYBSP_I2S_HW I2S
#define CYBSP_I2S_IRQ audioss_interrupt_i2s_IRQn
#define ioss_0_port_5_pin_1_HSIOM P5_1_AUDIOSS_TX_SCK
#define ioss_0_port_5_pin_2_HSIOM P5_2_AUDIOSS_TX_WS
#define ioss_0_port_5_pin_3_HSIOM P5_3_AUDIOSS_TX_SDO

#define CYBSP_DMA_I2S_tr_in_0_TRIGGER_OUT TRIG0_OUT_CPUSS_DW0_TR_IN3
#define CYBSP_DMA_PDM_tr_in_0_TRIGGER_OUT TRIG1_OUT_CPUSS_DW1_TR_IN2
#define CYBSP_I2S_tr_i2s_tx_req_0_TRIGGER_IN_0 TRIG0_IN_TR_GROUP13_OUTPUT1
#define CYBSP_I2S_tr_i2s_tx_req_0_TRIGGER_IN_1 TRIG13_IN_AUDIOSS_TR_I2S_TX_REQ
#define CYBSP_PDM_tr_pdm_rx_req_0_TRIGGER_IN_0 TRIG0_IN_TR_GROUP13_OUTPUT0
#define CYBSP_PDM_tr_pdm_rx_req_0_TRIGGER_IN_1 TRIG13_IN_AUDIOSS_TR_PDM_RX_REQ

#define CYBSP_PDM_ENABLED 1U
#define CYBSP_PDM_HW PDM

#define ioss_0_port_5_pin_1_ENABLED 1U
#define ioss_0_port_5_pin_1_PORT GPIO_PRT5
#define ioss_0_port_5_pin_1_PORT_NUM 5U
#define ioss_0_port_5_pin_1_PIN 1U
#define ioss_0_port_5_pin_1_NUM 1U
#define ioss_0_port_5_pin_1_DRIVEMODE CY_GPIO_DM_STRONG_IN_OFF
#define ioss_0_port_5_pin_1_INIT_DRIVESTATE 1
#ifndef ioss_0_port_5_pin_1_HSIOM
	#define ioss_0_port_5_pin_1_HSIOM HSIOM_SEL_GPIO
#endif
#define ioss_0_port_5_pin_1_IRQ ioss_interrupts_gpio_5_IRQn
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_1_HAL_PORT_PIN P5_1
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_1_HAL_IRQ CYHAL_GPIO_IRQ_NONE
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_1_HAL_DIR CYHAL_GPIO_DIR_OUTPUT
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_1_HAL_DRIVEMODE CYHAL_GPIO_DRIVE_STRONG
#endif //defined (CY_USING_HAL)
#define ioss_0_port_5_pin_2_ENABLED 1U
#define ioss_0_port_5_pin_2_PORT GPIO_PRT5
#define ioss_0_port_5_pin_2_PORT_NUM 5U
#define ioss_0_port_5_pin_2_PIN 2U
#define ioss_0_port_5_pin_2_NUM 2U
#define ioss_0_port_5_pin_2_DRIVEMODE CY_GPIO_DM_STRONG_IN_OFF
#define ioss_0_port_5_pin_2_INIT_DRIVESTATE 1
#ifndef ioss_0_port_5_pin_2_HSIOM
	#define ioss_0_port_5_pin_2_HSIOM HSIOM_SEL_GPIO
#endif
#define ioss_0_port_5_pin_2_IRQ ioss_interrupts_gpio_5_IRQn
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_2_HAL_PORT_PIN P5_2
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_2_HAL_IRQ CYHAL_GPIO_IRQ_NONE
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_2_HAL_DIR CYHAL_GPIO_DIR_OUTPUT
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_2_HAL_DRIVEMODE CYHAL_GPIO_DRIVE_STRONG
#endif //defined (CY_USING_HAL)
#define ioss_0_port_5_pin_3_ENABLED 1U
#define ioss_0_port_5_pin_3_PORT GPIO_PRT5
#define ioss_0_port_5_pin_3_PORT_NUM 5U
#define ioss_0_port_5_pin_3_PIN 3U
#define ioss_0_port_5_pin_3_NUM 3U
#define ioss_0_port_5_pin_3_DRIVEMODE CY_GPIO_DM_STRONG_IN_OFF
#define ioss_0_port_5_pin_3_INIT_DRIVESTATE 1
#ifndef ioss_0_port_5_pin_3_HSIOM
	#define ioss_0_port_5_pin_3_HSIOM HSIOM_SEL_GPIO
#endif
#define ioss_0_port_5_pin_3_IRQ ioss_interrupts_gpio_5_IRQn
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_3_HAL_PORT_PIN P5_3
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_3_HAL_IRQ CYHAL_GPIO_IRQ_NONE
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_3_HAL_DIR CYHAL_GPIO_DIR_OUTPUT
#endif //defined (CY_USING_HAL)
#if defined (CY_USING_HAL)
	#define ioss_0_port_5_pin_3_HAL_DRIVEMODE CYHAL_GPIO_DRIVE_STRONG
#endif //defined (CY_USING_HAL)


	const cy_stc_gpio_pin_config_t ioss_0_port_10_pin_4_config =
	{
		.outVal = 1,
		.driveMode = CY_GPIO_DM_STRONG_IN_OFF,
		.hsiom = ioss_0_port_10_pin_4_HSIOM,
		.intEdge = CY_GPIO_INTR_DISABLE,
		.intMask = 0UL,
		.vtrip = CY_GPIO_VTRIP_CMOS,
		.slewRate = CY_GPIO_SLEW_FAST,
		.driveSel = CY_GPIO_DRIVE_1_2,
		.vregEn = 0UL,
		.ibufMode = 0UL,
		.vtripSel = 0UL,
		.vrefSel = 0UL,
		.vohSel = 0UL,
	};
	#if defined (CY_USING_HAL)
		const cyhal_resource_inst_t ioss_0_port_10_pin_4_obj =
		{
			.type = CYHAL_RSC_GPIO,
			.block_num = ioss_0_port_10_pin_4_PORT_NUM,
			.channel_num = ioss_0_port_10_pin_4_PIN,
		};
	#endif //defined (CY_USING_HAL)
	const cy_stc_gpio_pin_config_t ioss_0_port_10_pin_5_config =
	{
		.outVal = 1,
		.driveMode = CY_GPIO_DM_HIGHZ,
		.hsiom = ioss_0_port_10_pin_5_HSIOM,
		.intEdge = CY_GPIO_INTR_DISABLE,
		.intMask = 0UL,
		.vtrip = CY_GPIO_VTRIP_CMOS,
		.slewRate = CY_GPIO_SLEW_FAST,
		.driveSel = CY_GPIO_DRIVE_1_2,
		.vregEn = 0UL,
		.ibufMode = 0UL,
		.vtripSel = 0UL,
		.vrefSel = 0UL,
		.vohSel = 0UL,
	};
	#if defined (CY_USING_HAL)
		const cyhal_resource_inst_t ioss_0_port_10_pin_5_obj =
		{
			.type = CYHAL_RSC_GPIO,
			.block_num = ioss_0_port_10_pin_5_PORT_NUM,
			.channel_num = ioss_0_port_10_pin_5_PIN,
		};
	#endif //defined (CY_USING_HAL)





	const cy_stc_pdm_pcm_config_t CYBSP_PDM_config =
	{
		.clkDiv = CY_PDM_PCM_CLK_DIV_1_4,
		.mclkDiv = CY_PDM_PCM_CLK_DIV_1_4,
		.ckoDiv = 2U,
		.ckoDelay = 3U,
		.sincDecRate = 32U,
		.chanSelect = CY_PDM_PCM_OUT_CHAN_LEFT,
		.chanSwapEnable = false,
		.highPassFilterGain = 8U,
		.highPassDisable = false,
		.softMuteCycles = CY_PDM_PCM_SOFT_MUTE_CYCLES_96,
		.softMuteFineGain = 1UL,
		.softMuteEnable = false,
		.wordLen = CY_PDM_PCM_WLEN_16_BIT,
		.signExtension = true,
		.gainLeft = CY_PDM_PCM_GAIN_10_5_DB,
		.gainRight = CY_PDM_PCM_GAIN_10_5_DB,
		.rxFifoTriggerLevel = 127U,
		.dmaTriggerEnable = false,
		.interruptMask = CY_PDM_PCM_INTR_RX_TRIGGER | CY_PDM_PCM_INTR_RX_OVERFLOW,



	};



	const cy_stc_i2s_config_t CYBSP_I2S_config =
	{
		.txEnabled = true,
		.rxEnabled = false,
		.txDmaTrigger = false,
		.rxDmaTrigger = false,
		.clkDiv = 12,
		.extClk = false,
		.txMasterMode = 1,
		.txAlignment = CY_I2S_I2S_MODE,
		.txWsPulseWidth = CY_I2S_WS_ONE_CHANNEL_LENGTH,
		.txWatchdogEnable = false,
		.txWatchdogValue = 4294967295,
		.txSdoLatchingTime = 0,
		.txSckoInversion = 0,
		.txSckiInversion = 0,
		.txChannels = 2,
		.txChannelLength = CY_I2S_LEN16,
		.txWordLength = CY_I2S_LEN16,
		.txOverheadValue = CY_I2S_OVHDATA_ONE,
		.txFifoTriggerLevel = 128,
		.rxMasterMode = 1,
		.rxAlignment = CY_I2S_I2S_MODE,
		.rxWsPulseWidth = CY_I2S_WS_ONE_CHANNEL_LENGTH,
		.rxWatchdogEnable = false,
		.rxWatchdogValue = 4294967295,
		.rxSdiLatchingTime = 0,
		.rxSckoInversion = 0,
		.rxSckiInversion = 0,
		.rxChannels = 2,
		.rxChannelLength = CY_I2S_LEN16,
		.rxWordLength = CY_I2S_LEN16,
		.rxSignExtension = 0,
		.rxFifoTriggerLevel = 128,
	};







		const cy_stc_gpio_pin_config_t ioss_0_port_5_pin_1_config =
		{
			.outVal = 1,
			.driveMode = CY_GPIO_DM_STRONG_IN_OFF,
			.hsiom = ioss_0_port_5_pin_1_HSIOM,
			.intEdge = CY_GPIO_INTR_DISABLE,
			.intMask = 0UL,
			.vtrip = CY_GPIO_VTRIP_CMOS,
			.slewRate = CY_GPIO_SLEW_FAST,
			.driveSel = CY_GPIO_DRIVE_1_2,
			.vregEn = 0UL,
			.ibufMode = 0UL,
			.vtripSel = 0UL,
			.vrefSel = 0UL,
			.vohSel = 0UL,
		};
		#if defined (CY_USING_HAL)
			const cyhal_resource_inst_t ioss_0_port_5_pin_1_obj =
			{
				.type = CYHAL_RSC_GPIO,
				.block_num = ioss_0_port_5_pin_1_PORT_NUM,
				.channel_num = ioss_0_port_5_pin_1_PIN,
			};
		#endif //defined (CY_USING_HAL)
		const cy_stc_gpio_pin_config_t ioss_0_port_5_pin_2_config =
		{
			.outVal = 1,
			.driveMode = CY_GPIO_DM_STRONG_IN_OFF,
			.hsiom = ioss_0_port_5_pin_2_HSIOM,
			.intEdge = CY_GPIO_INTR_DISABLE,
			.intMask = 0UL,
			.vtrip = CY_GPIO_VTRIP_CMOS,
			.slewRate = CY_GPIO_SLEW_FAST,
			.driveSel = CY_GPIO_DRIVE_1_2,
			.vregEn = 0UL,
			.ibufMode = 0UL,
			.vtripSel = 0UL,
			.vrefSel = 0UL,
			.vohSel = 0UL,
		};
		#if defined (CY_USING_HAL)
			const cyhal_resource_inst_t ioss_0_port_5_pin_2_obj =
			{
				.type = CYHAL_RSC_GPIO,
				.block_num = ioss_0_port_5_pin_2_PORT_NUM,
				.channel_num = ioss_0_port_5_pin_2_PIN,
			};
		#endif //defined (CY_USING_HAL)
		const cy_stc_gpio_pin_config_t ioss_0_port_5_pin_3_config =
		{
			.outVal = 1,
			.driveMode = CY_GPIO_DM_STRONG_IN_OFF,
			.hsiom = ioss_0_port_5_pin_3_HSIOM,
			.intEdge = CY_GPIO_INTR_DISABLE,
			.intMask = 0UL,
			.vtrip = CY_GPIO_VTRIP_CMOS,
			.slewRate = CY_GPIO_SLEW_FAST,
			.driveSel = CY_GPIO_DRIVE_FULL,
			.vregEn = 0UL,
			.ibufMode = 0UL,
			.vtripSel = 0UL,
			.vrefSel = 0UL,
			.vohSel = 0UL,
		};
		#if defined (CY_USING_HAL)
			const cyhal_resource_inst_t ioss_0_port_5_pin_3_obj =
			{
				.type = CYHAL_RSC_GPIO,
				.block_num = ioss_0_port_5_pin_3_PORT_NUM,
				.channel_num = ioss_0_port_5_pin_3_PIN,
			};
		#endif //defined (CY_USING_HAL)




const cy_stc_sysint_t intPDM_cfg = {.intrSrc = CYBSP_PDM_IRQ,
								    .intrPriority = 2u
				                   };

const cy_stc_sysint_t intI2S_cfg = {.intrSrc = CYBSP_I2S_IRQ,
					                 .intrPriority = 3u
					        		};

bool sound_busy_buffer = false;




int ind = 0;
bool half = false;




/* Index for FRAM storage, data back from 0 if start > end */
int sound_record_start_pos = 0;
int sound_record_end_pos = 0;
int sound_playingblock = 0;
int sound_endblock = 0;
int sound_steam_start_pos = 0;
int sound_steam_end_pos = 0;

bool sound_rec = false;
bool sound_play = false;


/* status used for update button*/
bool rec_button = false;
bool last_rec_button = false;
bool play_button = false;
bool last_play_button = false;
bool call_stop_play = false;
int last_volume = 0;
int volume = 0;
int lastdisplaysecond = -1;

void i2s_int();




int16_t sound_buffer[SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL*128+10];
bool sound_steam_onoff = false;

void sound_stop_playback(){





	NVIC_DisableIRQ((IRQn_Type)intI2S_cfg.intrSrc);

	Cy_I2S_DisableTx(CYBSP_I2S_HW);




	/* stop sound card */
    ak4954a_deactivate();



    /* free ports for UART*/

    cyhal_utils_disconnect_and_free(P5_0);
    cyhal_utils_disconnect_and_free(P5_1);

    /* Recovery UART console */
    const serial_pinmap_t pinmap = get_uart_pinmap(STDIO_UART_TX,STDIO_UART_RX );
    serial_init_direct(&stdio_uart, &pinmap);
    serial_baud(&stdio_uart, MBED_CONF_PLATFORM_STDIO_BAUD_RATE);





    stdio_uart_inited = true;
    UART_OUTPUT = true;

    sound_play = false;


}

void sound_start_playback(int start,int end){

	/* temp stop UART console output for playback */

	UART_OUTPUT = false;
	stdio_uart_inited = false;
	serial_break_set(&stdio_uart);
	serial_clear(&stdio_uart);
	serial_free(&stdio_uart);


	/* caculate the wait time for stop playback */

	int length =  end-start;

	if (start>end)
		length += RECORD_SOUND_TOTAL_BLOCK - start;

	/* every block time is 16000/128 = 125ms*/
	length *= 125 * 1000;


	/* activate GPIO port */
	Cy_GPIO_Pin_Init(ioss_0_port_5_pin_1_PORT, ioss_0_port_5_pin_1_PIN, &ioss_0_port_5_pin_1_config);

	/* init tracking var */
	sound_playingblock = start;
	sound_endblock = end;

	ak4954a_activate();

	Cy_SysInt_Init(&intI2S_cfg,&i2s_int);

	Cy_I2S_SetInterruptMask(CYBSP_I2S_HW, (CYBSP_I2S_HW,CY_I2S_INTR_TX_EMPTY | CY_I2S_INTR_TX_TRIGGER));

	Cy_I2S_ClearTxFifo(CYBSP_I2S_HW);

	Cy_I2S_EnableTx(CYBSP_I2S_HW);

	NVIC_EnableIRQ((IRQn_Type)intI2S_cfg.intrSrc);

	/* activate sound card */

	 sound_play = true;




}


int steam_input_section = 0;
int playback_repeat = 0;

void sound_steam_input(int section, int totalsection,char* data){
	APP_INFO(("sound write:%d,%d,%d\n",section,totalsection,steam_input_section));

    if (sound_rec)
    	return;

    if (steam_input_section==0){
    	sound_record_start_pos = sound_record_end_pos+1;
    	sound_record_start_pos += (SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL) + sound_record_start_pos%(SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL);
    	sound_record_end_pos = sound_record_start_pos;
    }

    int pos = sound_record_end_pos + SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL*section;

    if (pos>RECORD_SOUND_TOTAL_BLOCK)
    	pos-=RECORD_SOUND_TOTAL_BLOCK;

	fram_write_sound(pos,SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL,(void*)data);



	++steam_input_section;



	if (totalsection!=steam_input_section)
		return;

	steam_input_section = 0;

	sound_record_end_pos +=totalsection*SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL;
	if (sound_record_end_pos>RECORD_SOUND_TOTAL_BLOCK)
		sound_record_end_pos-=RECORD_SOUND_TOTAL_BLOCK;

	status.sound_start = sound_record_start_pos;
	status.sound_end = sound_record_end_pos;

	sound_playback_repeat(10);

	hw_set_button(0,true);



}


void sound_start_record(){

	/* suspend last record if steam not finish */



	sound_record_start_pos = sound_record_end_pos+1;
	sound_record_start_pos += (SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL) + sound_record_start_pos%(SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL);

	sound_record_end_pos = sound_record_start_pos+1;

	sound_steam_start_pos =  sound_record_start_pos;
	sound_steam_end_pos = sound_record_start_pos;

	/* Startup PDM and I2S */
	Cy_PDM_PCM_Enable(CYBSP_PDM_HW);




	APP_INFO(( "\nStart Record Audio  %d - %d frame \n",sound_record_start_pos,sound_record_end_pos ));

	Cy_PDM_PCM_ClearFifo(CYBSP_PDM_HW);

	NVIC_EnableIRQ((IRQn_Type)intPDM_cfg.intrSrc);

	sound_rec = true;



}


void sound_stop_record(){



	 NVIC_DisableIRQ((IRQn_Type)intPDM_cfg.intrSrc);

	 Cy_PDM_PCM_Disable(CYBSP_PDM_HW);

	 sound_record_end_pos-=sound_record_end_pos%(SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL);
	 if (sound_record_end_pos==sound_record_start_pos)
		 if(++sound_record_end_pos>RECORD_SOUND_TOTAL_BLOCK)
			 sound_record_end_pos=0;



	 APP_INFO(( "\nEnd Audio  %d - %d frame \n",sound_record_start_pos,sound_record_end_pos ));

	 status.sound_start = sound_record_start_pos;
	 status.sound_end = sound_record_end_pos;


	 fram_write_status();

	 hw_set_button(1,false);

	 sound_rec = false;

	 sound_steam_start_pos = sound_record_start_pos;
	 sound_steam_end_pos = sound_record_end_pos;

	 sound_steam_onoff = true;



}


void sound_volume_slider(int id,int pos){

	volume = pos;

}


void sound_button(int id,bool status){

	if (id==0)
		play_button = status;
	else
		rec_button = status;

}

void sound_playback_repeat(int num){
	playback_repeat = num;
}


void sound_update(){

	  // check_write();

		if (volume!=last_volume){

			int attenuat = (300-volume+30) / 4;

			ak4954a_adjust_volume(attenuat);
			 status.sound_volume = attenuat;
			 fram_write_status();

			last_volume = volume;

		}

	if (play_button != last_play_button){

		if (rec_button){
			hw_set_button(1,false);
			hw_set_button(0,false);
			return;
		}


		if (play_button){
			if (!sound_play)
				sound_start_playback(sound_record_start_pos,sound_record_end_pos);
		}else{
			if (sound_play)
				sound_stop_playback();
		}


		last_play_button = play_button;
	}

	if (rec_button != last_rec_button){

		if (play_button){
				hw_set_button(0,false);
				hw_set_button(1,false);

				return;
			}

		if (rec_button){
			if (!sound_rec)
				sound_start_record();
		}else{
			if (sound_rec)
				sound_stop_record();
		}

		last_rec_button = rec_button;
	}

	if (call_stop_play){
		if (sound_play){
			hw_set_button(0,false);
			//sound_stop_playback();
		}


		call_stop_play = false;

	}




	if (sound_rec){
		int total_block = 0;

		if (sound_record_end_pos<sound_record_start_pos)
			total_block+=(RECORD_SOUND_TOTAL_BLOCK-sound_record_start_pos)+sound_record_end_pos;
		else
			total_block = sound_record_end_pos-sound_record_start_pos;

		total_block/=128;

		if (total_block==lastdisplaysecond)
			return;


		lcd_msg("%d",2,total_block+1);

		lastdisplaysecond = total_block;

	}else if (sound_play){
		int total_block = 0;

				if (sound_endblock<sound_playingblock)
					total_block+=(RECORD_SOUND_TOTAL_BLOCK-sound_playingblock)+sound_endblock;
				else
					total_block = sound_endblock-sound_playingblock;

				total_block/=128;

				if (total_block==lastdisplaysecond)
					return;


				lcd_msg("%d",2,total_block);

				lastdisplaysecond = total_block;


	}else{
		if (lastdisplaysecond!=-1){
			lcd_msg("",2);
		  lastdisplaysecond = -1;
		}
	}


}





/*interrupt for pdm playback */
void pdm_int(){


/*
 *
 * The PDM FIFO trigger at 128,
 * interrupt read 128 sample and save to buffer.
 * Write to FRAM when buffer full filled
 *
 *
 */


	int next = ind+128;
	while(ind<next){

		sound_buffer[ind++] = Cy_PDM_PCM_ReadFifo(CYBSP_PDM_HW);
	}




	if (ind==(SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL*128)){
				fram_write_sound(sound_record_end_pos,SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL,(void*)sound_buffer);
				ind=0;

				bool cross = sound_record_end_pos <= sound_record_start_pos;

				sound_record_end_pos+=SOUND_STEAM_BUFFER_SIZE*SOUND_STEAM_BUFFER_MUL;

				if (cross)
					cross = sound_record_end_pos > sound_record_start_pos;



				if (sound_record_end_pos>=RECORD_SOUND_TOTAL_BLOCK)
										sound_record_end_pos = 0;



				if (cross)
						sound_record_start_pos = sound_record_end_pos+1;


	}




	Cy_PDM_PCM_ClearInterrupt(CYBSP_PDM_HW,CY_PDM_PCM_INTR_RX_TRIGGER | CY_PDM_PCM_INTR_RX_OVERFLOW);



}

/*interrupt for i2s playback */
void i2s_int(){



	/* interval read from fram because we feed i2s FIFO 64 sample a lot,
	* i2s with 256 samples, as Stero data require Left/Right sample, each sample
	 * write to i2s FIFO twice as comsume 64 mono samples */

	if (!half)
		fram_read_sound(sound_playingblock,(void*)sound_buffer);

	for(int i=0+half*64;i<(64+half*64);i++){
	       Cy_I2S_WriteTxData(CYBSP_I2S_HW, sound_buffer[i]);
		   Cy_I2S_WriteTxData(CYBSP_I2S_HW, sound_buffer[i]);

	}



	if (half){
		if (++sound_playingblock>=RECORD_SOUND_TOTAL_BLOCK)
			sound_playingblock = 0;

		if (sound_playingblock==sound_endblock){

			if (playback_repeat==0){
				Cy_I2S_SetInterruptMask(CYBSP_I2S_HW,0);
				call_stop_play=true;
			}else{
				sound_playingblock = sound_record_start_pos;
				--playback_repeat;

			}
		}


	}

	half=!half;


	Cy_I2S_ClearInterrupt(CYBSP_I2S_HW,(CYBSP_I2S_HW,CY_I2S_INTR_TX_EMPTY | CY_I2S_INTR_TX_TRIGGER));

}




void sound_init(){
	

	Cy_SysLib_Delay(2000);



/*    Init GPIO  */

	Cy_GPIO_Pin_Init(ioss_0_port_10_pin_4_PORT, ioss_0_port_10_pin_4_PIN, &ioss_0_port_10_pin_4_config);
#if defined (CY_USING_HAL)
	cyhal_hwmgr_reserve(&ioss_0_port_10_pin_4_obj);
#endif //defined (CY_USING_HAL)

	Cy_GPIO_Pin_Init(ioss_0_port_10_pin_5_PORT, ioss_0_port_10_pin_5_PIN, &ioss_0_port_10_pin_5_config);
#if defined (CY_USING_HAL)
	cyhal_hwmgr_reserve(&ioss_0_port_10_pin_5_obj);
#endif //defined (CY_USING_HAL)





	 Cy_GPIO_Pin_Init(ioss_0_port_5_pin_2_PORT, ioss_0_port_5_pin_2_PIN, &ioss_0_port_5_pin_2_config);
	#if defined (CY_USING_HAL)
		cyhal_hwmgr_reserve(&ioss_0_port_5_pin_2_obj);
	#endif //defined (CY_USING_HAL)

	 Cy_GPIO_Pin_Init(ioss_0_port_5_pin_3_PORT, ioss_0_port_5_pin_3_PIN, &ioss_0_port_5_pin_3_config);
	#if defined (CY_USING_HAL)
		cyhal_hwmgr_reserve(&ioss_0_port_5_pin_3_obj);
	#endif //defined (CY_USING_HAL)




	 /* Initialize PDM/PCM block */
		cy_en_pdm_pcm_status_t result = Cy_PDM_PCM_Init(CYBSP_PDM_HW, &CYBSP_PDM_config);
			if (result==CY_PDM_PCM_BAD_PARAM)
				APP_INFO(( "\nBAD PDM setup"));

	 /* Initialize AK4954A */
	 ak4954a_init(hw_ak4954a_transmit);

	 ak4954a_adjust_volume(status.sound_volume);
	 /* deactivate AK4954A */
	 ak4954a_deactivate();



	 Cy_I2S_Init(CYBSP_I2S_HW, &CYBSP_I2S_config);

	 Cy_SysInt_Init(&intPDM_cfg,&pdm_int);

	/* Startup PDM and I2S */
	Cy_PDM_PCM_Enable(CYBSP_PDM_HW);

	/* recovery record pos from FRAM */
	sound_record_start_pos = status.sound_start;
	sound_record_end_pos = status.sound_end;

	if (sound_record_start_pos==sound_record_end_pos){
		sound_record_start_pos = 0;
		sound_record_end_pos = 1;
	}

	return;




	
}

int sound_num_wait_for_steam(){

	if (!sound_steam_onoff || sound_steam_start_pos == sound_steam_end_pos)
		return 0;

	bool back = sound_steam_start_pos>sound_steam_end_pos;


	int len = sound_steam_start_pos>sound_steam_end_pos?RECORD_SOUND_TOTAL_BLOCK-sound_steam_start_pos+1+sound_steam_end_pos:
										sound_steam_end_pos-sound_steam_start_pos;


	return len;

}



int sound_steam(int16_t* buffer){


	if (!sound_steam_onoff)
		return 0;

	if (sound_steam_end_pos>=sound_steam_start_pos && sound_steam_end_pos-sound_steam_start_pos<SOUND_STEAM_BUFFER_SIZE)
		sound_steam_onoff=false;

	fram_read_sound(sound_steam_start_pos,SOUND_STEAM_BUFFER_SIZE,(void*)buffer);

	APP_INFO(("sound_load_steam %d,   %d\n",sound_steam_start_pos,sound_steam_end_pos));

	sound_steam_start_pos+=SOUND_STEAM_BUFFER_SIZE;

	if (sound_steam_start_pos>=RECORD_SOUND_TOTAL_BLOCK)
		sound_steam_start_pos = 0;


	return SOUND_STEAM_BUFFER_SIZE;



}





