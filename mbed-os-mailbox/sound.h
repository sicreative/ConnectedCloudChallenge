/*
 * sound.h
 *
 *  Created on: 21 Apr 2020
 *      Author: slee
 */

#ifndef SOUND_H_
#define	SOUND_H_


#include "cycfg_notices.h"
#include "cy_dma.h"
#if defined (CY_USING_HAL)
	#include "cyhal_hwmgr.h"
#endif //defined (CY_USING_HAL)

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *
 *
 * DMA Config
 *
 */
#define CYBSP_DMA_PDM_ENABLED 1U
#define CYBSP_DMA_PDM_HW DW1
#define CYBSP_DMA_PDM_CHANNEL 2U
#define CYBSP_DMA_PDM_IRQ cpuss_interrupts_dw1_2_IRQn
#define CYBSP_PDM_IRQ audioss_interrupt_pdm_IRQn
#define CYBSP_DMA_I2S_ENABLED 1U
#define CYBSP_DMA_I2S_HW DW0
#define CYBSP_DMA_I2S_CHANNEL 3U
#define CYBSP_I2S_IRQ audioss_interrupt_i2s_IRQn
#define CYBSP_DMA_I2S_IRQ cpuss_interrupts_dw0_3_IRQn

extern const cy_stc_dma_descriptor_config_t CYBSP_DMA_PDM_Descriptor_0_config;
extern cy_stc_dma_descriptor_t CYBSP_DMA_PDM_Descriptor_0;
extern const cy_stc_dma_channel_config_t CYBSP_DMA_PDM_channelConfig;
#if defined (CY_USING_HAL)
	extern const cyhal_resource_inst_t CYBSP_DMA_PDM_obj;
#endif //defined (CY_USING_HAL)
extern const cy_stc_dma_descriptor_config_t CYBSP_DMA_I2S_Descriptor_0_config;
extern cy_stc_dma_descriptor_t CYBSP_DMA_I2S_Descriptor_0;
extern const cy_stc_dma_channel_config_t CYBSP_DMA_I2S_channelConfig;
#if defined (CY_USING_HAL)
	extern const cyhal_resource_inst_t CYBSP_DMA_I2S_obj;
#endif //defined (CY_USING_HAL)

/* Size of the recorded buffer */
#define BUFFER_SIZE     256u

/* Number of channels (Stereo) */
#define NUM_CHANNELS    1u

/* Number of block send once MQTT */
#define SOUND_STEAM_BUFFER_SIZE			6

/* INTERNAL BUFFER MUL */
#define SOUND_STEAM_BUFFER_MUL			1


extern int16_t sound_buffer[];


/* Init Sound */
void sound_init();
/* called by main loop for sound interface update */
void sound_update();

/* call for volume slider */
void sound_volume_slider(int id,int pos);

/* call for sound button */
void sound_button(int id,bool status);

/*
 * call from awsiot to get sound steam, repeat call this for update next block
 * @param buffer Send buffer
 *
 * @Return sended block size
 * */
int sound_steam(int16_t* buffer);

/*
 * get num of steam waiting
 *
 *
 * @Return num of total waiting block
 * */
int sound_num_wait_for_steam();


void sound_steam_input(int section, int totalsection,char* data);


void sound_playback_repeat(int num);



#if defined(__cplusplus)
}
#endif

#endif /* SOUND_H_ */
