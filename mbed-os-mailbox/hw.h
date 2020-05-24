/**
 *
 *
 *
 *  Layer between Hardware and AWS
 *
 *
 */






#ifndef HW_H 

//#define USE_CAPSENSE_TUNER

/* 1 for send sensor data to AWS IoT */
#define SEND_AWS_LOG	1

#include "json_aws.h"


#define HW_ENVELOPE_AWS_ID "envelope"
#define HW_ALS_ID "als"
#define HW_MOVEMENT_ID "movement"

#define HW_BUTTON_AWS_ID "button_"
#define HW_SLIDER_AWS_ID "slider_"

#define HW_BUTTON_AWS_ON "0"
#define HW_BUTTON_AWS_OFF "1"

#define NUM_OF_BUTTON 2
#define NUM_OF_SLIDER 1

extern bool hw_button[];
extern int hw_slider[];



/**
 *
 * Function for AK4954A transmit
 *
 * @param reg_addr address to be updated
 * @param data 8-bit data to be written in the register
 *
 * @return error status
 *
 */
uint32_t hw_ak4954a_transmit(uint8_t reg_addr, uint8_t data);
/**
 *
 * For Read Sensor
 *
 *
 *
 */

void sensor_read(void);


/**
 *
 * For Init BMI260 sensor
 *
 *
 *
 */

void hw_bmi160_init(void);

/**
 *
 * Read BMI260 sensor
 *
 *
 *
 */

void hw_read_bmi160();



/**
 *
 *
 * Read ambient light sensor
 *
 *
 */
void hw_read_als();


/**
 *
 * Flipping the condition of button
 *
 * @param id buttonid
 * @return button current status
 *
 */
int hw_flipping_button(int id);

/**
 *
 * Set the condition of button
 *
 * @param id buttonid
 * @param onoff status of button
 * @return button current status
 *
 */
int hw_set_button(int id, bool onoff);

/**
 *
 * Set slider pos
 *
 * @param id sliderid
 * @param slidervalue
 * @return slider current pos
 *
 */
int hw_set_slider(int id, int value);


/**
 *
 * Sync from AWS
 *
 * @param json from AWS
 *
 * @return 1 with update 0 no update
 */
int hw_aws_sync(JSON* json);

#endif
