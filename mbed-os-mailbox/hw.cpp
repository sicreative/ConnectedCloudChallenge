/**
 *
 *
 * middle layer for update and cooperate the devices status and sensor reading
 *
 */


#include "hw.h"
#include "lcd_ui.h"
#include "awsiot.h"
#include "mbed.h"
#include <string>
#include "cybsp.h"
#include "sound.h"

#include "analogin_api.h"

#include "mbed_error.h"
#include "cyhal_adc.h"
#include "cyhal_utils.h"
#include "cyhal_pin_package.h"
#include "cy_pdl.h"
#include "ak4954a.h"

bool hw_button[NUM_OF_BUTTON];
int hw_slider[NUM_OF_SLIDER];

bool hw_aws_publish = false;




#include "bmi160.h"


#define SENSOR_SCAN_PERIOD_MS 1000


struct bmi160_dev sensor;

I2C i2c(P6_1 , P6_0 );
AnalogIn als(P10_0);
DigitalOut ext_led(P13_6);



struct bmi160_sensor_data accel;
struct bmi160_sensor_data gyro;
float lux;
bool hw_with_mail = false;

void hw_sendAWSCommand(string device, string value);



/* function pointer for sound card  */
uint32_t hw_ak4954a_transmit(uint8_t reg_addr, uint8_t data){

	uint32_t result;
	char buffer[AK4954A_PACKET_SIZE];

    buffer[0] = reg_addr;
    buffer[1] = data;

    /* Send the data over the I2C */
    result = i2c.write(AK4954A_I2C_ADDR << 1, buffer,AK4954A_PACKET_SIZE);

    return result;


}

/* function pointer for motion driver */
void hw_bmi160_delay(uint32_t period){

	Cy_SysLib_Delay(period);
}

/* function pointer for read motion sensor */
int8_t hw_bmi160_read(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, uint16_t len){


	i2c.write(dev_addr,(char*)&reg_addr,1);
	return i2c.read(dev_addr,(char*)data,len);



}

/* function pointer for write motion sensor */
int8_t hw_bmi160_write(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, uint16_t len){

	char writedata[len+1];

	writedata[0]= reg_addr;

	for(int i=0;i<len;i++)
		writedata[i+1]=data[i];

	return i2c.write(dev_addr,writedata,len+1);

}

/* Init of motion sensor */
void hw_bmi160_init(){



	sensor.id = BMI160_I2C_ADDR << 1;
	sensor.interface = BMI160_I2C_INTF;
	sensor.read = hw_bmi160_read;
	sensor.write = hw_bmi160_write;


	sensor.delay_ms = hw_bmi160_delay;

	int8_t rslt = BMI160_OK;
	rslt = bmi160_init(&sensor);

	if (rslt==BMI160_OK){

		APP_INFO(( "BMI160 Init Sucess."));
	}else if (rslt==BMI160_E_DEV_NOT_FOUND){
		APP_INFO(( "BMI160 Device not found."));
	}else{

		APP_INFO(( "BMI160 Init Error. %d",rslt));
	}




	/* Select the Output data rate, range of accelerometer sensor */
	sensor.accel_cfg.odr = BMI160_ACCEL_ODR_1600HZ ;
	sensor.accel_cfg.range = BMI160_ACCEL_RANGE_2G;
	sensor.accel_cfg.bw = BMI160_ACCEL_BW_NORMAL_AVG4;

	/* Select the power mode of accelerometer sensor */
	sensor.accel_cfg.power = BMI160_ACCEL_NORMAL_MODE;


	sensor.gyro_cfg.odr = BMI160_GYRO_ODR_3200HZ ;
	sensor.gyro_cfg.range = BMI160_GYRO_RANGE_2000_DPS;
	sensor.gyro_cfg.bw = BMI160_GYRO_BW_NORMAL_MODE;

	sensor.gyro_cfg.power = BMI160_GYRO_NORMAL_MODE;

	rslt = bmi160_set_sens_conf(&sensor);




	rslt = bmi160_set_sens_conf(&sensor);




	if (rslt==BMI160_OK){

			APP_INFO(( "BMI160 Conf Sucess.\n"));
		}else{

			APP_INFO(( "BMI160 Conf Error. %d\n",rslt));
		}


	struct bmi160_pmu_status power_status;
		rslt = bmi160_get_power_mode(&power_status,&sensor);


		if (rslt==BMI160_OK){
		APP_INFO(( "BMI160 power status: %d %d \n",power_status.accel_pmu_status,power_status.gyro_pmu_status));
		}




	bmi160_int_settg int_config;

	/* Select the Interrupt channel/pin */
	int_config.int_channel = BMI160_INT_CHANNEL_1;// Interrupt channel/pin 1

	/* Select the Interrupt type */
	int_config.int_type = BMI160_ACC_ANY_MOTION_INT;// Choosing Any motion interrupt
	/* Select the interrupt channel/pin settings */




	int_config.int_pin_settg.output_en = BMI160_ENABLE;// Enabling interrupt pins to act as output pin
	int_config.int_pin_settg.output_mode = BMI160_ENABLE;// Choosing push-pull mode for interrupt pin
	int_config.int_pin_settg.output_type = BMI160_DISABLE;// Choosing active low output
	int_config.int_pin_settg.edge_ctrl = BMI160_ENABLE;// Choosing edge triggered output
	int_config.int_pin_settg.input_en = BMI160_DISABLE;// Disabling interrupt pin to act as input
	int_config.int_pin_settg.latch_dur = BMI160_LATCH_DUR_NONE;// non-latched output

	/* Select the Any-motion interrupt parameters */
	int_config.int_type_cfg.acc_any_motion_int.anymotion_en = BMI160_ENABLE;// 1- Enable the any-motion, 0- disable any-motion
	int_config.int_type_cfg.acc_any_motion_int.anymotion_x = BMI160_ENABLE;// Enabling x-axis for any motion interrupt
	int_config.int_type_cfg.acc_any_motion_int.anymotion_y = BMI160_ENABLE;// Enabling y-axis for any motion interrupt
	int_config.int_type_cfg.acc_any_motion_int.anymotion_z = BMI160_ENABLE;// Enabling z-axis for any motion interrupt
	int_config.int_type_cfg.acc_any_motion_int.anymotion_dur = 0;// any-motion duration
	int_config.int_type_cfg.acc_any_motion_int.anymotion_thr = 20;// (2-g range) -> (slope_thr) * 3.91 mg, (4-g range) -> (slope_thr) * 7.81 mg, (8-g range) ->(slope_thr) * 15.63 mg, (16-g range) -> (slope_thr) * 31.25 mg

	/* Set the Any-motion interrupt */
	rslt = bmi160_set_int_config(&int_config, &sensor);
	/* sensor is an instance of the structure bmi160_dev  */
	if (rslt==BMI160_OK){

					APP_INFO(( "BMI160 set int Sucess.\n"));
				}else{

					APP_INFO(( "BMI160 set int Error. %d\n",rslt));
				}

	int_config.int_channel = BMI160_INT_CHANNEL_2;
	rslt = bmi160_set_int_config(&int_config, &sensor);
	if (rslt==BMI160_OK){

				APP_INFO(( "BMI160 set int Sucess.\n"));
			}else{

				APP_INFO(( "BMI160 set int Error. %d\n",rslt));
			}





	if (rslt==BMI160_OK){

			APP_INFO(( "BMI160 reset Sucess.\n"));
		}else{

			APP_INFO(( "BMI160 reset Error. %d\n",rslt));
		}

	ext_led = 0;


}



/* display sensor data */
void display_sensor(){

	lcd_status_msg("a:%02d,%02d,%02d g:%03d,%03d,%03d, l:%04.2f %s",accel.x,accel.y,accel.z,gyro.x,gyro.y,gyro.z,lux,hw_with_mail?"Mail":"");


#if (SEND_AWS_LOG == 1)

	char buffer[50];

	sprintf(buffer,"a:%05d,%05d,%05d g:%03d,%03d,%03d, l:%04.2f %s",accel.x,accel.y,accel.z,gyro.x,gyro.y,gyro.z,lux,hw_with_mail?"Mail":"");

	hw_sendAWSCommand("log",buffer);

#endif

}

/* read lighting sensor */
void hw_read_als(){

	lux = (als.read()*3.3*500);

	APP_INFO(( "\nLight Sensor value:  %f\n",lux));


}


/* read motion sensor */
void hw_read_bmi160(){

	int8_t rslt = BMI160_OK;


	/* To read both Accel and Gyro data */
	bmi160_get_sensor_data((BMI160_ACCEL_SEL | BMI160_GYRO_SEL), &accel, &gyro, &sensor);


	if (rslt==BMI160_OK){
		APP_INFO(( "BMI160 Read Sucess %d %d %d, %d, %d,%d, %d \n",accel.x,accel.y,accel.z,gyro.x,gyro.y,gyro.z,gyro.sensortime));
	}else{
		APP_INFO(( "BMI160 Conf Error. %d\n",rslt));
	}

}


/* check with mail from ALS sensor */
void check_with_mail(){

	/* record before LED flash */
	hw_read_als();
	float pre_no_ext_led_lux = lux;
	/* LED flash and record */
	ext_led = 1;
	Cy_SysLib_Delay(3);
	hw_read_als();
	float ext_led_lux = lux;
	ext_led = 0;
	/* LED off and record */
	Cy_SysLib_Delay(5);
	hw_read_als();

	bool prev_hw_with_mail = hw_with_mail;

	/* Check diff of flash/non flash light level, if more that threshold (50) mean with obstruct (mail)*/
	if (abs(ext_led_lux-lux)>50 && abs(ext_led_lux-pre_no_ext_led_lux)>50 && abs(pre_no_ext_led_lux-lux)<20)
		hw_with_mail = false;
	else
		hw_with_mail = true;

	/* update shadow if status changed  */
	if (prev_hw_with_mail != hw_with_mail)
		hw_sendAWSCommand("mail",hw_with_mail?"1":"0");

	APP_INFO(( hw_with_mail ? "with obstruct\n" : "no obstruct\n" ));

}

/* read sensor data, call from main loop */
void sensor_read(){
	hw_read_bmi160();
	check_with_mail();
	display_sensor();



}

/* function for send AWS shadows for device no id */
void hw_sendAWSCommand(string device, string value){

	if (!hw_aws_publish)
		return;

	awsiot_update(device,value);
}

/* function for send AWS shadows for device with id */
void hw_sendAWSCommand(string device, int id, int value){

	if (!hw_aws_publish)
		return;
	device.append(to_string(id));
	awsiot_update(device,string(to_string(value)));
}

/* function for flip the button */
int hw_flipping_button(int id){

	if (id>=NUM_OF_BUTTON)
		return -1;

	return hw_set_button(id,!hw_button[id]);


}

/* update button */
int hw_set_button(int id,bool onoff){

	if (id>=NUM_OF_BUTTON)
		return -1;

	if (hw_button[id]==onoff)
		return 0;


	hw_button[id]=onoff;

/* update LCD */
    if (onoff){

    	lcd_show_button(id);
    }
    else{


    	lcd_close_button(id);
    }

/* update sound */
    sound_button(id,onoff);

/* update shadow */
	hw_sendAWSCommand(HW_BUTTON_AWS_ID,id,onoff);

	return 1;


}

/* update slider */
int hw_set_slider(int id, int value){

	if (id>=NUM_OF_SLIDER)
		return -1;

	if (hw_slider[id]==value)
		return 0;

	/* update lcd pos */
	int moved = hw_slider[id]-value;

	hw_slider[id]=value;

	lcd_show_slice(value);

	/* update shadow */
	hw_sendAWSCommand(HW_SLIDER_AWS_ID,id,value);

	/* updte volume */
	sound_volume_slider(id,value);

		return moved;

}

/* check the name format accept */
bool check_is_aws_name(string prefix,int id, string name ){
	prefix.append(to_string(id));
	return prefix.compare(name)==0;

}


/* received and process AWS shadows update request */
int hw_aws_sync(JSON* json){

	map<string,JSON>* m = json->value.m;

	 APP_INFO(( "\r\n hw_aws_sync m size %d",m->size() ));



	int updated = 0;
	bool exit = false;

	for(map<string,JSON>::iterator it=m->begin();it!=m->end();it++){

		string name = it->first;
		string value = *(it->second.value.s);

		APP_INFO(( "\r\n hw_aws_syncing %s - %s",name.c_str(),value.c_str() ));

		for(int i=0;i<NUM_OF_BUTTON;i++){
			if (check_is_aws_name(HW_BUTTON_AWS_ID,i,name)){
				if (hw_set_button(i,stoi(value)) != 0)
						updated++;
				exit=true;
				break;
			}
		}

		if (exit)
			continue;

		for(int i=0;i<NUM_OF_SLIDER;i++){
			if (check_is_aws_name(HW_SLIDER_AWS_ID,i,name)){
				if (hw_set_slider(i,stoi(value)) != 0)
							updated++;

				break;

			}

		}

	}
	hw_aws_publish = true;

	return updated;

}
