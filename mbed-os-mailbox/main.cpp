/*
 *      Author: sc lee
 *
 *
 *      Demo of capture capsense button and publised data to AWS server
 *
 *      Inorder to run this demo, you should have CY8CKIT-062-WIFI-BT+TFT with CY8CKIT-028-TFT
 *
 *      Modifiy the aws_config.h before run
 *
 *		Detail:
 *
 *
 * Licensed under the Apache License, Version 2.0
 */




#include "mbed.h"
#include "lcd_ui.h"
#include "cy_pdl.h"
#include "cyhal.h"
#include "cybsp.h"
#include "GUI.h"
#include "network.h"
#include "awsiot.h"
#include "mbed_memory_status.h"
#include "capsense.h"
#include "network_activity_handler.h"
#include "NTPClient.h"
#include "time.h"



#include "hw.h"
#include "sound.h"
#include "fram.h"

/* SHOW HW STATISTIC */
#define HW_STATS 0

/* SLEEP WHEN NO NETWORK */
#define SUSPEND_NO_NETWORK 0



#define LOOP_MS 20
#define SENSOR_READ_MS 1024

/* Time zone */
#define TIME_ZONE_HOUR 8
#define TIME_ZONE_MIN 0

time_t lastdisplaytime;

/* update lcd real time clock */
void update_clock(){
	time_t now = time(NULL);

	double diff = difftime(now,lastdisplaytime);

	if (diff<1)
		return;

	lastdisplaytime = now;

	now+=TIME_ZONE_HOUR*60*60+TIME_ZONE_MIN*60;

	char buffer[32];
	strftime(buffer, 32, "%D %r\n", localtime(&now));

	lcd_msg("%s",-1,buffer);
}


#if (HW_STATS)
#if !defined(MBED_CPU_STATS_ENABLED) || !defined(DEVICE_LPTICKER) || !defined(DEVICE_SLEEP)
#error [NOT_SUPPORTED] test not supported
#endif


#define HW_STATS_MAX_THREAD_STACK        384
#define HW_STATS_SAMPLE_TIME             5000    // msec


uint64_t prev_idle_time = 0;
int32_t wait_time = 20000;


void calc_cpu_usage()
{
    mbed_stats_cpu_t stats;
    mbed_stats_cpu_get(&stats);

    uint64_t diff = (stats.idle_time - prev_idle_time);
    uint8_t idle = (diff * 100) / (HW_STATS_SAMPLE_TIME*1000);    // usec;
    uint8_t usage = 100 - ((diff * 100) / (HW_STATS_SAMPLE_TIME*1000));    // usec;;
    prev_idle_time = stats.idle_time;

    APP_INFO(("Idle: %d Usage: %d \n", idle, usage));


}


void print_memory_info() {
    // allocate enough room for every thread's stack statistics
    int cnt = osThreadGetCount();
    mbed_stats_stack_t *stats = (mbed_stats_stack_t*) malloc(cnt * sizeof(mbed_stats_stack_t));

    cnt = mbed_stats_stack_get_each(stats, cnt);
    for (int i = 0; i < cnt; i++) {
    	APP_INFO(("Thread: 0x%lX, Stack size: %lu / %lu\r\n", stats[i].thread_id, stats[i].max_size, stats[i].reserved_size));
    }
    free(stats);

    // Grab the heap statistics
    mbed_stats_heap_t heap_stats;
    mbed_stats_heap_get(&heap_stats);
    APP_INFO(("Heap size: %lu / %lu bytes\r\n", heap_stats.current_size, heap_stats.reserved_size));


    print_whd_bus_stats(get_network_interface());
}

void hw_stats(){
	calc_cpu_usage();
	print_memory_info();
}

void busy_thread()
{
    volatile uint64_t i = ~0;

    while(i--) {

    	Cy_SysLib_Delay(wait_time);
    }
}


#endif

// Blinking rate in milliseconds
#define BLINKING_RATE_MS     500



/* Call for idle the System when no network activate */
void power_save(){
#if (SUSPEND_NO_NETWORK)
	wait_net_suspend( get_network_interface(), osWaitForever,INACTIVE_INTERVAL_MS,INACTIVE_WINDOW_MS);
#endif
}



int main()
{

	 UART_OUTPUT = true;

	 fram_init();

	 fram_read_status();


#if (HW_STATS)

	   EventQueue *stats_queue = mbed_event_queue();
	    Thread *thread;
	    int id;

	    id = stats_queue->call_every(HW_STATS_SAMPLE_TIME, hw_stats);
	    thread = new Thread(osPriorityNormal, HW_STATS_MAX_THREAD_STACK);
	   // thread->start(busy_thread);
#endif





   initNetworkInterface();



   lcd_StartUp();


#if defined(USE_BMI160)

	    hw_bmi160_init();

#endif

   capsense_init();

   sound_init();


   lcd_msg("START DEVICES...",0);

   lcd_msg_delay_close("Mbed: %d.%d.%d",2,1000,MBED_MAJOR_VERSION, MBED_MINOR_VERSION, MBED_PATCH_VERSION);

   if (!get_network_interface())
   	 return 1;

   network_connect();




   lcd_msg("GET NTP Time...",0);
/*Get NTP Time */

   for(int i=0;i<5;i++){
	   NTPClient ntp(get_network_interface());

	  time_t current_time = ntp.get_timestamp();



	  if (current_time < 0 )
		  continue;

	  set_time(current_time);


	  break;


   }





  if (!awsiot_connect(get_network_interface()))
	  return -1;


  uint count = 0;

#if (SUSPEND_NO_NETWORK)
   Thread powersave_thread(osPriorityNormal, 4098 , NULL, "powersave_thread");



  powersave_thread.start(power_save);

#endif


  while(true){

	  RunCapSenseScan();

	  if (count%SENSOR_READ_MS==0)
		  sensor_read();

	  sound_update();

	  if (count%16==0)
		  update_clock();

	  count+=LOOP_MS;

	  wait_us(LOOP_MS*1000);


  }





}
