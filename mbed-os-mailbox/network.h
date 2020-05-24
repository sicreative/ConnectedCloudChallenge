/*
 * network.h
 *
 * Handling Network activites
 *
 *      Author: sc lee
 *  Licensed under the Apache License, Version 2.0
 *
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "WhdSTAInterface.h"
#include "whd_wifi_api.h"
#include "whd_types.h"

#define UNKNOWN_MODE            (-1)

#define RETURN_TO_SLEEP_MS      (10u)




/**
 *
 * Set WIFI SSID and WPA2 password
 *
 */
#define WIFI_SSID "SSID"
#define WIFI_PASSWORD "PASSWORD"



/* This macro specifies the interval in milliseconds that the device monitors
   the network for inactivity. If the network is inactive for duration lesser
   than INACTIVE_WINDOW_MS in this interval, the MCU does not suspend the network
   stack and informs the calling function that the MCU wait period timed out
   while waiting for network to become inactive.
*/
#define INACTIVE_INTERVAL_MS    (300u)

/* This macro specifies the continuous duration in milliseconds for which the
   network has to be inactive. If the network is inactive for this duration, the
   MCU will suspend the network stack. Now, the MCU will not need to service the
   network timers which allows it to stay longer in sleep/deepsleep.
*/
#define INACTIVE_WINDOW_MS      (200u)

/* This macro defines the power-save mode that the WLAN device has to be
   configured to. The valid values for this macro are POWERSAVE_WITH_THROUGHPUT,
   POWERSAVE_WITHOUT_THROUGHPUT, POWERSAVE_WITH_THROUGHPUT, POWERSAVE_DISABLED
   which are defined in the enumeration wlan_powersave_mode_t.*/
#define POWERSAVE_SELECT        POWERSAVE_WITH_THROUGHPUT


/**
 *
 * Initialization the network Interface
 *
 */
void initNetworkInterface();

/**
 *
 * Connect to network
 *
 */
void network_connect( void );

/**
 *
 * Disconnect to network
 */
void network_disconnect( void );


void network_restart();

/**
 *
 *
 * Get Network Interface
 *
 * @return WhdSTAInterface* WHD WIFI Network interface
 *
 */
WhdSTAInterface* get_network_interface( void );

enum wlan_powersave_mode_t
{
    POWERSAVE_WITHOUT_THROUGHPUT,
    POWERSAVE_WITH_THROUGHPUT,
    POWERSAVE_DISABLED
};

int32_t powersave_handler( WhdSTAInterface* wifi, wlan_powersave_mode_t mode );



#endif /* NETWORK_H_ */
