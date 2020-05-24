/*
 * network.cpp
 *
 *
 *  Author: sc  lee
 *
 * Licensed under the Apache License, Version 2.0
 *
 */

#include "network.h"

#include "lcd_ui.h"
#include "mbed.h"
#include "whdstainterface.h"






WhdSTAInterface  *network_interface;


int32_t powersave_handler( WhdSTAInterface* wifi, wlan_powersave_mode_t mode )
{
    cy_rslt_t       result = CY_RSLT_SUCCESS;
    whd_interface_t ifp;
    whd_security_t  security_param = WHD_SECURITY_WPA2_MIXED_PSK;
    whd_bss_info_t  ap_info;

    /* Get the instance of the WLAN interface. This instance is used to obtain
       network parameters and to configure power-save mode of the device.
    */
    result = wifi->wifi_get_ifp( &ifp );
    if( CY_RSLT_SUCCESS == result )
    {
        /* The power-save handler can be called even before connecting to an AP
           to configure the power-save mode of WLAN device. But, the power-save
           mode can't be configured until the interface is powered. Hence, the
           check for ifp is performed here to ensure that the interface is
           powered and power-save mode can be successfully set.
        */
        if( ifp == NULL )
        {
            wifi->wifi_on( );
        }
        if( WHD_SUCCESS == wifi->is_interface_connected( ) )
        {
            result = whd_wifi_get_ap_info( ifp, &ap_info, &security_param );
            if( CY_RSLT_SUCCESS == result )
            {
                APP_INFO( ( "Beacon period = %d, DTIM period = %d\n",
                          ap_info.beacon_period, ap_info.dtim_period ) );
            }
            else
            {
            	APP_INFO( ( "Failed to get AP info.\r\n" ) );
            }
        }
        switch( mode )
        {
            case POWERSAVE_WITHOUT_THROUGHPUT:
                result = whd_wifi_enable_powersave( ifp );
                if( CY_RSLT_SUCCESS != result )
                {
                	APP_INFO( ( "Failed to enable PM1 mode\r\n" ) );
                }
                break;
            case POWERSAVE_WITH_THROUGHPUT:
                result = whd_wifi_enable_powersave_with_throughput( ifp, RETURN_TO_SLEEP_MS );
                if( CY_RSLT_SUCCESS != result )
                {
                	APP_INFO( ( "Failed to enable PM2 mode\r\n" ) );
                }
                break;
            case POWERSAVE_DISABLED:
                result = whd_wifi_disable_powersave( ifp );
                if( CY_RSLT_SUCCESS != result )
                {
                	APP_INFO( ( "Failed to disable powersave\r\n" ) );
                }
                break;
            default:
                APP_INFO( ( "Unknown mode\r\n" ) );
                result = UNKNOWN_MODE;
                break;
        }
    }
    else
    {
    	APP_INFO( ( "WiFi interface is not obtained.\r\n" ) );
    }

    return result;
}




void network_disconnect( void ){
	if (!network_interface)
		return;
	network_interface->disconnect();

	//network_interface = NULL;
	return;
}

void network_restart(){

	network_disconnect();
	network_interface->get_emac().power_down();
	network_interface->get_emac().power_up();
	network_connect();
}


WhdSTAInterface* get_network_interface(){
	return network_interface;
}

void initNetworkInterface(){
	network_interface = new WhdSTAInterface();



	if (!network_interface){
		APP_INFO(("ERROR: No WiFiInterface found.\n"));
		return;
	}




}


 void network_connect( void ){


	 if (!network_interface)
		 return;


    SocketAddress address;
    cy_rslt_t result = CY_RSLT_SUCCESS;

    APP_INFO(( "Connecting to the network using Wifi...\r\n" ));

    nsapi_error_t net_status = -1;
    for ( int tries = 0; tries < 3; tries++ )
    {

        int numofwifi = network_interface->scan(NULL, 0);

        if (numofwifi>10)
        	numofwifi=10;


        lcd_msg("num of wifi: %d",1,numofwifi);

        WiFiAccessPoint* wifipoint = new WiFiAccessPoint[numofwifi];

        network_interface->scan(wifipoint, numofwifi);

        for(int i=0;i<numofwifi;i++){
        	if (strcmp(wifipoint[i].get_ssid(),WIFI_SSID))
        			continue;

        	  network_interface->set_credentials(WIFI_SSID,WIFI_PASSWORD , NSAPI_SECURITY_WPA_WPA2);

        	  break;
        }


        net_status = network_interface->connect();
        if ( net_status == NSAPI_ERROR_OK )
        {

            break;
        }
        else
        {


            APP_INFO(( "Unable to connect to network. Retrying...\r\n" ));
        }
    }

    if ( net_status != NSAPI_ERROR_OK )
    {
    	lcd_msg("No Wifi, %d", net_status);
        APP_INFO(( "ERROR: Connecting to the network failed (%d)!\r\n", net_status ));
        network_interface = NULL;
        return ;
    }

    network_interface->get_ip_address(&address);
    lcd_msg("IP: %s ",1,address.get_ip_address() );
    APP_INFO(( "Connected to the network successfully. IP address: %s\n", address.get_ip_address() ));

	return ;

}
