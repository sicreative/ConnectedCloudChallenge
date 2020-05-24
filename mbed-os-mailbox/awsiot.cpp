/*
 * awsiot.cpp
 *
 * For AWS IoT
 *
 * Author: sc lee
 *
 *  Licensed under the Apache License, Version 2.0
 */

#include "awsiot.h"


#define SHADOW_USE_JSON_MAP 0

/**
 *
 * Handling AWS Activities
 */
#include "mbed.h"
#include "aws_client.h"
#include "aws_config.h"
#include "lcd_ui.h"
#include "json_aws.h"
#include "hw.h"
#include "network_activity_handler.h"
#include "sound.h"
#include "base64.h"
#include "fram.h"
#include "time.h"
#include "us_ticker_api.h"
#include "sha256.h"
#include "https_request.h"
#include "mbedtls/debug.h"


#include <map>

NetworkInterface* _network;



// define working with AWS IOT shadow
#define SEND_SHADOW

#define AWSIOT_KEEPALIVE_TIMEOUT            (60)
#define AWS_IOT_SECURE_PORT                 (8883)
#define AWSIOT_TIMEOUT_IN_USEC              (1000UL * 1000UL)
#define AWSIOT_SEND_PERIOD_MS                700

AWSIoTClient *client = NULL;
aws_publish_params_t publish_params;
string uuid;
string passhash;

Thread awsiot_thread(osPriorityNormal, 8196 , NULL, "awsiot_thread");
Semaphore awsiot_sem(1);

#if SHADOW_USE_JSON_MAP
	map<string,JSON> shadow_status_map;
#else
#define SHADOW_STATUS_PREFIX 		"{ \"state\": { \"reported\" : {"
#define SHADOW_STATUS_SUFFIX    	"}}}"
	string shadow_status;
#endif


int awsiot_subscribe_shadow(char* command,subscriber_callback callback);
int awsiot_publish_shadow( const char* message,const char* command);
int awsiot_subscribe(const char* topic,subscriber_callback callback);
int awsiot_unsubscribe_shadow(char* command);
int awsiot_unsubscribe(const char* topic);
int awsiot_publish( const char* message,const char* topic);
void awsiot_update(string name, string value,bool publish);
void awsiot_connect_thread( NetworkInterface* network,bool retry );
int awsiot_publish_steam( const char* message,const char* command, int size);
void awsiot_connect_start_thread( NetworkInterface* network);



bool close_connect_signal = false;


/* AWSIOT Subscribe Callbacks */
void awsiot_subscribe_ai_callback( aws_iot_message_t& md){





	string pl = string((char*)md.message.payload,md.message.payloadlen);



	map<string,JSON>* m = json_parse(pl);

	if (m->find("requests") == m->end()){
			delete m;
			return;
	}

	string requests = *((*m)["requests"].value).s;

	if ( !requests.compare("finish")){
		if (m->find("dogscore") == m->end()){
					delete m;
					return;
		}



		string dogscore = *((*m)["dogscore"].value).s;



		if (m->find("classtype") == m->end()){
				delete m;
				return;
		}



		map<string,JSON>* classtype = ((*m)["classtype"].value).m;



		for (int i=0;i<2;i++){
			string classtypetext;
			float value = 0;
			for(map<string,JSON>::iterator it = classtype->begin();it!=classtype->end();it++){
				if (stof(*(it->second.value.s))>value){
					classtypetext=it->first;
					value=stof(*(it->second.value.s));

				}

			}

			classtype->erase(classtypetext);

			lcd_msg("%s:%3f",i,classtypetext.c_str(),value);
		}

		//APP_INFO(("\nAWS AI classtype: %s\n",classtypetext.c_str()));





	}else if (!requests.compare("alarm")){
		char base64buffer[1536];
		size_t len;

		int section = stoi(*((*m)["section"].value).s);
		int totalsection = stoi(*((*m)["totalsection"].value).s);

		string data = *((*m)["data"].value).s;

		char c_data[data.size() + 1];

		strcpy(c_data, data.c_str());

        int result = mbedtls_base64_decode((unsigned char*)base64buffer,sizeof(base64buffer),&len,(const unsigned char*)c_data,sizeof(c_data)-1);

        if (result==0)
        	sound_steam_input(section,totalsection,base64buffer);
        else
        	APP_INFO(("base64 decode error:%d\n",result));





	}



	//APP_INFO(("\nAWS AI:\n%s\n",pl.c_str()));

	delete m;

}


/* AWSIOT Subscribe Callbacks */
void awsiot_subscribe_transcript_callback( aws_iot_message_t& md){

	lcd_msg("AWS Transcript",0);

	string pl = string((char*)md.message.payload,md.message.payloadlen);
	map<string,JSON>* m = json_parse(pl);

	if (m->find("requests") == m->end()){
			delete m;
			return;
	}

	string requests = *((*m)["requests"].value).s;;

	if ( !requests.compare("finish")){
		if (m->find("transcript") == m->end())
					return;

		string transcript = *((*m)["transcript"].value).s;


		if (transcript.size()>24){
			size_t sublength = transcript.size()>48?48:transcript.size()-24;
			string sub = transcript.substr(0,25);
			lcd_msg(sub.c_str(),1);

			sub = transcript.substr(25,sublength);
			lcd_msg(sub.c_str(),2);

		}else{
			lcd_msg(transcript.c_str(),1);
		}


	}

	delete m;

}

void awsiot_subscribe_productmetadata_callback( aws_iot_message_t& md){

	string pl = string((char*)md.message.payload,md.message.payloadlen);




	APP_INFO(("\npayload\n%s\n",pl.c_str()));

	map<string,JSON>* m = json_parse(pl);



	/* no job if no requests inside json */
	if (m->find("requests") == m->end())
		return;


	string requests = *((*m)["requests"].value).s;


	if ( !requests.compare("ios_pairing")){

		 string dsn = *((*m)["dsn"].value).s;
		if (strcmp(dsn.c_str(),DSN)!=0)
			return;


		char codeverifier[9];



		mbedtls_ctr_drbg_context ctx_drbg;
		mbedtls_entropy_context entropy;

		mbedtls_entropy_init( &entropy );

		mbedtls_ctr_drbg_init(&ctx_drbg);

		 char * pers = "generatekey_123";


		 mbedtls_ctr_drbg_seed( &ctx_drbg, mbedtls_entropy_func, &entropy,
		    (unsigned char *) pers, sizeof( pers ) );


		mbedtls_ctr_drbg_random((void*)&ctx_drbg,(unsigned char *)codeverifier,8);

		 for(int i=0;i<8;i++){

			 if (codeverifier[i]>='A' && codeverifier[i]<='Z')
				 continue;
			 if (codeverifier[i]>='a' && codeverifier[i]<='z')
			 	 continue;
			 if (codeverifier[i]>='0' && codeverifier[i]<='9')
			 	 continue;

			 codeverifier[i] += 23;
			 --i;

		 }

		 unsigned char hash256[32];


		 mbedtls_sha256_ret((const unsigned char*)codeverifier, 8, hash256, 0);

		 uuid = *((*m)["uuid"].value).s;


		 char base64buffer[64];
		 size_t len;


		 int result = mbedtls_base64_encode((unsigned char*)base64buffer,64,&len,hash256,32);


		 char productmetadata[256];
		 sprintf(productmetadata,"{\"DSN\":\"%s\",\"TOPIC\":\"%s\",\"HASH\":\"%s\",\"UUID\":\"%s\"}",DSN,AWSIOT_THING_NAME,base64buffer,uuid.c_str());




		 awsiot_publish(productmetadata,"mailbox/productmetadata/get");


		// lcd_msg_delay_close("IOS APP PAIRING CODE",0,20*1000);

		 codeverifier[8]='\0';

		 char codestring[10];
		 sprintf(codestring,"%s",codeverifier);

		 lcd_downside_msg_delay_close(codestring,20*1000);

		 APP_INFO(("\ncodeverifier: %s\n",codestring));

	}else if (!requests.compare("ios_paired")){

		 string dsn = *((*m)["dsn"].value).s;
				if (strcmp(dsn.c_str(),DSN)!=0)
					return;

	     string thisuuid = *((*m)["uuid"].value).s;
				if (uuid.compare(thisuuid)!=0)
				    return;

		 passhash = string(*((*m)["pass_hash"].value).s);



		 APP_INFO(("\nPASSHASH RECEIVED: %s\n",passhash.c_str()));

		 strcpy(status.pass_hash,passhash.c_str());
		 //status.pass_hash = passhash.c_str();
		 fram_write_status();

		 lcd_downside_msg("");

		 awsiot_update("hash", passhash);



	}


	delete m;



}



void awsiot_subscribe_update_callback( aws_iot_message_t& md){

	string pl = string((char*)md.message.payload,md.message.payloadlen);




	 map<string,JSON>* m = json_parse(pl);


	 	 	 JSON* state = &((*m)["state"]);
	 	 	 map<string,JSON>* sm = state->value.m;
	 	 	if ( sm->find("reported") != sm->end() ) {



	 	 	 JSON* reported = &((*sm)["reported"]);
	 	 	 hw_aws_sync(reported);

	 	 	}


	 	 	if ( sm->find("desired") != sm->end() ) {



	 	 			 	 	 JSON* desired = &((*sm)["desired"]);
	 	 			 	 	 if (hw_aws_sync(desired))
	 	 			 	 		lcd_msg_delay_close("Update Accepted",2,1000);
	 	 		}




	 	 	delete m;



}





void awsiot_subscribe_get_callback( aws_iot_message_t& md){

	string pl = string((char*)md.message.payload,md.message.payloadlen);
	 aws_message_t &message = md.message;
	 APP_INFO(( "\r\nMessage arrived: qos %d, retained %d, dup %d, packetid %d", message.qos, message.retained, message.dup, message.id ));
     APP_INFO(( "\r\nPayload %.*s\r\n", message.payloadlen, (char*)message.payload ));
	 map<string,JSON>* m = json_parse(pl);
	 JSON* state = &((*m)["state"]);
	 map<string,JSON>* sm = state->value.m;
		 	 	if ( sm->find("reported") != sm->end() ) {

					 JSON* reported = &((*sm)["reported"]);
					 hw_aws_sync(reported);

		 	 	}

		 		delete m;
}


/* loop for AWSIOT  publish / steam, Shadows subscribe listening */
/* All MQTT active within this loop */

int awsiot_shadow_cycle(){

	/* init get all AWS shadow data */

	awsiot_publish_shadow("","/get");
	lcd_msg("AWS shadow working",0);

	/* entry forever loop */
while(true){


	/* publish waiting shadow update */

#if SHADOW_USE_JSON_MAP



		if (!shadow_status_map.empty()){

			awsiot_update(string name, string value,bool publish)

			awsiot_sem.acquire();

		string message = json_writer(&shadow_status_map);
		int result = awsiot_publish_shadow(message.c_str(),"/update");

		if (result==1)
			shadow_status_map.clear();

		awsiot_sem.release();

		}

#else
		if (!shadow_status.empty()){

			awsiot_sem.acquire();
			shadow_status.insert(0,SHADOW_STATUS_PREFIX);
			shadow_status.append(SHADOW_STATUS_SUFFIX);
			int result = awsiot_publish_shadow(shadow_status.c_str(),"/update");

			if (result==1)
						shadow_status.clear();
			awsiot_sem.release();


		}
# endif

	/* enquire for any waiting sound steam and work for that */
	int num_of_sound_steam = sound_num_wait_for_steam();
	if (num_of_sound_steam>0){

		int buffer_size = SOUND_STEAM_BUFFER_SIZE*128+4+1;

		int base64buffer_size = buffer_size*2*4/3+16;
		/* buffer after BASE64 */
		char base64buffer[base64buffer_size];

		/* AWS IoT Rules */
		char rules[64];
		sprintf (rules,"$aws/rules/Voice/%s/sound/send",AWSIOT_THING_NAME);

		int total_length = num_of_sound_steam/SOUND_STEAM_BUFFER_SIZE;

		APP_INFO(( "Start steam %d, section %d\n\r", num_of_sound_steam,total_length ));

		time_t last_time = time(NULL);
		int count = 0;

		/*Loop for send MQTT message, sample dismiss last if size not alignment */
		while(sound_steam(sound_buffer)>=SOUND_STEAM_BUFFER_SIZE){


			/* time stamp */
			time_t current_time = time(NULL);

			if(current_time==last_time)
				++count;
			else{
				last_time = current_time;
				count=0;
			}

			current_time = current_time*1000+count;

			time_t now = current_time;


			/* write time_t to last 8 char */
			for(int i=buffer_size*2-1;i>buffer_size*2-9;i--){
				((char*)sound_buffer)[i]= current_time & 0xff;
				current_time >>= 8;

			}

			/* write totallength to pos: len-5 */
			sound_buffer[buffer_size-5]=total_length;


			/* len after process BASE64 */
			size_t len = 0;

			int result = mbedtls_base64_encode((unsigned char*)base64buffer,base64buffer_size,&len,(const unsigned char*)sound_buffer,buffer_size*2);

			if (result == MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL){
					APP_INFO(( "BASE64 Error %d\r\n",len ));
			}

			/* publish the steam */
			awsiot_publish_steam(base64buffer,rules,len);

			APP_INFO(("%" PRId64 "\n",now ));

			wait_us(50*1000);

		}
	}

	/* yield for listening subscribe */

	cy_rslt_t result = client->yield();

	if ( result == CY_RSLT_AWS_ERROR_DISCONNECTED ){

		APP_INFO(( "Yield Error\n" ));
			if (_network->get_connection_status()!=NSAPI_STATUS_GLOBAL_UP){
				_network->disconnect();
				_network->emacInterface()->get_emac().power_down();
				_network->emacInterface()->get_emac().power_up();
				_network = NULL;
			}
			awsiot_connect_thread(_network,true);

	}


		wait_us(AWSIOT_SEND_PERIOD_MS*1000);

	}


}


/* AWS IoT init connect */
void awsiot_connect_thread( NetworkInterface* network,bool retry )
{

	aws_connect_params_t conn_params;
	aws_endpoint_params_t endpoint_params;

	while (!network){

		network_connect();

	}


	if (client!=NULL){

		if (network->get_connection_status()==NSAPI_STATUS_DISCONNECTED)
			network->connect();

		client->disconnect();
		retry=true;
		delete(client);
	    client = NULL;
	}


    conn_params = { 0, 0, NULL, NULL, NULL, NULL, NULL };
    publish_params = { AWS_QOS_ATMOST_ONCE };
    endpoint_params = { AWS_TRANSPORT_MQTT_NATIVE, NULL, 0, NULL, 0 };

    cy_rslt_t result = CY_RSLT_SUCCESS;

    _network = network;

    passhash = string(status.pass_hash);

    if ( ( strlen(SSL_CLIENTKEY_PEM) | strlen(SSL_CLIENTCERT_PEM) | strlen(SSL_CA_PEM) ) < 64 )
    {
    	   lcd_msg("aws_config.h error",0);

        APP_INFO(( "Please configure SSL_CLIENTKEY_PEM, SSL_CLIENTCERT_PEM and SSL_CA_PEM in aws_config.h file\n" ));
        return ;
    }

    /* Initialize AWS Client library */
    client = new AWSIoTClient( network, AWSIOT_THING_NAME, SSL_CLIENTKEY_PEM, strlen(SSL_CLIENTKEY_PEM), SSL_CLIENTCERT_PEM, strlen(SSL_CLIENTCERT_PEM) );

    /* set MQTT endpoint parameters */
    endpoint_params.transport = AWS_TRANSPORT_MQTT_NATIVE;
    endpoint_params.uri = (char *)AWSIOT_ENDPOINT_ADDRESS;
    endpoint_params.port = AWS_IOT_SECURE_PORT;
    endpoint_params.root_ca = SSL_CA_PEM;
    endpoint_params.root_ca_length = strlen(SSL_CA_PEM);

    /* set MQTT connection parameters */
    conn_params.username = NULL;
    conn_params.password = NULL;
    conn_params.keep_alive = AWSIOT_KEEPALIVE_TIMEOUT;
    conn_params.peer_cn = (uint8_t*)AWSIOT_ENDPOINT_ADDRESS;
    conn_params.client_id = (uint8_t*)AWSIOT_THING_NAME;

    /* connect to an AWS endpoint */
    result = client->connect( conn_params, endpoint_params );
    if ( result != CY_RSLT_SUCCESS )
    {
    	lcd_msg("connection AWS failed %d",0,result);
        APP_INFO(( "connection to AWS endpoint failed\r\n" ));
        if( client != NULL )
        {
            delete client;
            client = NULL;
        }
        return ;
    }

    lcd_msg("connected to AWS: %s",0,AWSIOT_THING_NAME);
    APP_INFO(( "Connected to AWS endpoint\r\n" ));


    /* subscribe get for first time get all shadow data */
    if(!retry){
    	awsiot_subscribe_shadow("/get/accepted",awsiot_subscribe_get_callback);

    }

    /* subscribe update for shadow  */
	awsiot_subscribe_shadow("/update/accepted",awsiot_subscribe_update_callback);

	/* subscribe share channel for product pairing */
	awsiot_subscribe("mailbox/productmetadata",awsiot_subscribe_productmetadata_callback);

	/* subscribe private transcript for received speech recognized text  */
	char topic_transcript[32];
	sprintf(topic_transcript,"%s/transcript/get",AWSIOT_THING_NAME);

	awsiot_subscribe(topic_transcript,awsiot_subscribe_transcript_callback);

	/* subscribe private AI for received sound classified */
	char ai_transcript[32];
	sprintf(ai_transcript,"%s/ai/get",AWSIOT_THING_NAME);

	awsiot_subscribe(ai_transcript,awsiot_subscribe_ai_callback);


#ifdef SEND_SHADOW
if (!retry)
    awsiot_shadow_cycle();
#endif

    return;
}

/* thread function for AWSIOT */
void awsiot_connect_start_thread( NetworkInterface* network){

	awsiot_connect_thread(network,false);

	return;
}

/* call for start AWSIOT thread */
int awsiot_connect( NetworkInterface* network ){

	awsiot_thread.start(callback(&awsiot_connect_start_thread,network));

}

/* MQTT subscribe */
int awsiot_subscribe(const char* topic,subscriber_callback callback){
    cy_rslt_t result = CY_RSLT_SUCCESS;


    if (client==NULL)
    	return 1;


    result = client->subscribe(topic, AWS_QOS_ATMOST_ONCE, callback );

    if ( result != CY_RSLT_SUCCESS )
	{
	   lcd_msg("subscribe to AWS failed %d ",2,result);
	   APP_INFO(( "subscribe to topic failed\r\n" ));

	   return 1;
	}

	APP_INFO(( "subscribe to %s \n",topic ));

	return 0;


}

/* MQTT unsubscribe */
int awsiot_unsubscribe(const char* topic){
	cy_rslt_t result = client->unsubscribe((char*)topic);


    if ( result != CY_RSLT_SUCCESS )
    {
 	   lcd_msg("unsubscribe to AWS failed %d ",2,result);
        APP_INFO(( "unsubscribe to topic failed\r\n" ));




        return 1;
    }


    APP_INFO(( "unsubscribe to %s \n",topic ));
    return 0;

}

/* AWS IoT shadow subscribe */
int awsiot_subscribe_shadow(char* command,subscriber_callback callback){

	char* temp = (char*)malloc(12+7+strlen(command)+strlen(AWSIOT_THING_NAME));
	temp[0]='\0';

	strcat(temp,"$aws/things/");
	strcat(temp,AWSIOT_THING_NAME);
	strcat(temp,"/shadow");
	strcat(temp,command);

	puts(temp);

	command = temp;

	APP_INFO(( "subscribe  %s \n",command ));

	return awsiot_subscribe(command,callback);

}

/* AWS IoT shadow unsubscribe */
int awsiot_unsubscribe_shadow(char* command){
	char* temp = (char*)malloc(12+7+strlen(command)+strlen(AWSIOT_THING_NAME));
	temp[0]='\0';

	strcat(temp,"$aws/things/");
	strcat(temp,AWSIOT_THING_NAME);
	strcat(temp,"/shadow");
	strcat(temp,command);

	puts(temp);

	command = temp;

	awsiot_unsubscribe(temp);

	free(temp);

}

/* AWS IoT publish */
int awsiot_publish( const char* message,const char* topic){
    cy_rslt_t result = CY_RSLT_SUCCESS;
    publish_params.QoS = AWS_QOS_ATMOST_ONCE;

    if (client==NULL)
    	return 1;

   result = client->publish( topic, message, strlen(message), publish_params );
   if ( result != CY_RSLT_SUCCESS )
   {
	   lcd_msg("publish failed",2);
	   APP_INFO(( "publish to topic failed, %d \r\n",result ));
	   if (_network->get_connection_status() != NSAPI_STATUS_GLOBAL_UP){

		   _network->disconnect();
		   _network = NULL;
	   }

	   awsiot_connect_thread(_network,true);

	   return -1;


   }



   lcd_msg_delay_close("published to AWS",2,1000);
   APP_INFO(( "Published to %s - %s",topic,message ));

   return 1;




}

int awsiot_publish( const char* message){
	return awsiot_publish(message,AWSIOT_TOPIC);
}


int awsiot_publish_steam( const char* message, const char* topic,int size){


	cy_rslt_t result = client->publish( topic, message, size, publish_params );

	if ( result != CY_RSLT_SUCCESS )
	   {
		   lcd_msg("publish failed",2);
		   APP_INFO(( "publish to topic failed, %d %d\r\n",result,size ));
		   if (_network->get_connection_status() != NSAPI_STATUS_GLOBAL_UP){

			   _network->disconnect();
			   _network = NULL;
		   }
	   return -1;
	   }

	return result;
}

int awsiot_publish_shadow( const char* message,const char* command){

	char* temp =  (char*)malloc(12+7+strlen(command)+strlen(AWSIOT_THING_NAME));
	temp[0]='\0';
	strcat(temp,"$aws/things/");
	strcat(temp,AWSIOT_THING_NAME);
	strcat(temp,"/shadow");
	strcat(temp,command);

	int result = awsiot_publish(message,temp);
	free(temp);

	if (result!=1)
	{
		APP_INFO(( "Published shadow failure retry\n" ));
		return result;
	}

	APP_INFO(( "Published to shadow %s\n. topic: %s\n",message,temp ));
	lcd_msg_delay_close("AWS shadow published",2,1000);

	return result;
}

/* internal function for shadow update in map or string */
void awsiot_update(string name, string value,bool publish){


	if (!publish)
		return;

	awsiot_sem.try_acquire();

	/* store update data under shadow_status_map for waiting published to AWS*/

#if SHADOW_USE_JSON_MAP

	if (!shadow_status_map["state"].value.m){
		//struct JSON state;
		shadow_status_map["state"].value.m = new map<string,JSON>;
		shadow_status_map["state"].type = AWS_JSON_CHILD;

	}



	if (!(*(shadow_status_map["state"].value.m))["reported"].value.m){

		(*(shadow_status_map["state"].value.m))["reported"].value.m = new map<string,JSON>;
		(*(shadow_status_map["state"].value.m))["reported"].type = AWS_JSON_CHILD;

		}


	 (*(*(shadow_status_map["state"].value.m))["reported"].value.m)[name].type = AWS_JSON_STRING;
	 (*(*(shadow_status_map["state"].value.m))["reported"].value.m)[name].value.s = new string(value);
#else

	 if (!shadow_status.empty())
		 shadow_status.append(",");

	 shadow_status.append("\"");
	 shadow_status.append(name);
	 shadow_status.append("\":\"");
	 shadow_status.append(value);
	 shadow_status.append("\"");



#endif

	awsiot_sem.release();

	APP_INFO(( "Waiting Published to shadow %s. hw: %s\n",value.c_str(),name.c_str() ));


}

/* convenient update class for shadow require post */
void awsiot_update(string name, string value){
	awsiot_update(name,value,true);
}







