import base64
import boto3
from datetime import datetime
import time
import pickle
import urllib.request
import json

#  AWS Lambda functions for transfer Kinesis to Transcription and send back message though AWS IoT


TEMP_S3_INDEX = "income/sound_temp_index.tmp_"
TEMP_S3_LAST_SOUND_KEY = "income/sound_temp_last_key.tmp_"
NEXT_FILE_SECOND = 2
BATCH_SIZE = 6
RECORD_SIZE = BATCH_SIZE * 256
SAMPLE_SIZE = RECORD_SIZE + 10;

s3_client = boto3.client('s3')
s3 = boto3.resource('s3')


def sendTranscriptToIoT(transcript,things):
    iotclient = boto3.client('iot-data')
    
    message = "{ \"requests\":\"finish\",\"transcript\":\""+ transcript + "\"}"
    
 
    
    try:
        transcripttopic = things+'/transcript/get'
        response = iotclient.publish(
            topic=transcripttopic,
            qos=0,
            payload=message
            )
            
        print("published to:",transcripttopic,message,"response:",response)    
    except IoTDataPlane.Client.exceptions.UnauthorizedException:
        print ("UnauthorizedException")
        
    


def sendTranscription(status,things):
    print("finish transcription")
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['FAILED']:
        print("Error transcription")
        return
# read https transcripted file obtained from transcription result     
    resulturl =  status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    with urllib.request.urlopen(resulturl) as response:
        html = response.read().decode('utf-8')
        print("transcribe:",html)
        transcripts = json.loads(html)
        transcript = ""
        for trans in transcripts['results']['transcripts']:
            transcript += trans['transcript']
            transcript += ' '
        
        transcript = transcript[:len(transcript)-1]
        
        print("transcribe:",transcript)
        sendTranscriptToIoT(transcript,things)
    

def lambda_handler(event, context):

    for record in event['Records']:
    # temp var for each record
        partitionkey = record["kinesis"]["partitionKey"]
        things = partitionkey[:len(partitionkey)-11]
        
        
        soundclip = bytearray()
        key = ''
    
        soundclip_index = []
    
        soundclip_obj:s3.Object
    
        s3soundlast_sound_key_obj:s3.Object
        s3soundclip_index_obj:s3.Object
    
    # check or build relative temp file 
        try:
            s3soundlast_sound_key_obj = s3.Object('voicerecognise',TEMP_S3_LAST_SOUND_KEY+things)
            s3soundlast_sound_key = s3soundlast_sound_key_obj.get()['Body'].read()
        
            key = pickle.loads(s3soundlast_sound_key)
        
            soundclip_obj = s3.Object('voicerecognise',key)
            soundclip = bytearray(soundclip_obj.get()['Body'].read())
        
            print ("num of record",len(soundclip)/RECORD_SIZE)
        
        except s3_client.exceptions.NoSuchKey as e:
            print ("no previous temp soundclip found")
    
      
    
        try:    
            s3soundclip_index_obj = s3.Object('voicerecognise',TEMP_S3_INDEX+things)
            s3soundclip_index = s3soundclip_index_obj.get()['Body'].read()
            soundclip_index = pickle.loads (s3soundclip_index)
        
        except s3_client.exceptions.NoSuchKey as e:
            print ("no previous index found")
        
        print("index len",len(soundclip_index),soundclip_index)  
        
        
        
        
        #Kinesis data is base64 encoded so decode here
        payload=base64.b64decode(record["kinesis"]["data"])
        #Another decord for MQTT
        payload=base64.b64decode(payload)
        recordsize = len(payload);
       #Check sample size match
       
        if recordsize != SAMPLE_SIZE:
            print( "recordsize mismatch %d %d",recordsize,SAMPLE_SIZE)
            break
        
         
        
       # timestamp ms decorded from back of payload      
        stamp = int.from_bytes(payload[SAMPLE_SIZE-8:], byteorder='big',signed='true')
        
      # decode total num of block    
        total_num = int.from_bytes(payload[SAMPLE_SIZE-10:SAMPLE_SIZE-8], byteorder='little',signed='true')
        
        print( "Decoded time",stamp,"size:",len(payload),"total_num",total_num," payload: " + str(payload))
        
 
        
       # if the simple timestamp more that 2.5s the lastest index, simple build new file  
        if len(soundclip_index)>0 :
            last = soundclip_index[len(soundclip_index)-1]
            if stamp - last > 2500:
                key = ''
                soundclip_index.clear()
                soundclip = bytearray()
                
                
       # Dismiss process if soundclip length more that totalnum
       
        length = len(soundclip)/RECORD_SIZE
        
        if total_num < length :
            continue        
        
        timestamp = datetime.fromtimestamp(stamp/1000)
        
        
        indexLen = len(soundclip_index)
        
        for i in range(indexLen-1, -1, -1):
            if soundclip_index[i] > stamp:
                continue
            # insert sound data on soundclip 
            soundclip[(i+1)*RECORD_SIZE:(i+1)*RECORD_SIZE] = payload[0:RECORD_SIZE]
            # insert timestamp in index
            soundclip_index.insert(i+1,stamp)
            break
        
        # No insert, which is most early sound clip. append 
        if indexLen == len(soundclip_index):
            soundclip[0:0] = payload[0:RECORD_SIZE]
            soundclip_index.insert(0,stamp)
       
        length = len(soundclip)/RECORD_SIZE
    
        print("Num soundclip: ",len(soundclip)/RECORD_SIZE)
    
    
    
    
    
    	# continue next if no block  
        if len(soundclip_index) == 0:
            continue;
    
	# For first section, establish the name of file 
        if key=='' :
            key = "income/sound_" + things + "_%m%d%Y-%H%M%S.wav"
            key = datetime.fromtimestamp(soundclip_index[0]/1000).strftime(key)
            s3soundlast_sound_key_obj.put(Body=pickle.dumps(key))
            soundclip_obj = s3.Object('voicerecognise',key)
        
        s3soundclip_index_obj.put(Body=pickle.dumps(soundclip_index))
    
    
    # Update the soundclip to S3 Storage
    
        soundclip_obj.put(Body=soundclip)   
    
    # check  all steam arrived 
        if total_num == length:
	# add wav header to PCM soundclip 16k 16bit MONO
            soundclip[0:0]=bytes.fromhex('52494646')
            soundclip[4:4]=(36+total_num*RECORD_SIZE).to_bytes(4,byteorder='little');
            soundclip[8:8]=bytes.fromhex('57415645666d7420')
            soundclip[16:16]=(16).to_bytes(4,byteorder='little')
            soundclip[20:20]=(1).to_bytes(2,byteorder='little')
            soundclip[22:22]=(1).to_bytes(2,byteorder='little')
            soundclip[24:24]=(16000).to_bytes(4,byteorder='little')
            soundclip[28:28]=(32000).to_bytes(4,byteorder='little')
            soundclip[32:32]=(2).to_bytes(2,byteorder='little')
            soundclip[34:34]=(16).to_bytes(2,byteorder='little')
            soundclip[36:36]=bytes.fromhex('64617461')
            soundclip[40:40]=(total_num*RECORD_SIZE).to_bytes(4,byteorder='little')
        
            soundclip_obj.put(Body=soundclip)      
        # Build and start transcribe job
            transcribe = boto3.client('transcribe')
            job_name = key.replace("income/","")
     
            job_uri = "https://voicerecognise.s3-us-west-2.amazonaws.com/income/"+job_name
            transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': job_uri},
                MediaFormat='wav',
                LanguageCode='en-US'
                )
	# Wait for transcript complete
            while True:
                status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
                if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                print("Not ready yet...")
                time.sleep(5)
	# call for afterward process 
            sendTranscription(status,things)
        
