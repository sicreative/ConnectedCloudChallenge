# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import boto3
import base64
import time
import params
import yamnet as yamnet_model
import csv



def sendResultToIoT(dogscore,classtype,things):
    iotclient = boto3.client('iot-data')
    
    message = "{ \"requests\":\"finish\",\"dogscore\":\""+str(dogscore)+"\",\"classtype\":"+ classtype + "}"
    
 
    
    try:
        aitopic = things+'/ai/get'
        response = iotclient.publish(
            topic=aitopic,
            qos=0,
            payload=message
            )
            
        print("published to:",aitopic,message,"response:",response)    
    except:
        print ("UnauthorizedException")

    if dogscore < 0.5 :
        return
    s3 = boto3.resource('s3')
    obj = s3.Object('voicerecognise','alarm.pcm')
    alarm = obj.get()['Body'].read()
    total_alarm_section = int(len(alarm)/1536)
    alarm_section = total_alarm_section
    while alarm_section:
        print(alarm_section,',')
        section_data = base64.b64encode(alarm[alarm_section*1536:(alarm_section+1)*1536]).decode("utf-8")
        message = "{ \"requests\":\"alarm\",\"section\":\""+str(alarm_section)+"\",\"totalsection\":\""+str(total_alarm_section)+"\",\"data\":\""+ section_data + "\"}"
        try:
            aitopic = things+'/ai/get'
            response = iotclient.publish(
               topic=aitopic,
               qos=0,
               payload=message
            )
        except:
            print ("UnauthorizedException")
        
        alarm_section-=1
        time.sleep(0.005)      
          


def class_id(class_map_csv):
  with open(class_map_csv) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)   # Skip header
    return np.array([int(index) for (index, _, _) in reader])


def main(argv):
  assert argv

  graph = tf.Graph()
  with graph.as_default():
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
  yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')
  yamnet_id = class_id('yamnet_class_map.csv')
  for soundkey in argv:

    # Download S3 document
 
    s3 = boto3.client('s3')
    s3.download_file('voicerecognise',soundkey,"sample.wav")	
    # Decode the WAV file.
    wav_data, sr = sf.read("sample.wav", dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != params.SAMPLE_RATE:
      waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

    # Predict YAMNet classes.
    # Second output is log-mel-spectrogram array (used for visualizations).
    # (steps=1 is a work around for Keras batching limitations.)
    with graph.as_default():
      scores, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top5_i = np.argsort(prediction)[::-1][:5]
    print(soundkey, ':\n' + 
          '\n'.join('{}:  {:12s}: {:.3f}'.format(yamnet_id[i],yamnet_classes[i], prediction[i])
                    for i in top5_i))
    	
    # Analysis  


    classtype = '{'+','.join('"{:s}":"{:.3f}"'.format(yamnet_classes[i], prediction[i])
		     for i in top5_i)+'}'
   

    dogscore = 0
    for i in top5_i:
      if yamnet_id[i] == 67:
        dogscore += prediction[i] * 0.25
      elif yamnet_id[i] == 68:
        dogscore+=prediction[i]*0.25
      elif yamnet_id[i] == 69:
        dogscore+=prediction[i]*0.7
      elif yamnet_id[i] == 79:
        dogscore+=prediction[i]*0.25
     # we hate cat
      elif yamnet_id[i] == 76:
        dogscore-=prediction[i]*0.25

    print('like_dog:',dogscore,'\n')


    splited = soundkey.split('_')
    things=splited[1]
    sendResultToIoT(dogscore,classtype,things)


  
if __name__ == '__main__':
  main(sys.argv[1:])
