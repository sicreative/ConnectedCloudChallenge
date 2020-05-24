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
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, layers
import features

from yamnet import _YAMNET_LAYER_DEFS


def _yamnet(features):
  """Define the core YAMNet mode in Keras."""
  net = layers.Reshape(
    (params.PATCH_FRAMES, params.PATCH_BANDS, 1),
    input_shape=(params.PATCH_FRAMES, params.PATCH_BANDS))(features)
  for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
    net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters)(net)
  net = layers.GlobalAveragePooling2D()(net)
  logits = layers.Dense(units=params.NUM_CLASSES, use_bias=True)(net)
  predictions = layers.Activation(
    name=params.EXAMPLE_PREDICTIONS_LAYER_NAME,
    activation=params.CLASSIFIER_ACTIVATION)(logits)
  return predictions, net


def _yamnet_frames_model(feature_params):
  """Defines the YAMNet waveform-to-class-scores model.
  Args:
    feature_params: An object with parameter fields to control the feature
    calculation.
  Returns:
    A model accepting (1, num_samples) waveform input and emitting a
    (num_patches, num_classes) matrix of class scores per time frame as
    well as a (num_spectrogram_frames, num_mel_bins) spectrogram feature
    matrix.
  """
  waveform = layers.Input(batch_shape=(1, None))
  # Store the intermediate spectrogram features to use in visualization.
  spectrogram = features.waveform_to_log_mel_spectrogram(
    tf.squeeze(waveform, axis=0), feature_params)
  patches = features.spectrogram_to_patches(spectrogram, feature_params)
  predictions, net = _yamnet(patches)
  frames_model = Model(name='yamnet_frames',
                       inputs=waveform, outputs=[predictions, spectrogram, net, patches])
  return frames_model, net

def sendResultToIoT(dogscore,things,scores):
    iotclient = boto3.client('iot-data')
    
    message = "{ \"requests\":\"finish\",\"dogscore\":\""+str(dogscore)+"\",\"classtype\":{\"angrydog\":\""+str(scores[0])+"\",\"other\":\""+str(scores[1])+"\"}}"
    
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

    if dogscore < 0.8 :
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
  
  yamnet,_ = _yamnet_frames_model(params)
  yamnet.load_weights('yamnet.h5')

  angrydog_model = load_model('angrydog.h5') 
  
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

    _, _,dense,_ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)

    scores = []      
    for patch in dense:
      score = angrydog_model.predict( np.expand_dims(patch,0)).squeeze()
      scores.append(score)
    
    scores = np.mean(scores,axis=0)
    
    
    splited = soundkey.split('_')
    things=splited[1] 
    sendResultToIoT(scores[0],things,scores)


  
if __name__ == '__main__':
  main(sys.argv[1:])
