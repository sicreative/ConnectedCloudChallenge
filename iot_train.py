import soundfile as sf
import resampy
import params
import yamnet
import features
import numpy as np
import random

import boto3
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from yamnet import _YAMNET_LAYER_DEFS

import tensorflow as tf
print("tf version: ", tf.__version__)
print("tf.keras version: ", tf.keras.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

samples = []
classtypes = []
classtypesindex = []

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
                       inputs=waveform, outputs=[predictions, spectrogram, net])
  return frames_model




def load_wav(filename):
  wav,sr = sf.read(filename,dtype=np.int16)
  if len(wav) > 441000:
     wav = wav[:441000]

  if len(wav.shape) > 1:
    wav = np.mean(wav, axis=1) 
 
  if sr != 16000:
    wav = resampy.resample(wav,sr,16000)

  wav = wav / 32768.0
     
  return wav.astype(np.float64)


def random_section(wav):
  length = len(wav)  
  start = int(np.random.uniform()*length*0.5)
  end = int(np.random.uniform()*length*0.5)
  if (start>end):
    temp = start
    start = end
    end = temp
  return start,end

def augment(wav):
    
  wav = wav.copy() 
  start,end = random_section(wav)
 
  # volume 
  volume = np.random.uniform(0.7, 1.3)
  wav[start:end] = wav[start:end] * volume    
  # sample 
  if np.random.uniform() > 0.75:  
    delta = int(16000 * np.random.uniform(0.92, 1.08))
    wav = resampy.resample(wav, 16000, delta)
  # noise
  if np.random.uniform() > 0.75:
    noise = np.random.uniform() * 0.002 
    wav += np.random.uniform(-noise, noise, size=wav.shape)
  #  shift  
  if np.random.uniform() > 0.75:
    start,end = random_section(wav)
    wav[start:end] = wav[end:end+(end-start)]
    
  return wav  

def load_dataset(classtype,test):
  index = len(classtypes)
  s3 = boto3.client('s3')
  bucket = 'soundsampletest' if test else 'soundsample' 
  lists = s3.list_objects_v2(Bucket=bucket,Prefix=classtype)
  for list in lists['Contents']: 
    if list['Size']==0:
      continue
    key = list['Key']
    s3.download_file(bucket,key,'train.wav')
    wavform = load_wav('train.wav')
    for i in range (3):    
      wav = wavform.copy()
      if i > 0 :
        wav = augment(wav)
      _,_, dense = yamnetmodel.predict(np.reshape(wav, [1, -1]),steps=1)
      for patch in dense:
        samples.append(patch)
        classtypesindex.append(index)
    print(key)

  classtypes.append(classtype)

def shuffle():
  global classtypesindex
  global samples
  index = list(range(len(classtypesindex)))
  random.shuffle(index)

  samples = [samples[i] for i in index]
  classtypesindex = [classtypesindex[i] for i in index]

  samples = np.array(samples)
  classtypesindex = np.array(classtypesindex)



yamnetmodel = _yamnet_frames_model(params)
yamnetmodel.load_weights('yamnet.h5')
class_names = yamnet.class_names('yamnet_class_map.csv')
load_dataset("angrydog",False)
load_dataset("other",False)
shuffle()

print(" Loaded samples: " , samples.shape, samples.dtype,  classtypesindex.shape)

input_layer = layers.Input(shape=(1024,))
output = layers.Dense(1024, activation=None)(input_layer)
output = layers.Dense(2, activation='softmax')(output)
model = Model(inputs=input_layer, outputs=output)
opt = SGD(lr=0.002, decay=1e-5, momentum=0.8, nesterov=True)   
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(samples, classtypesindex, epochs=160, validation_split=0.200)
history = model.fit(samples, classtypesindex, epochs=5)

samples = []
classtypesindex = []
classtypes = []

load_dataset("angrydog",True)
load_dataset("other",True)
shuffle()
test_mse_score, test_mae_score = model.evaluate(samples,classtypesindex)


print ("Test Mse: {}, Test Mae: {}",test_mse_score,test_mae_score)


model.save("angrydog.h5", include_optimizer=False)
