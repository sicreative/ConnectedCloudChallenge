# Demo of Cypress CY8CKIT-062-WIFI-BT connect with AWS IoT ConnectedCloudChallenge AI Code

Licensed under the Apache License, Version 2.0
 
Demo for element14  Connected Cloud Challenge project

LED and ALS sensor for sense the mail.
Record Sound from TFT shield PDM MIC to on-board FRAM, stream via AWSIoT, pass the sound clip to SageMaker, run fine-tuning deep learning YAMNet to detect angry dog barking.
Use AWS IoT shadow connected with IOS APP

More detail: 
https://www.element14.com/community/community/design-challenges/connected-cloud-challenge-with-cypress-and-aws-iot/blog/2020/05/09/an-intelligent-mailbox-summary-of-the-challenge-11

## Quick Install 
Mbed and IOS reference of README.md under relative directory.


## Location
1. IOS source ./ios-mailbox
2. Mbed source ./mbed-os-mailbox

### 3. Lambra python source 
####ambda.py
Build new lambda function under AWS console and paste it, set relative kinesis stream trigger.
####test_sound_ai.py
Build a new lambda function under AWS console.
follow https://docs.aws.amazon.com/lambda/latest/dg/python-package.html
install python package requests and websocket

### 4. SageMaker Tensorflow
1. Under Amazon SageMaker, build a new Notebook instances
2. Git repositories https://github.com/tensorflow/models 
3. IAM add IoT, S3 access
4. Open JupyterLab
5. File > New > Terminal
6. source activate tensorflow_p36
7. conda install -c conda-forge resampy
8. conda install -c conda-forge pysoundfile
9. conda install -c conda-forge libsndfile
7. copy iot_inference_yamnet.py iot_inference.py iot_train.py and angrydog.h5 to /home/ec2-user/SageMaker/models/research/audioset/yamnet
8.  for fine tuning sound sample, copy dog and other folder to S3 Buckets soundsample, and copy inside test folder to S3 Buckets soundsampletest.
     Inside jupyterlab terminal      > cd /home/ec2-user/SageMaker/models/research/audioset/yamnet
                                                    > python iot_train.py




