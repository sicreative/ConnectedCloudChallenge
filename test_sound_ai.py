import boto3
import time
import requests
import websocket

def lambda_handler(event, context):
    
    soundkey = event['soundkey'] 
    
    print(soundkey)
    
    sm_client = boto3.client('sagemaker')
#    url = sm_client.create_presigned_domain_url(DomainId='d-bvddkoijifh5',UserProfileName='iot-sound-dog')['AuthorizedUrl']
    url = sm_client.create_presigned_notebook_instance_url(NotebookInstanceName='iot-sound-yamnet')['AuthorizedUrl']

    url_tokens = url.split('/')
    http_proto = url_tokens[0]
    http_hn = url_tokens[2].split('?')[0].split('#')[0]


    
    s = requests.Session()
    r = s.get(url)
    cookies = "; ".join(key + "=" + value for key, value in s.cookies.items())

    ws = websocket.create_connection(
        "wss://{}/terminals/websocket/1".format(http_hn),
        cookie=cookies,
        host=http_hn,
        origin=http_proto + "//" + http_hn
    )
    
    ws.send("""[ "stdin", "source activate tensorflow_p36\\r" ]""")
       
    ws.send("""[ "stdin", "cd /home/ec2-user/SageMaker/models/research/audioset/yamnet\\r" ]""")
    
    
    
    ws.send("""[ "stdin", "python iot_inference.py """+soundkey+"""\\r" ]""")
    
   # time.sleep(2)
    
   # ws.send("""[ "stdin", "exit\\r" ]""")
    
    ws.close()
    return None  
