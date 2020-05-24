//
//  AWSIOT.swift
//  cypress_awsiot_two_way_demo
//
//  Created by slee on 10/4/2020.
//  Copyright © 2020 sclee. All rights reserved.
//

import Foundation

import AWSCognito
import AWSIoT
import AWSMobileClient
import CommonCrypto


var aws_iot_isConnected = false;

/* var for Device Things*/
var myThings = ""
/* paired passcode used for paired verify check*/
var paired_passcode:Data!

/* temp used for pairing process */
var auth_things : String!
var auth_HASH : String!


var get_finished = false;


/*view pointer*/
var viewController : ViewController!;
var viewButton0 : UISwitch!;
var viewButton1 : UISwitch!;
var viewSlider0 : UISlider!;
var username_label : UILabel!;
var log_label : UILabel!;
var envelope_view : UIImageView!;

/* condition of shadow */
var json_button_0 = false;
var json_button_1 = false;
var json_slider_0  = 0;
var json_log = "";
var json_envelope = false;

var username : String = "";


var sendJSON = Data();
var numofsend = 0;

var resend = 5;
var maxsend = resend*5;

/* prdouct serial no */
var dsn:String!
/* APP UUID */
var uuid:String!


/** function cacluate sha256 */
func sha256(data : Data) -> Data {
    var hash = [UInt8](repeating: 0,  count: Int(CC_SHA256_DIGEST_LENGTH))
    data.withUnsafeBytes {
        _ = CC_SHA256($0.baseAddress, CC_LONG(data.count), &hash)
    }
    return Data(hash)
}

/** function cacluate radom paired hash*/
func random_paired_hash() ->  String {
    var bytes = [UInt8](repeating: 0, count: 32)
    _ = SecRandomCopyBytes(kSecRandomDefault,bytes.count, &bytes)
    paired_passcode = Data(bytes)
    let hash = sha256(data: paired_passcode)
    return hash.base64EncodedString()
    
}



/** called when shadow get or update */
func update_item(message:Data){
    
    do{
     let json = try JSONSerialization.jsonObject(with: message )
     if let root = json as? [String:AnyObject] {
        if let state = root["state"] as? [String:AnyObject] {
            if let reported = state["reported"] as? [String:AnyObject] {
                if let shadow_hash = reported["hash"] as? String{
                    /* check paired_passcode exist and shadows hash */
                    if (paired_passcode==nil || shadow_hash != sha256(data:paired_passcode).base64EncodedString()){
                        print("HASH mismatch")
                        aws_iot_disconnected()
                        aws_iot_pairing()
                        return;
                    }
       
                }
        
        
            if let button_0 = reported["button_0"] as? String{
                print("button 0 update %s",button_0)
                json_button_0 = Int(button_0)==1
                DispatchQueue.main.async {
                    viewButton0!.isOn=json_button_0;
                }
            }
            if let button_1 = reported["button_1"] as? String{
                print("button 1 update %s",button_1)
                json_button_1 = Int(button_1)==1
                DispatchQueue.main.async {
                 viewButton1!.isOn=json_button_1;
                    }
                }
            if let slider_0 = reported["slider_0"] as? String{
                print("slider 0 update %s",slider_0);
                json_slider_0 = Int(slider_0)!
                DispatchQueue.main.async {
                    viewSlider0!.setValue(Float(slider_0)!, animated: true)
                    }
                }
        
                /* for center envelop icon */
            if let envelop = reported["mail"] as? String{
                print("envelop update %s",envelop);
                json_envelope = Int(envelop)==1
                DispatchQueue.main.async {
                    if json_envelope {
                        if #available(iOS 13.0, *) {
                            envelope_view.image = UIImage(systemName: "envelope.fill" )
                        } else {
                            // Fallback on earlier versions
                        };}
                    else{
                        if #available(iOS 13.0, *) {
                            envelope_view.image = UIImage(systemName: "envelope" )
                        } else {
                            // Fallback on earlier versions
                        };
                        
                    }
                }
  
            }
        
            /* for update log message */
            if let awslog = reported["log"] as? String{
                DispatchQueue.main.async {
                    log_label.text = awslog;
                        }
                    }

                }
            }
        
        }
     
    }catch{
     }
}


/* loop for compare UI status and Reported json, if diff publish desired */
func run_loop(){

    var desired = [String:AnyObject]();
    
    let button0_status = viewButton0!.isOn;
    let button1_status = viewButton1!.isOn;
    let slider0_status = Int(viewSlider0!.value);
    
    
  

    if (json_button_0 != button0_status){
        if button0_status {
            desired["button_0"] = "1" as AnyObject}
        else {
            desired["button_0"] = "0" as AnyObject}
    }

    if (json_button_1 != button1_status){
      if button1_status {
            desired["button_1"] = "1" as AnyObject}
        else {
            desired["button_1"] = "0" as AnyObject}
    }

    if (json_slider_0 != slider0_status){
        desired["slider_0"] = String(slider0_status) as AnyObject
    }
    
    
    if (desired.isEmpty){
        return;
    }
    
    var state = [String:AnyObject]();
    
    state["desired"] = desired as AnyObject;
    
    var root = [String:AnyObject]();

    root["state"] = state as AnyObject;
    
   
    
    do{
   
        if (JSONSerialization.isValidJSONObject(state)){
            let jsondata = try JSONSerialization.data(withJSONObject: root)
        let dataManager = AWSIoTDataManager(forKey: "iot_dm")
        
        /* if this jsondata some as last json published, mean IoT no response, resend until maxsend */
        if jsondata==sendJSON {
     
            numofsend+=1;
            print("resend ",numofsend);
            if resend % numofsend != 0 || numofsend > maxsend {
                return;
            }
            
        }else{
            numofsend = 0
            sendJSON = jsondata
        }
            
        /* publish to shadow update */
        dataManager.publishData(jsondata, onTopic: "$aws/things/"+myThings+"/shadow/update", qoS: .messageDeliveryAttemptedAtLeastOnce)
            
 
    }
        
        
    }catch{
    }
        

    
   
    
    
}


/** function for subscribe to AWS shadow */
func aws_iot_subscribe(){

    /** for handle transcript callback */
    func messageTranscriptblock(message:Data){
        
        do{
            let json = try JSONSerialization.jsonObject(with: message )
            if let root = json as? [String:AnyObject] {
            /*show trascript in username_label */
            /*
                 let transcript =  root["transcript"] as! String
                    DispatchQueue.main.async {
                    username_label.text = transcript
                    }
             */
       
                let ai =  root["classtype"] as! [String:AnyObject]
        
                    DispatchQueue.main.async {
                        for (type,value) in ai{
                                         username_label.text?.append("\(type):\(value as! String)\n")
                                                    }
                                       
                    }
              
                
            }
        }catch{
            
        }
        
        print(message);
    }
    
    /** for handle shadow callback */
    func messageblock(message:Data){

        update_item(message: message)
     
        /*start timer loop for received first shadow for get*/
        if (!get_finished){
            
            DispatchQueue.main.async {
                let timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true, block:{
                    timer in
                        run_loop()
                    })
                }
            
            dataManager.subscribe(toTopic: "$aws/things/"+myThings+"/shadow/update/accepted", qoS: .messageDeliveryAttemptedAtLeastOnce, messageCallback:messageblock )
            
            get_finished = true;
        }
           
        print(message);
    }
    
    
    let dataManager = AWSIoTDataManager(forKey: "iot_dm")
        
 
     /*subscribe to get/accepted*/
    dataManager.subscribe(toTopic: "$aws/things/"+myThings+"/shadow/get/accepted", qoS: .messageDeliveryAttemptedAtLeastOnce, messageCallback:messageblock )

    /*publish to get to get all shadow data */
    dataManager.publishString("",onTopic: "$aws/things/"+myThings+"/shadow/get", qoS: .messageDeliveryAttemptedAtLeastOnce )
       
    /*subscribe to /transcript/get for received transcript text*/
 //   dataManager.subscribe(toTopic: myThings+"/transcript/get", qoS: .messageDeliveryAttemptedAtLeastOnce, messageCallback:messageTranscriptblock )

    
    /*subscribe to /ai/get for received transcript text*/
    dataManager.subscribe(toTopic: myThings+"/ai/get", qoS: .messageDeliveryAttemptedAtLeastOnce, messageCallback:messageTranscriptblock )

}

/**  for start pairing, user input DSN */
func aws_iot_pairing(){

    uuid = UUID().uuidString
    
    DispatchQueue.main.async {
            viewController.display_dsn_dialog()
    }
    
}

/**  for pairing passed */
func aws_iot_pairing_passed(){
    let dataManager = AWSIoTDataManager(forKey: "iot_dm")

    let passhash = random_paired_hash()
    
    let preferences = UserDefaults.standard
    
    let pairing_passcodekey = "mythingshash"

                   
    preferences.set(paired_passcode, forKey: pairing_passcodekey)
    
    dataManager.publishString("{\"requests\":\"ios_paired\",\"dsn\":\"\(dsn!)\",\"uuid\":\"\(uuid!)\",\"pass_hash\":\"\(passhash)\"}",onTopic: "mailbox/productmetadata", qoS: .messageDeliveryAttemptedAtLeastOnce )
    
    
    sleep(1)

    aws_iot_subscribe()
    
}

func aws_iot_productmetadata(_ input_dsn:String!){
    
    
    
    let dataManager = AWSIoTDataManager(forKey: "iot_dm")
    
    func messageblock(message:Data){
        
        do{
           
            let json = try JSONSerialization.jsonObject(with: message )
            
            if let root = json as? [String:AnyObject] {
                
                let product_dsn =  root["DSN"] as! String?
                let product_uuid = root["UUID"] as! String?
   
                
                
                if  product_uuid == uuid && dsn != product_dsn {
                    DispatchQueue.main.async {
                              viewController.display_dsn_dialog()
                    }
                    return;
                }
                
                auth_things = root["TOPIC"] as! String?
                auth_HASH = root["HASH"] as! String?
                
                
                DispatchQueue.main.async {
                    viewController.display_pairing_dialog()
                }
            
                
            }

            
        }catch{
            
        }
        
    }
    
    /* subscribe for listen get device productmetadata */
    dataManager.subscribe(toTopic: "mailbox/productmetadata/get", qoS: .messageDeliveryAttemptedAtLeastOnce, messageCallback:messageblock )
    
    dsn = input_dsn;
    
    /* send publish to get device productmetadata */
    dataManager.publishString("{\"requests\":\"ios_pairing\",\"dsn\":\"\(dsn!)\",\"uuid\":\"\(uuid!)\"}",onTopic: "mailbox/productmetadata", qoS: .messageDeliveryAttemptedAtLeastOnce )

}


func aws_iot_connected(){
    
    if (aws_iot_isConnected){
         return
     }
    
    /*reterive Things and paired hash from preferences*/
    let preferences = UserDefaults.standard

    let mythingskey = "mythings"

    if preferences.object(forKey: mythingskey) == nil {
        aws_iot_pairing()
    } else {
        myThings = preferences.string(forKey: mythingskey)!
    }
    
    let pairing_passcodekey = "mythingshash"
    if preferences.object(forKey: pairing_passcodekey) == nil {
        aws_iot_pairing()
    } else {
        paired_passcode = preferences.data(forKey: pairing_passcodekey)!
    }
    
    /*call subscribe */
    if (myThings != ""){
     aws_iot_subscribe()
    }
    
    aws_iot_isConnected = true;
    

   
    
}


func aws_iot_disconnected(){
    /*unsubscribe relative topic*/
    let dataManager = AWSIoTDataManager(forKey: "iot_dm")
    dataManager.unsubscribeTopic("mailbox/productmetadata/get")
    dataManager.unsubscribeTopic(myThings+"/transcript/get")
    dataManager.unsubscribeTopic(myThings+"/ai/get")
    dataManager.unsubscribeTopic("$aws/things/"+myThings+"/shadow/get/accepted")
    aws_iot_isConnected = false;
}


func mqttEventCallback(_ status: AWSIoTMQTTStatus ) {
    switch status {
    case .connecting: print("Connecting to AWS IoT")
    case .connected:
        print("Connected to AWS IoT")
        aws_iot_connected()
    case .connectionError: print("AWS connectionError")
        aws_iot_disconnected()
    case .connectionRefused: print("AWS connectionRefused")
    case .protocolError: print("AWS IoT protocolError")
    case .disconnected: print("AWS IoT disconnected")
        aws_iot_disconnected()
    case .unknown: print("AWS IoT unknown")
    default: print("AWS Error MQTT unknown")
    }
}

/** for connect AWS IOT*/
func connect_aws_iot(clientid:String!){
      
      
    username = clientid;
    
    /*attach policy*/
    let policy = AWSIoTAttachPolicyRequest();
    policy?.target = clientid
    policy?.policyName = "cognito_policy"
    
    AWSIoT.default().attachPolicy(policy!).continueWith(block: { (task:AWSTask<AnyObject>) -> Any? in
        
        if (task.result != nil){
            return nil;
        }
        
       // DispatchQueue.global(qos: .background).async {

        let dataManager = AWSIoTDataManager(forKey: "iot_dm")
            /* connect mqtt websocket */
        dataManager.connectUsingWebSocket(withClientId: clientid,
                                          cleanSession: true,
                                          statusCallback: mqttEventCallback)
      //  }
        return nil;
    })
}


func AWSIOT_init(){
// Initialising AWS IoT And IoT DataManager
    
    
   // AWSIoT.register(with:  AWSServiceManager.default().defaultServiceConfiguration, forKey: "us-west-2_iot")  // Same configuration var as above
    let iotEndPoint = AWSEndpoint(urlString: "wss://a1xayahoc1mnre-ats.iot.us-west-2.amazonaws.com/mqtt") // Access from AWS IoT Core --> Settings
    let iotDataConfiguration = AWSServiceConfiguration(region: .USWest2,     // Use AWS typedef .Region
                                                   endpoint: iotEndPoint,
                                                   credentialsProvider: AWSMobileClient.default())
    
    AWSIoTDataManager.register(with: iotDataConfiguration!, forKey: "iot_dm")

    AWSMobileClient.default().getIdentityId().continueWith(block:{ (task:AWSTask<NSString>) -> Any? in
        if let error = task.error as NSError? {
            print("Error get IdentityId => \(error)")
            return nil  // Required by AWSTask closure
        }
        
        
        connect_aws_iot(clientid: task.result! as String)
        return nil
    })
    
    
    
}
