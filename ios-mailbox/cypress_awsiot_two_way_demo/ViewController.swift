//
//  ViewController.swift
//  cypress_awsiot_two_way_demo
//
//  Created by slee on 6/4/2020.
//  Copyright © 2020 sclee. All rights reserved.
//

import UIKit

import AWSDynamoDB
import AWSSQS
import AWSSNS
import AWSS3
import AWSCognito
import AWSMobileClient
import AWSIoT
import CommonCrypto




class ViewController: UIViewController {
    

    @IBOutlet weak var logoutButton: UIButton!
    @IBOutlet weak var button_0: UISwitch!
    @IBOutlet weak var button_1: UISwitch!
    @IBOutlet weak var slider_0: UISlider!
    @IBOutlet weak var user_label: UILabel!
    @IBOutlet weak var envelope: UIImageView!
    @IBOutlet weak var log: UILabel!
    
    @IBAction func logoutButton_touch_down(_ sender: UIButton) {
        if sender.titleLabel?.text=="Log In"{
            awsLogin()}
        else{
            awsLogout()
        }
    }
    
    @IBAction func button_0_changed(_ sender: UISwitch) {
        
        json_button_0 = sender.isOn
    }
    @IBAction func button_1_changed(_ sender: UISwitch) {
        json_button_1 = sender.isOn
    }
    @IBAction func slider_0_changed(_ sender: UISlider) {
        json_slider_0 = Int(sender.value)
    }
    
    
    
/* pairing dialog for user serial no input  */
    func display_dsn_dialog(){
           
           
        let alert = UIAlertController(title: "DSN", message: "Please enter the product serial no", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Next", style: .default, handler: {action in
            let dsn = alert.textFields?[0]
            /*DSN require more that 8 char*/
            if (dsn?.text!.count)! < 8 {
                self.display_dsn_dialog()
                return
            }
            aws_iot_productmetadata(dsn?.text)
        }))
           alert.addTextField(configurationHandler: {(textField: UITextField!) in
               textField.placeholder = "serial no"
               textField.isSecureTextEntry = false
        
           })
           present(alert, animated: true, completion: nil)
       }
    
/* pairing dialog for user pairing code input */
    func display_pairing_dialog(){
        
        
        let alert = UIAlertController(title: "Pairing", message: "Please enter the pairing code showed on device display", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default,handler: {action in
            
            /*check passcod*/
            let passcode = alert.textFields?[0].text
            let decodedHASH = Data(base64Encoded: auth_HASH!)
            let passcodehash = sha256(data: passcode!.data(using: .utf8)!)
            
            
            if (passcodehash==decodedHASH){

                /* update Things and save to preferences */
                myThings = auth_things;
                
                let preferences = UserDefaults.standard

                let mythingskey = "mythings"

                
                preferences.set(myThings, forKey: mythingskey)

           
                let result = preferences.synchronize()

                if !result {
                   
                }
                
                /* display OK dialog */
                self.display_pairing_ok_dialog()
                
                /* call send passed message to IoT device */
                aws_iot_pairing_passed()
                
    
                return
            }
            
            /*no pass, show DSN dialog again*/
            self.display_dsn_dialog();
            
        }))
        
        
        alert.addTextField(configurationHandler: {(textField: UITextField!) in
            textField.placeholder = "passcode"
            textField.isSecureTextEntry = true // for password input
        })
        present(alert, animated: true, completion: nil)
        
        
        
    }
    
/** dialog for pairing passed */
    func display_pairing_ok_dialog(){
        
        
        let alert = UIAlertController(title: "Pairing", message: "Accepted", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default,handler: nil))
       
        present(alert, animated: true, completion: nil)
        
        
        
    }
    
    
    /** UI for AWS Cognito Log in  */
    func log_in(){
       DispatchQueue.main.async {
        
        self.button_0.isHidden = false;
        self.button_1.isHidden = false;
        self.user_label.isHidden = false;
        self.slider_0.isHidden = false;
        self.envelope.isHidden = false;
        self.log.isHidden = false;

        
        self.log.text = "";
        
       
        self.logoutButton.setTitle("Log Out",for: .normal)
        self.user_label.text = username;
        
         AWSIOT_init();

        }
        
    }
    
    /** UI  for AWS Cognito Log out  */
    func log_out(){
        DispatchQueue.main.async {

            self.button_0.isHidden = true;
            self.button_1.isHidden = true;
            self.user_label.isHidden = true;
            self.slider_0.isHidden = true;
            self.logoutButton.setTitle("Log In",for: .normal)
            self.user_label.text = "";
            self.envelope.isHidden = true;
            self.log.isHidden = true;
        }
    }



    
    override func viewDidLoad() {
        super.viewDidLoad()
        
   /* start AWS Cognitor Login when view load */
        
      AWSMobileClient.default().addUserStateListener(self){ (userState, info) in
               switch (userState) {
               case .signedIn:
                    self.log_in()
               case .signedOut:
                    self.log_out()
               default:
                   break
               }
               
           }
        
        awsLogin()
        
        
      
    }
    
    
    /** for AWS Cognitor Login */
    func awsLogin(){
        
        AWSMobileClient.default().initialize { (userState, error) in
                   if let userState = userState {
                    switch(userState){
                       case .signedIn:
                               self.log_in()
                       case .signedOut:
                        var options :SignInUIOptions;
                                                   if #available(iOS 13.0, *) {
                                                   
                                                       options = SignInUIOptions(canCancel: false, logoImage: UIImage(systemName:"envelope.fill"),secondaryBackgroundColor: .black, primaryColor: .lightGray, disableSignUpButton: true)
                                                   } else {
                                                       
                                                       options = SignInUIOptions(canCancel: false,secondaryBackgroundColor: .black, primaryColor: .blue, disableSignUpButton: true)
                                                       // Fallback on earlier versions
                                                   }
                                                   
                        
                        
                        AWSMobileClient.default().showSignIn(navigationController: self.navigationController!, signInUIOptions:options, { (userState, error) in
                                   if(error == nil){       //Successful signin
                                    self.log_in()
                                   }
                               })
                       default:
                           AWSMobileClient.default().signOut()
                       }
                    
                    
                   } else if let error = error {
                       print("error: \(error.localizedDescription)")
                   }
               }

    }

 
    override func viewWillAppear(_ animated: Bool) {
        
        /*get view by tag and storage pointer for future use, IBOutlet may loss when use AWS Cognitor log in */
        viewButton0  = view.viewWithTag(1000) as? UISwitch
        viewButton1  = view.viewWithTag(1001) as? UISwitch
        viewSlider0  = view.viewWithTag(1002) as? UISlider
        username_label = view.viewWithTag(1003) as? UILabel
        log_label = view.viewWithTag(1004) as? UILabel
        envelope_view = view.viewWithTag(1005) as? UIImageView
        
        viewController = self;
      
        
    }
    
    
 /** call for redirect to AWS API */
    func awsLogout(){
            AWSMobileClient.default().signOut()
     
       }
    
    
  
    
    
}

