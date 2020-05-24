//
//  LoginController.swift
//  cypress_awsiot_two_way_demo
//
//  Created by slee on 9/4/2020.
//  Copyright © 2020 sclee. All rights reserved.
//
import UIKit

import AWSDynamoDB
import AWSSQS
import AWSSNS
import AWSS3
import AWSCognito
import AWSMobileClient

class LoginController: UINavigationController {
    
    @IBOutlet weak var signInStateLabel: UILabel!
     

       override func viewDidLoad() {
           super.viewDidLoad()
           
           awsLogin()
           // Do any additional setup after loading the view.
       }
    
    

       
       func awsLogin(){
           AWSMobileClient.default().initialize { (userState, error) in
                      if let userState = userState {
                       switch(userState){
                          case .signedIn:
                                  DispatchQueue.main.async {
                                     
                              }
                          case .signedOut:
                            var options :SignInUIOptions;
                            if #available(iOS 13.0, *) {
                            
                                options = SignInUIOptions(canCancel: false, logoImage: UIImage(systemName:"envelope.fill"),secondaryBackgroundColor: .black, primaryColor: .blue, disableSignUpButton: true)
                            } else {
                                
                                options = SignInUIOptions(canCancel: false,secondaryBackgroundColor: .black, primaryColor: .blue, disableSignUpButton: true)
                                // Fallback on earlier versions
                            }
                            
                            
                            AWSMobileClient.default().showSignIn(navigationController: self.navigationController!,signInUIOptions: options){ (userState, error) in
                                      if(error == nil){       //Successful signin
                                          DispatchQueue.main.async {
                                              self.signInStateLabel.text = "Logged In"
                                          }
                                      }
                              }
                          default:
                              AWSMobileClient.default().signOut()
                          }
                       
                       
                       
                      } else if let error = error {
                          print("error: \(error.localizedDescription)")
                      }
                  }

               
           
       }
    
}
