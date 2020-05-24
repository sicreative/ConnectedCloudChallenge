/*
 *
 *
 * Simple json write and parser
 * (not fully function)
 *
 * 	Author: slee
 *
 */

#include "json_aws.h"
#include "mbed.h"







#define BRO '{'
#define BRC '}'
#define COL ':'
#define COA ','
#define QTM '"'
#define SPC ' '



#define BRO '{'
#define BRC '}'
#define COL ':'
#define COA ','
#define QTM '"'
#define SPC ' '






void json_writer(map<string,JSON> *m,string& s){
 //   string s;


    for(map<string,JSON>::iterator it=m->begin();it!=m->end(); it++){
        if (it!=m->begin())
            s.append(",");
        else
            s.append("{");

        s.append("\"");
        s.append(it->first);
        s.append("\":");

        if (it->second.type==AWS_JSON_CHILD)
            json_writer(it->second.value.m,s);
        else if (it->second.type==AWS_JSON_STRING){
                s.append("\"");
            s.append(*it->second.value.s);
                s.append("\"");
        }else if  (it->second.type==AWS_JSON_INT_NUM){
            s.append(to_string(it->second.value.i));
        }

    }
     s.append("}");
  //  return s;


}


string json_writer(map<string,JSON> *m){
    string s;

    json_writer(m,s);



    return s;


}



map<string,JSON>* json_parse(string::iterator start,string::iterator end){

    map<string,JSON>* m = new map<string,JSON>();



    string::iterator it = start;

    if (it==end)
            return m;


    string key;
    string::iterator first = start;

    int status = 0;


    do{
        if (status==0){
            if (*it==BRO)
                ++status;
            continue;
        }


        if (status==1){
            if (*it==QTM){
                first = it;
                ++first;
                ++status;}

                continue;
        }

        if (status==2){
            if (*it==QTM){
                key = string(first,it);
                ++status;
            }

            continue;
        }

        if (status==3){
            if (*it==COL)
                ++status;
            continue;
        }

        if (status==4){
            if (*it==BRO){
                first = it;



                //Count for find closure pos
                 int count = 1;
                while( ++it!=end && count ){
                    if (*it==BRO)
                        ++count;
                    else if (*it==BRC)
                        --count;


              }




                (*m)[key].type = AWS_JSON_CHILD;
                (*m)[key].value.m = json_parse(first,it);









                status+=3;
            }else if(*it==QTM){
                first = it;
                ++first;
                ++status;
                continue;
            }else if(*it-'0'<10){
                //JSON Num
                first = it;
                status+=2;
                continue;

            }

        }

        if (status==5){

            if (*it==QTM){

            	(*m)[key].type = AWS_JSON_STRING;
            	(*m)[key].value.s = new string(first,it);



                status+=2;

            }
            continue;
        }


        if (status==6){

                 if (*it-'0'>9 || *it-'0'<0){

                	 (*m)[key].type = AWS_JSON_INT_NUM;

                	 (*m)[key].value.i = stoi(string(first,it));




                     --it;



                     ++status;

                 }
                 continue;
             }


        if (status==7)
            if (*it==COA)
                status=1;






    }while(++it != end);

    return m;

}

map<string,JSON>* json_parse(string& text){
    return json_parse(text.begin(),text.end());
}

