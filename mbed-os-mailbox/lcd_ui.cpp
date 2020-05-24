/*
 * lcd_ui.cpp
 *
 *  class for handling lcd display
 *
 *      Author: sc lee
 *
 * Licensed under the Apache License, Version 2.0
 */

#include <stdio.h>
#include <stdarg.h>

#include "lcd_ui.h"
#include "mbed.h"
bool UART_OUTPUT = false;

EventQueue lcd_queue(24 * EVENTS_EVENT_SIZE);
Thread lcd_thread( osPriorityNormal,4098,NULL,"lcd_thread");

 int downside_delay_id = NULL;
 int text_delay_id = NULL;


char* textbuffer[4];


void lcd_clear_queue(int id){

	lcd_queue.cancel(id);
}



void lcd_draw_text_top_bar_thread(char *text){
	 GUI_SetFont(&GUI_Font16_1);
	 GUI_RECT Rect = {20,2,300,18};
	 GUI_SetBkColor(GUI_DARKRED);
	 GUI_SetColor(GUI_LIGHTGRAY);
	 GUI_DispStringInRectWrap(text , &Rect, GUI_TA_LEFT, GUI_WRAPMODE_WORD );


		text[0]='^';


}







void lcd_draw_text_top_bar(const char *text){

	for(int i=0;i<4;i++){
		if(!textbuffer[i]){
			textbuffer[i] = new char[64];
			textbuffer[i][0]='^';
		}
		if(textbuffer[i][0]=='^'){

			strcpy(textbuffer[i],text);

			lcd_queue.call(lcd_draw_text_top_bar_thread,textbuffer[i]);
			break;

		}



	}




}



void lcd_draw_downside_thread(char* text){
	   GUI_SetFont(&GUI_Font32_1);
	   //GUI_SetPenSize(10);




	   GUI_RECT Rect = {78,180,242,220};


	   GUI_SetClearTextRectMode(1u);
	   GUI_SetBkColor(GUI_DARKGREEN);
	   GUI_SetColor(GUI_DARKRED);




		GUI_DispStringInRectWrap(text , &Rect, GUI_TA_CENTER, GUI_WRAPMODE_WORD );




		text[0]='^';







}



void lcd_draw_text_thread(char* text,int line){
	   GUI_SetFont(&GUI_Font24_1);
	   //GUI_SetPenSize(10);





	   GUI_RECT Rect = {20,50+line*30,300,50+(line+1)*30};


	   GUI_SetClearTextRectMode(1u);
	   GUI_SetBkColor(GUI_DARKGREEN);
	   GUI_SetColor(GUI_LIGHTGRAY);




		GUI_DispStringInRectWrap(text , &Rect, GUI_TA_CENTER, GUI_WRAPMODE_WORD );




		text[0]='^';




}



int lcd_draw_downside_text(const char *text){



	for(int i=0;i<4;i++){
		if(!textbuffer[i]){
			textbuffer[i] = new char[64];
			textbuffer[i][0]='^';
		}
		if(textbuffer[i][0]=='^'){

			strcpy(textbuffer[i],text);

			return lcd_queue.call(lcd_draw_downside_thread,textbuffer[i]);

		}



	}


}

int lcd_draw_downdside_text_delay(const char *text,int delay_ms){


	for(int i=0;i<4;i++){
			if(!textbuffer[i]){
				textbuffer[i] = new char[64];
				textbuffer[i][0]='^';
			}
			if(textbuffer[i][0]=='^'){

				strcpy(textbuffer[i],text);

				return lcd_queue.call_in(delay_ms,lcd_draw_downside_thread,textbuffer[i]);


			}



		}



}




int lcd_draw_text(const char *text,int line){



	for(int i=0;i<4;i++){
		if(!textbuffer[i]){
			textbuffer[i] = new char[64];
			textbuffer[i][0]='^';
		}
		if(textbuffer[i][0]=='^'){

			strcpy(textbuffer[i],text);

			return lcd_queue.call(lcd_draw_text_thread,textbuffer[i],line);


		}



	}


}

int lcd_draw_text_delay(const char *text,int line,int delay_ms){




	for(int i=0;i<4;i++){
			if(!textbuffer[i]){
				textbuffer[i] = new char[64];
				textbuffer[i][0]='^';
			}
			if(textbuffer[i][0]=='^'){

				strcpy(textbuffer[i],text);

				return lcd_queue.call_in(delay_ms,lcd_draw_text_thread,textbuffer[i],line);


			}



		}



}

void lcd_show_button_thread(int button_id,int onoff){
	int x=50;
	int y=200;
	U32 color=0x00D3D300;



	switch(button_id){
	case 0:
		break;
	case 1:
		color=0x00FF0000;
		x=320-50;
		break;

	}

	if (!onoff){
		color=GUI_DARKGRAY;
	}

	GUI_SetColor(GUI_MAKE_COLOR(color));

	int inc = onoff?1:-1;
	int r = onoff?20:25;
	while(r<26 && r>19){
	       GUI_DrawCircle(x, y, r);
	       r+=inc;
	}

	//GUI_FillCircle(x,y,25);


}

int lcd_slice_pos=-250;

void lcd_show_slice_thread(int pos){

	const int MIN_SLICE = 5;

	if (abs(lcd_slice_pos-pos)<MIN_SLICE)
		return;









	int dx = 5;
	int dy = 20;
	int x = 35+pos*0.8;
	int lastx=35+lcd_slice_pos*0.8;
	int y = 145;


	GUI_SetBkColor(GUI_DARKGREEN);


	GUI_SetColor(GUI_GetBkColor());
	GUI_FillRoundedRect(lastx,y,lastx+dx,y+dy,2);


	GUI_SetColor(GUI_LIGHTGRAY);
		GUI_FillRect(lastx,152,lastx+dx,158);


	GUI_SetColor(GUI_BLACK);
	GUI_FillRoundedRect(x,y,x+dx,y+dy,2);

	lcd_slice_pos=pos;

}

void lcd_show_button(int button_id){
	lcd_queue.call(lcd_show_button_thread,button_id,1u);
	 //lcd_queue.call(lcd_draw_text_thread,text,line);
}

void lcd_close_button(int button_id){
	lcd_queue.call(lcd_show_button_thread,button_id,0u);
	 //lcd_queue.call(lcd_draw_text_thread,text,line);
}

void lcd_show_slice(int pos){
	 lcd_queue.call(lcd_show_slice_thread,pos);
}




void lcd_startup_thread(){
	  GUI_Init();

	   // GUI_SetFont(&GUI_Font32B_1);
	    GUI_SetBkColor(GUI_DARKRED);
	    GUI_SetColor(GUI_LIGHTGRAY);
	  //  GUI_SetPenSize(20);

	    GUI_Clear();
	    GUI_SetColor(GUI_DARKGREEN);
	    GUI_FillRoundedRect(20,20,300,220,5);

	    GUI_SetColor(GUI_LIGHTGRAY);

	    GUI_FillRoundedRect(30,152,285,158,2);




	    lcd_show_button_thread(0,0);
	    lcd_show_button_thread(1,0);


	    lcd_show_slice_thread(180);

	    GUI_SetColor(GUI_DARKGREEN);

}


void lcd_StartUp(){

	lcd_thread.start(callback(&lcd_queue, &EventQueue::dispatch_forever));
	 lcd_queue.call(lcd_startup_thread);







}




