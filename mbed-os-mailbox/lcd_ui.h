/*
 * log.h
 *
 *      Author: slee
 */

#ifndef LCD_UI_H_
#define LCD_UI_H_

#include "GUI.h"
#include <stdlib.h>

/**
 *
 * Init LCD display
 *
 *
 */
void lcd_StartUp(void);





void lcd_draw_text_top_bar(const char *text);
void lcd_show_button(int button_id);
void lcd_close_button(int button_id);
void lcd_show_slice(int pos);
int lcd_draw_text(const char *text,int line);
int lcd_draw_text_delay(const char *text,int line,int delay_ms);
int lcd_draw_downside_text(const char *text);
int lcd_draw_downdside_text_delay(const char *text,int delay_ms);



void lcd_clear_queue(int id);




template <typename... ArgTs>
void lcd_status_msg(const char *format, ArgTs... args ) {

		char buffer[64];
		sprintf(buffer,format,args...);

		lcd_draw_text_top_bar(buffer);





}

/**
 *
 * Display text message
 *
 * @param *format the formatted text
 * @param line the display time
 * @param formatted text arg
 *
 */
template <typename... ArgTs>
void lcd_msg(const char *format,int line, ArgTs... args ) {

	//char* buffer = new char[128];
	    char buffer[64];
		sprintf(buffer,format,args...);
		lcd_draw_text(buffer,line);

}

/**
 *
 * Display text message period of time
 *
 * @param *format the formatted text
 * @param line the display time
 * @param close_ms time of display
 * @param formatted text arg
 *
 */

template <typename... ArgTs>
void lcd_msg_delay_close(const char *format,int line,int close_ms, ArgTs... args ) {


		lcd_msg(format,line,args...);






	 lcd_draw_text_delay("",line,close_ms);


}


/**
 *
 * Display downside text message
 *
 * @param *format the formatted text
 *
 * @param formatted text arg
 *
 */
template <typename... ArgTs>
void lcd_downside_msg(const char *format, ArgTs... args ) {

	//char* buffer = new char[128];
	    char buffer[64];
		sprintf(buffer,format,args...);
		lcd_draw_downside_text(buffer);

}



/**
 *
 * Display downside text message period of time
 *
 * @param *format the formatted text
 *
 * @param close_ms time of display
 * @param formatted text arg
 *
 */



template <typename... ArgTs>
void lcd_downside_msg_delay_close(const char *format,int close_ms, ArgTs... args ) {


		lcd_downside_msg(format,args...);





		 lcd_draw_downdside_text_delay("",close_ms);


}


extern bool UART_OUTPUT;


#define APP_INFO( x )                       if (UART_OUTPUT) printf x






#endif /* LCD_UI_H_ */
