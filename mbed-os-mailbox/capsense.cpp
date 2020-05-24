
/*******************************************************************************
* File Name: capsense.cpp
*
*  Handling capsense based on example from cypress
*
*
*  * Licensed under the Apache License, Version 2.0
*
*******************************************************************************/


/*******************************************************************************
* Header files including
*******************************************************************************/

#include "mbed.h"
#include "cy_pdl.h"
#include "cycfg_capsense.h"
#include "cycfg.h"
#include "cybsp.h"
#include "lcd_ui.h"
#include "awsiot.h"
#include "hw.h"
#include "capsense.h"

DigitalOut ledStatus(CYBSP_USER_LED) ;


/* Defines periodicity of the CapSense scan and touch processing in
 * milliseconds.
 */
#define CAPSENSE_SCAN_PERIOD_MS                 (20u)

#define SLIDER_NUM_TOUCH                        (1u)
#define LED_OFF                                 (1u)
#define LED_ON                                  (0u)

// define direct control lcd by capsense, normally control by AWS
//#define CAPSENSE_DRAW_LCD


/***************************************
* Function Prototypes
**************************************/
void RunCapSenseScan(void);
void InitTunerCommunication(void);
void ProcessTouchStatus(void);
void EZI2C_InterruptHandler(void);
void CapSense_InterruptHandler(void);
void CapSenseEndOfScanCallback(cy_stc_active_scan_sns_t * ptrActiveScan);

/*******************************************************************************
* Interrupt configuration
*******************************************************************************/
const cy_stc_sysint_t CapSense_ISR_cfg =
{
    .intrSrc = CYBSP_CSD_IRQ,
    .intrPriority = 4u
};

#ifdef USE_CAPSENSE_TUNER
const cy_stc_sysint_t EZI2C_ISR_cfg = {
    .intrSrc = CYBSP_CSD_COMM_IRQ,
    .intrPriority = 3u
};
#endif

/*******************************************************************************
* Global variables
*******************************************************************************/

Semaphore capsense_sem;
cy_stc_scb_ezi2c_context_t EZI2C_context;
uint32_t prevBtn0Status = 0u;
uint32_t prevBtn1Status = 0u;
uint32_t prevSliderPos = 0u;




/* SysPm callback params */
cy_stc_syspm_callback_params_t callbackParams =
{
    .base       = CYBSP_CSD_HW,
    .context    = &cy_capsense_context
};

cy_stc_syspm_callback_t capsenseDeepSleepCb =
{
    Cy_CapSense_DeepSleepCallback,
    CY_SYSPM_DEEPSLEEP,
    (CY_SYSPM_SKIP_CHECK_FAIL | CY_SYSPM_SKIP_BEFORE_TRANSITION | CY_SYSPM_SKIP_AFTER_TRANSITION),
    &callbackParams,
    NULL,
    NULL
};


/*******************************************************************************
* Function Name: capsense_init()
********************************************************************************
* Summary:
*   Main function that starts a thread for CapSense scan and enters a forever
*   wait state.
*
*******************************************************************************/
int capsense_init()
{
#ifdef USE_CAPSENSE_TUNER


    InitTunerCommunication();
#endif
    /* Initialize the CSD HW block to the default state. */
    cy_status status = Cy_CapSense_Init(&cy_capsense_context);

    if(CY_RET_SUCCESS != status)
    {
        printf("CapSense initialization failed. Status code: %lu\r\n", status);
        Cy_SysLib_Delay(osWaitForever);
    }

    /* Initialize CapSense interrupt */
    Cy_SysInt_Init(&CapSense_ISR_cfg, &CapSense_InterruptHandler);
    NVIC_ClearPendingIRQ(CapSense_ISR_cfg.intrSrc);
    NVIC_EnableIRQ(CapSense_ISR_cfg.intrSrc);

    /* Initialize the CapSense firmware modules. */
    Cy_CapSense_Enable(&cy_capsense_context);
    Cy_SysPm_RegisterCallback(&capsenseDeepSleepCb);
    Cy_CapSense_RegisterCallback(CY_CAPSENSE_END_OF_SCAN_E,
                                 CapSenseEndOfScanCallback, &cy_capsense_context);

    /* Start the thread to run CapSense scan periodically using an event queue
     * dispatcher.
     */

    //capsense_thread.start(callback(&capsense_queue, &EventQueue::dispatch_forever));
    //capsense_queue.call_every(CAPSENSE_SCAN_PERIOD_MS, RunCapSenseScan);

    /* Initiate scan immediately since the first call of RunCapSenseScan()
     * happens CAPSENSE_SCAN_PERIOD_MS after the event queue dispatcher has
     * started.
     */
    Cy_CapSense_ScanAllWidgets(&cy_capsense_context);

    printf("Application has started. Touch any CapSense button or slider.\r\n");



}


/*******************************************************************************
* Function Name: RunCapSenseScan()
********************************************************************************
* Summary:
*   This function starts the scan, and processes the touch status. It is
* periodically called by an event dispatcher.
*
*******************************************************************************/
void RunCapSenseScan(void)
{
    Cy_CapSense_Wakeup(&cy_capsense_context);

    if (CY_CAPSENSE_NOT_BUSY == Cy_CapSense_IsBusy(&cy_capsense_context))
    {
        Cy_CapSense_ScanAllWidgets(&cy_capsense_context);
    }

    capsense_sem.acquire();
    Cy_CapSense_ProcessAllWidgets(&cy_capsense_context);
    Cy_CapSense_RunTuner(&cy_capsense_context);
    ProcessTouchStatus();
}


/*******************************************************************************
* Function Name: InitTunerCommunication
********************************************************************************
*
* Summary:
*   This function performs the following functions:
*       - Initializes SCB block to operate in EZI2C mode
*       - Sets communication data buffer to CapSense data structure
*
*******************************************************************************/
#ifdef USE_CAPSENSE_TUNER
void InitTunerCommunication(void)
{
    Cy_SCB_EZI2C_Init(CYBSP_CSD_COMM_HW, &CYBSP_CSD_COMM_config, &EZI2C_context);

    /* Initialize and enable EZI2C interrupts */
    Cy_SysInt_Init(&EZI2C_ISR_cfg, &EZI2C_InterruptHandler);
    NVIC_EnableIRQ(EZI2C_ISR_cfg.intrSrc);

    /* Set up communication data buffer to CapSense data structure to be exposed
     * to I2C master at primary slave address request.
     */
    Cy_SCB_EZI2C_SetBuffer1(CYBSP_CSD_COMM_HW, (uint8 *)&cy_capsense_tuner,
        sizeof(cy_capsense_tuner), sizeof(cy_capsense_tuner), &EZI2C_context);

    /* Enable EZI2C block */
    Cy_SCB_EZI2C_Enable(CYBSP_CSD_COMM_HW);
}
#endif

/*******************************************************************************
* Function Name: ProcessTouchStatus
********************************************************************************
*
* Summary:
*
*   Call awsiot function for publishing data to AWS
*
*******************************************************************************/
void ProcessTouchStatus(void)
{
    uint32_t currSliderPos;
    uint32_t currBtn0Status = Cy_CapSense_IsSensorActive(CY_CAPSENSE_BUTTON0_WDGT_ID, CY_CAPSENSE_BUTTON0_SNS0_ID, &cy_capsense_context);
    uint32_t currBtn1Status = Cy_CapSense_IsSensorActive(CY_CAPSENSE_BUTTON1_WDGT_ID, CY_CAPSENSE_BUTTON1_SNS0_ID, &cy_capsense_context);
    cy_stc_capsense_touch_t *sldrTouch = Cy_CapSense_GetTouchInfo(CY_CAPSENSE_LINEARSLIDER0_WDGT_ID, &cy_capsense_context);

    if(currBtn0Status != prevBtn0Status)
    {
        printf("Button_0 status: %lu\r\n", currBtn0Status);
        prevBtn0Status = currBtn0Status;

        if (!currBtn0Status)
        	hw_flipping_button(0);


    }

    if(currBtn1Status != prevBtn1Status)
    {
        printf("Button_1 status: %lu\r\n", currBtn1Status);
        prevBtn1Status = currBtn1Status;

        if (!currBtn1Status)
        	hw_flipping_button(1);
    }

    if (sldrTouch->numPosition == SLIDER_NUM_TOUCH)
    {
        currSliderPos = sldrTouch->ptrPosition->x;

        if(currSliderPos != prevSliderPos)
        {
            printf("Slider position: %lu\r\n", currSliderPos);
            prevSliderPos = currSliderPos;
            hw_set_slider(0,currSliderPos);
           // awsiot_update("slider_0",to_string(currSliderPos));
        }



    }

    ledStatus = (currBtn0Status || currBtn1Status || (SLIDER_NUM_TOUCH == sldrTouch->numPosition)) ? LED_ON : LED_OFF;


}


/*******************************************************************************
* Function Name: EZI2C_InterruptHandler
********************************************************************************
* Summary:
*   Wrapper function for handling interrupts from EZI2C block.
*
*******************************************************************************/
#ifdef USE_CAPSENSE_TUNER
void EZI2C_InterruptHandler(void)
{
    Cy_SCB_EZI2C_Interrupt(CYBSP_CSD_COMM_HW, &EZI2C_context);
}
#endif

/*******************************************************************************
* Function Name: CapSense_InterruptHandler()
********************************************************************************
* Summary:
*  Wrapper function for handling interrupts from CSD block.
*
*******************************************************************************/
void CapSense_InterruptHandler(void)
{
    Cy_CapSense_InterruptHandler(CYBSP_CSD_HW, &cy_capsense_context);
}


/*******************************************************************************
* Function Name: CapSenseEndOfScanCallback()
********************************************************************************
* Summary:
*  This function releases a semaphore to indicate end of a CapSense scan.
*
* Parameters:
*  cy_stc_active_scan_sns_t* : pointer to active sensor details.
*
*******************************************************************************/
void CapSenseEndOfScanCallback(cy_stc_active_scan_sns_t * ptrActiveScan)
{
    capsense_sem.release();
}
