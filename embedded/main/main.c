#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_dsp.h"

// Weights
#define W_1   7.68781361f   
#define W_2   18.26145058f
#define W_3   -7.12684181f
#define W_4   -1.11383152f
#define W_5   8.92195609f
#define W_6   -15.36296413f
#define W_7   3.6794663f
#define W_8   0.0f
#define W_9   -0.94439872f
#define W_10  -60.50801995f  
#define W_11  -0.46996411f
#define W_12  -1.77666056f
#define W_13  -0.37737301f

#define BIAS  -15.75853187f 

#define QUEUE_SIZE 10
#define WEIGHTS_SIZE 13
#define TASK_STACK_SIZE 4096  // bytes


/// @brief Calculates the sigmoid function for a given input value.
///
/// The sigmoid function is a commonly used mathematical function in neural networks,
/// transforming a value to an output between 0 and 1. It is often used as an activation function.
///
/// @param z The input value to calculate the sigmoid function.
///
/// @return The result of applying the sigmoid function to the input value.
float sigmoid(float z);


/// @brief Makes a prediction using the given input vector.
/// 
/// This function takes an input vector (typically features) and performs a prediction
/// using the model implemented in the `predict()` function. The predicted value is returned.
/// This function can be used for classification, regression, or other modeling tasks.
/// 
/// @param input Pointer to the input vector for making the prediction.
/// 
/// @return The predicted value based on the provided input or -1 if something went wrong.
int8_t predict(float *input);


/// @brief Task function that performs prediction using input data.
/// 
/// This function is executed as a task in an RTOS (FreeRTOS). It receives data from a queue
/// (typically a vector of input values), performs a prediction using the `predict()` function,
/// and prints the result.
/// 
/// @param pvParameters Pointer to task parameters. Not used in this case.
void predict_task(void *pvParameters);


/// @brief Task function that reads user input and sends it to the prediction task.
/// 
/// This function runs as a task in an RTOS (FreeRTOS). It is responsible for receiving user serial input,
/// processing it (e.g., converting a string to numbers), and sending it to a queue to be consumed
/// by the prediction task.
/// 
/// @param pvParameters Pointer to task parameters. Not used in this case.
void read_input_task(void *pvParameters);


/// @brief Queue for communication between tasks
QueueHandle_t queue;


__attribute__((aligned(16)))  // Memory alignment -> optimazing SIMD instructions
const float weights[WEIGHTS_SIZE] = { W_1, W_2, W_3, W_4, W_5, W_6, W_7, W_8, W_9, W_10, W_11, W_12, W_13 };


//======================
// Entrypoint
//======================
void app_main(void)
{
  queue = xQueueCreate(QUEUE_SIZE, sizeof(float*));

  if(queue) {
    ESP_LOGI("main", "Queue created succesfully!\n");
    xTaskCreate(read_input_task, "read_input_task", TASK_STACK_SIZE, NULL, 0, NULL);
    xTaskCreate(predict_task, "predict_task", TASK_STACK_SIZE, NULL, 0, NULL);
  }
}


//======================================
// Functions implementations          //
//======================================
float sigmoid(float z)
{
  return 1.0f / (1.0f + expf(-z));
}


//============================
int8_t predict(float *features)
{
  // validation
  if(features == NULL) {
    ESP_LOGE("predict", "NULL features vector\n");
    return -1;
  }

  // weighted sum
  float logit = 0.0;
  esp_err_t ret = dsps_dotprod_f32_aes3(features, weights, &logit, WEIGHTS_SIZE);
  
  if(ret != ESP_OK){
    ESP_LOGE("predict", "Vector operation error: %i", ret);
    return -1;
  }

  logit += BIAS;

  float proba = sigmoid(logit);
  return (proba > 0.5f) ? 1 : 0;
}


//===================================
void predict_task(void *pvParameters)
{
  float * result = NULL;
  int8_t prediction = -1;

  while(1){

    // if has an element at queue
    if(xQueueReceive(queue, &result, portMAX_DELAY)) {
      
      //debug==========================
      // printf("\nRESULT VECTOR RECEIVED\n");
      // for(int x=0; x<WEIGHTS_SIZE; x++) {
      //   printf("%.2f ", result[x]);
      // }
      printf("Queue pop...\n");
      //==============================

      // logistic regressor
      prediction = predict(result);

      //debug==========================
      if(prediction == 1) printf("Prediction: SQUAT");
      else if(prediction == 0) printf("Prediction: STEP");
      else printf("Prediction: ERROR\n");

      // dealloc memory
      if(result){
        heap_caps_free(result);
      }
    }

    // 100ms delay
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}


// ====================================
void read_input_task(void *pvParameters)
{
  char input[128] = ""; //serial data
  char *token;
  float * result = NULL;  // features vector
  uint8_t r_index = 0;  

  while(1) {
    // async data read
    scanf("%127s", input);

    // if input is not void
    if(strcmp(input, "")) {
      
      token = strtok(input, ",");
      
      // aligned dynamic allocation
      result = (float*)heap_caps_aligned_alloc(16, WEIGHTS_SIZE * sizeof(float), MALLOC_CAP_DEFAULT);
      if(!result) {
        ESP_LOGE("read","Dynamic allocation error!\n");
        continue;
      }

      // Parsing string to float and filling vector
      r_index = 0;
      while(token != NULL) {

        result[r_index] = atoff(token);
        r_index ++;
        
        token = strtok(NULL, ",");
      }
      

      // Sending data to predict task
      // TODO: create queue veritication pdPASS / pdTRUE to dealloc memory if it fails
      xQueueSend(queue, &result, portMAX_DELAY);


      //debug==============================
      printf("\nFeatures Vector Received\n");
      for(int x=0; x<WEIGHTS_SIZE; x++) {
        printf("%.2f ", result[x]);
      }
      printf("\n\nQueue push...\n");
      //===================================


      // Resetting before iter
      result = NULL;
      strcpy(input, "");
    }

    // 100ms delay
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}
