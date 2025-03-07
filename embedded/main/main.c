#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "string.h"
#include <stdlib.h>
#include <stdio.h>

void app_main(void)
{
  char input[128] = "", *token;
  float result[3] = {0.0};
  uint8_t r_index = 0;
  
  while(1){
    //Receiving serial data
    scanf("%127s", input);
    fflush(stdin);
    
    // If data is not void
    if(strcmp(input, "")) {
      token = strtok(input, ",");

      r_index = 0;
      while(token != NULL) {
        result[r_index] = atoff(token);
        r_index ++;
        token = strtok(NULL, ",");
      }

      strcpy(input, "");
      printf("Saida: %.2f, %.2f, %.2f\n", result[0], result[1], result[2]);
    }

    // 1 second delay
    vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
}
