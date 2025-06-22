#include <Arduino.h>
#include <Modulino.h>
#include <Arduino_LED_Matrix.h>

#include <animations.h>

ModulinoButtons buttons;
ModulinoPixels leds;
ModulinoMovement movement;
ArduinoLEDMatrix matrix;


ModulinoColor palette[] = {
  ModulinoColor(255, 255, 0),   
  ModulinoColor(255, 210, 0),   
  ModulinoColor(255, 170, 0),  
  ModulinoColor(255, 130, 0),  
  ModulinoColor(255, 90, 0),  
  ModulinoColor(255, 45, 0),  
  ModulinoColor(255, 0, 0),  
};

float maxVel = 150;  // Maximum expected acceleration value

int currentLevel = 0;  // Current LED level based on acceleration
int numLEDs = 8;       // Number of LEDs in the strip
int numButtons = 3;    // Number of buttons
boolean isWalking = false; // Walking state



void updateAnimation() {


  if (!isWalking) {
    isWalking = true;
    matrix.loadSequence(stickman_walking);
    matrix.play(true); // Start the animation in loop mode
  }
}

void setup() {
  Serial.begin(115200);
  
  // Initialize Modulino system
  Modulino.begin();
  buttons.begin();
  leds.begin();
  movement.begin();
  
  // Clear all LEDs at startup
  leds.clear();
  leds.show();
  
  matrix.begin();
  matrix.loadSequence(stickman_walking);
  matrix.play(true); // Start the animation in loop mode
  
}

void loop() {
  // Update movement sensor and map acceleration to LED levels
  if (movement.update()) {
    float accelX = movement.getX();
    float accelY = movement.getY();
    float accelZ = movement.getZ();

    float gyroX = movement.getRoll();
    float gyroY = movement.getPitch();
    float gyroZ = movement.getYaw();
    
    // Debug output
    Serial.print("Accel X: ");
    Serial.print(accelX, 2);
    Serial.print(", Y: ");
    Serial.print(accelY, 2);
    Serial.print(", Z: ");
    Serial.println(accelZ, 2);

    Serial.print("Gyro X: ");
    Serial.print(gyroX, 2);
    Serial.print(", Y: ");
    Serial.print(gyroY, 2);
    Serial.print(", Z: ");
    Serial.println(gyroZ, 2);
    
    float verticalAccel = 1.0; // Default vertical acceleration threshold
    currentLevel = map(constrain(abs(gyroX), 0, maxVel), 0, maxVel, 0, numLEDs);
    
    Serial.print("LED Level: ");
    Serial.println(currentLevel);
  }
  
  // Handle button presses
  if (buttons.update()) {
    boolean buttonPressed = false;
    for (int i = 0; i < numButtons; i++) {
      if (buttons.isPressed(i)) {
        
        leds.set(numLEDs - 1,  ModulinoColor(0, 255, 255), 50+ i * 50); // Set last LED for button feedback
        buttonPressed = true;
        Serial.print("Button ");
        Serial.print(i);
        Serial.println(" pressed");
      }
    }
    if (!buttonPressed) {
        leds.clear(numLEDs - 1);
      }
    
  }
  
  for (int i = 0; i < numLEDs - 1; i++) { // Reserve last LED for buttons
    int ledIndex = (numLEDs - 2) - i; // Light up LEDs from bottom (index 6) to top (index 0)

    if (i < currentLevel) {
      leds.set(ledIndex, palette[i], 10);
    } else {
      leds.clear(ledIndex);
    }
  }

  
  leds.show();
  
  delay(50);
}