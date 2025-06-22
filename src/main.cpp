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
float movementThreshold = 0.8; // Threshold to consider as movement

int currentLevel = 0;  // Current LED level based on acceleration
int numLEDs = 8;       // Number of LEDs in the strip
int numButtons = 3;    // Number of buttons
boolean isWalking = false; // Walking state
boolean lastAnimationState = false; // false = still, true = walking
long lastUpdateTime = 0;
long lastMovementTime = 0;
long movementTimeout = 5000; // 5 seconds timeout for movement detection

void updateAnimation() {
  if (isWalking == lastAnimationState)
    return; // No change in animation state

  if (isWalking) {
    lastAnimationState = true;
    matrix.loadSequence(stickman_walking);
  } else {
    lastAnimationState = false;
    matrix.loadSequence(stickman_standing);
  }
  matrix.play(true);

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
    
    float gyroMagnitude = sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ);

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
    
    currentLevel = map(constrain(abs(gyroX), 0, maxVel), 0, maxVel, 0, numLEDs);
    
    Serial.print("LED Level: ");
    Serial.println(currentLevel);

    if (gyroMagnitude > movementThreshold* maxVel) {
      lastMovementTime = millis();
    }
    
  
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

  if (millis() - lastMovementTime > movementTimeout) {
    isWalking = false; // No movement detected for a while, set to still
  } else {
    isWalking = true; // Movement detected
  }

  updateAnimation();
  
  leds.show();
  
  delay(50);
}