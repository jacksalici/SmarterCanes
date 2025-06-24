#include <Arduino.h>
#include <Modulino.h>
#include <Arduino_LED_Matrix.h>
#include <ArduinoBLE.h>

#include <animations.h>

ModulinoButtons buttons;
ModulinoPixels leds;
ModulinoMovement movement;
ArduinoLEDMatrix matrix;

// BLE Service and Characteristic UUIDs
BLEService sensorService("eb7f25c3-8d96-4311-92c9-45e90f6b6f5b");
BLEStringCharacteristic sensorGyroData("a94090de-f49a-49f4-97c0-a95abc6cbb95", BLERead | BLENotify, 50);
BLEStringCharacteristic sensorAccelData("ca52c70a-3eb6-4043-add4-df23393e387f", BLERead | BLENotify, 50);

ModulinoColor palette[] = {
    ModulinoColor(255, 255, 0),
    ModulinoColor(255, 210, 0),
    ModulinoColor(255, 170, 0),
    ModulinoColor(255, 130, 0),
    ModulinoColor(255, 90, 0),
    ModulinoColor(255, 45, 0),
    ModulinoColor(255, 0, 0),
};

float maxVel = 150;            // Maximum expected acceleration value
float movementThreshold = 0.8; // Threshold to consider as movement

int currentLevel = 0;
int numLEDs = 8;
int numButtons = 3;
boolean isWalking = false;          // Walking state
boolean lastAnimationState = false; // false = still, true = walking
boolean playAnimation = true; // Control animation playback
long lastUpdateTime = 0;
long lastMovementTime = 0;
long movementTimeout = 5000;

long lastSendTime = 0;
long sendInterval = 50; 

boolean bleInitialized = false; // BLE initialization state
boolean isConnected = false;     // Track connection state

void onBLEConnected(BLEDevice central) {
  Serial.print("Connected to central: ");
  Serial.println(central.address());
  isConnected = true;
}

void onBLEDisconnected(BLEDevice central) {
  Serial.print("Disconnected from central: ");
  Serial.println(central.address());
  isConnected = false;

  BLE.advertise();
  Serial.println("Restarted advertising after disconnect");
}

void updateAnimation()
{
  if (isWalking == lastAnimationState)
    return; // No change in animation state
  
  if (!playAnimation)
  {
    if (!matrix.sequenceDone())
      matrix.play(false);
    return; // Animation playback is disabled
  }

  if (isWalking)
  {
    lastAnimationState = true;
    matrix.loadSequence(stickman_walking);
  }
  else
  {
    lastAnimationState = false;
    matrix.loadSequence(stickman_standing);
  }

  matrix.play(true);
}

boolean initBLE()
{
  if (!BLE.begin())
  {
    Serial.println("Starting BLE failed!");
    return false;
  }

  BLE.setLocalName("Better Walking Cane");
  BLE.setDeviceName("Better Walking Cane");

  BLE.setConnectionInterval(0x0006, 0x0C80); // 7.5ms to 4s
  BLE.setSupervisionTimeout(0x0C80); // 20 seconds

  BLE.setAdvertisedService(sensorService);
  sensorService.addCharacteristic(sensorGyroData);
  sensorService.addCharacteristic(sensorAccelData);

  BLE.addService(sensorService);

  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);
  BLE.setPairable(1); 

  BLE.advertise();
  Serial.println("BLE device is now advertising...");
  return true;
}

void sendBLE(float accelX, float accelY, float accelZ, float gyroX, float gyroY, float gyroZ)
{
  if (isConnected && BLE.connected())
  {
    String accelData = String(accelX, 3) + "," + String(accelY, 3) + "," + String(accelZ, 3);
    String gyroData = String(gyroX, 3) + "," + String(gyroY, 3) + "," + String(gyroZ, 3);




    if (accelData.length() < 50) {
      sensorAccelData.writeValue(accelData);
      Serial.println("Sent: " + accelData);
    } else {
      Serial.println("Data too long (" + String(accelData.length()) + " chars), skipping send");
    }

    if (gyroData.length() < 50) {
      sensorGyroData.writeValue(gyroData);
      Serial.println("Sent: " + gyroData);
    } else {
      Serial.println("Data too long (" + String(gyroData.length()) + " chars), skipping send");
    }
  } else if (!isConnected) {
    Serial.println("BLE not connected, skipping data send");
  }
}

void setup()
{
  Serial.begin(115200);
  
  // Wait a moment for serial to initialize
  delay(1000);
  Serial.println("Starting Better Walking Cane...");

  // Initialize Modulino system
  Modulino.begin();
  buttons.begin();
  leds.begin();
  movement.begin();

  // Clear all LEDs at startup
  leds.clear();
  leds.show();

  matrix.begin();
  matrix.loadSequence(stickman_standing); // Start with standing animation
  matrix.play(true); // Start the animation in loop mode

  bleInitialized = initBLE(); // Initialize BLE
  
  if (bleInitialized) {
    Serial.println("BLE initialized successfully");
  } else {
    Serial.println("BLE initialization failed");
  }
  
  Serial.println("Setup complete!");
}

void loop()
{
  BLE.poll(); 
  
  if (movement.update())
  {
    float accelX = movement.getX();
    float accelY = movement.getY();
    float accelZ = movement.getZ();

    float gyroX = movement.getRoll();
    float gyroY = movement.getPitch();
    float gyroZ = movement.getYaw();

    float gyroMagnitude = sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ);

    static long lastDebugTime = 0;
    if (millis() - lastDebugTime > 500) { 
      lastDebugTime = millis();
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
    }

    currentLevel = map(constrain(abs(gyroX), 0, maxVel), 0, maxVel, 0, numLEDs);

    if (gyroMagnitude > movementThreshold * maxVel)
    {
      lastMovementTime = millis();
    }

    // Send BLE data at specified interval
    if (millis() - lastSendTime > sendInterval && bleInitialized)
    {
      lastSendTime = millis();
      sendBLE(accelX, accelY, accelZ, gyroX, gyroY, gyroZ);
    }
  }

  // Handle button presses
  if (buttons.update())
  {
    boolean buttonPressed = false;
    for (int i = 0; i < numButtons; i++)
    {
      if (buttons.isPressed(i))
      {
        leds.set(numLEDs - 1, ModulinoColor(0, 255, 255), 50 + i * 50); // Set last LED for button feedback
        buttonPressed = true;
        Serial.print("Button ");
        Serial.print(i);
        Serial.println(" pressed");
      }
    }
    if (!buttonPressed)
    {
      leds.clear(numLEDs - 1);
    }
  }

  // Update LED strip based on movement level
  for (int i = 0; i < numLEDs - 1; i++)
  {                                   
    int ledIndex = (numLEDs - 2) - i; // Light up LEDs from bottom (index 6) to top (index 0)

    if (i < currentLevel)
    {
      leds.set(ledIndex, palette[i], 10);
    }
    else
    {
      leds.clear(ledIndex);
    }
  }

  // Determine walking state based on recent movement
  if (millis() - lastMovementTime > movementTimeout)
  {
    isWalking = false; // No movement detected for a while, set to still
  }
  else
  {
    isWalking = true; 
  }

  // Update animation based on walking state
  updateAnimation();

  // Update LED display
  leds.show();

  // Poll BLE again before delay for better responsiveness
  BLE.poll();

  delay(50);
}