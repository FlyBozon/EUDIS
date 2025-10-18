#include <Arduino.h>
#include <I2S.h>
#include <Adafruit_NeoPixel.h>
#include "drone_inference.h"

// I2S Pin Configuration for XIAO RP2040
// WS (LRCLK) MUST be SCK + 1
#define I2S_SCK   6   // GPIO6  (D4) - Serial Clock (BCLK) - white
#define I2S_WS    7   // GPIO7  (D5) - Word Select (LRCLK) - blue
#define I2S_SD    4   // GPIO4  (D9) - Serial Data (DIN) - green

// Audio Configuration
#define SAMPLE_RATE 44100      // 44.1kHz sampling rate
#define BITS_PER_SAMPLE 16     // 16-bit samples
#define BUFFER_SIZE 512        // Number of samples per buffer

// NeoPixel Configuration
#define NEOPIXEL_PIN 12
#define NUM_PIXELS 1

// Drone detection configuration (from drone_inference.h)
// #define DRONE_MODEL_INPUT_SIZE already defined in drone_inference.h
// #define DRONE_TENSOR_ARENA_SIZE 8*1024

I2S i2s(INPUT);
int16_t audioBuffer[BUFFER_SIZE];
Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

// Drone detection
DroneDetector droneDetector;
int16_t droneBuffer[DRONE_MODEL_INPUT_SIZE];

// Heartbeat variables
unsigned long lastPingTime = 0;
unsigned long lastPongTime = 0;
const unsigned long PING_INTERVAL_MS = 250;

// Loud sound detection variables
unsigned long lastMaxReset = 0;
float maxVolume = 0;
unsigned long lastLoudTime = 0;

// Drone detection variables
unsigned long lastDroneTime = 0;

void setup() {
  Serial.begin(921600);
  while (!Serial && millis() < 3000) {
    delay(100);
  }

  Serial.println("INMP441 Audio Recorder - XIAO RP2040");

  // LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // I2S for INMP441
  i2s.setBCLK(I2S_SCK);
  i2s.setDIN(I2S_SD);
  i2s.setBitsPerSample(BITS_PER_SAMPLE);
  if (!i2s.begin(SAMPLE_RATE)) {
    Serial.println("ERROR: Failed to initialize I2S!");
    while (1) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(100);
      digitalWrite(LED_BUILTIN, LOW);
      delay(100);
    }
  }

  Serial.println("I2S initialized successfully");

  // Initialize UART for heartbeat
  Serial1.begin(9600);

  // Enable NeoPixel power
  pinMode(11, OUTPUT);
  digitalWrite(11, HIGH);

  // Initialize NeoPixel
  pixels.begin();
  pixels.setPixelColor(0, pixels.Color(255, 0, 0)); // Red on start
  pixels.show();

  Serial.print("Sample Rate: ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.print("Bits per Sample: ");
  Serial.println(BITS_PER_SAMPLE);
  Serial.print("Buffer Size: ");
  Serial.print(BUFFER_SIZE);
  Serial.println(" samples");
  Serial.println();

  // Blink LED
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN, LOW);
    delay(100);
  }

  // Initialize drone detector
  if (droneDetector.initialize()) {
    Serial.println("Drone detector initialized successfully");
    droneDetector.setThreshold(0.6f);  // 60% confidence threshold
  } else {
    Serial.println("WARNING: Failed to initialize drone detector");
  }
}

bool recording = false;
unsigned long samplesRead = 0;
unsigned long lastStatusUpdate = 0;

void loop() {
  unsigned long currentTime = millis();

  // Check for serial commands
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'r' || cmd == 'R') {
      recording = true;
      samplesRead = 0;
      Serial.println("Recording started");
      digitalWrite(LED_BUILTIN, HIGH);
    } else if (cmd == 's' || cmd == 'S') {
      recording = false;
      Serial.println("Recording stopped");
      digitalWrite(LED_BUILTIN, LOW);
    }
  }

  if (recording) {
    // Read audio samples from I2S
    int samplesAvailable = i2s.available();

    if (samplesAvailable > 0) {
      int samplesToRead = min(samplesAvailable, BUFFER_SIZE);
      for (int i = 0; i < samplesToRead; i++) {
        int16_t left = 0, right = 0;
        i2s.read16(&left, &right);
        audioBuffer[i] = left;
      }

      samplesRead += samplesToRead;

      // Send audio data as binary to serial for Python recording
      size_t bytesRead = samplesToRead * sizeof(int16_t);

      // Format: Start marker (0xAA 0x55) + sample count (2 bytes) + audio data
      Serial.write(0xAA); // Start marker byte 1
      Serial.write(0x55); // Start marker byte 2
      Serial.write((uint8_t)(samplesToRead & 0xFF));        // Sample count low byte
      Serial.write((uint8_t)((samplesToRead >> 8) & 0xFF)); // Sample count high byte
      Serial.write((uint8_t*)audioBuffer, bytesRead);       // Raw audio data

      // Drone detection inference
      static int droneBufferIdx = 0;
      static unsigned long lastDroneCheck = 0;
      
      for (int i = 0; i < samplesToRead; i++) {
        if (droneBufferIdx < DRONE_MODEL_INPUT_SIZE) {
          droneBuffer[droneBufferIdx++] = audioBuffer[i];
        }
        
        if (droneBufferIdx >= DRONE_MODEL_INPUT_SIZE) {
          // Run inference when buffer is full
          float confidence = droneDetector.detectDrone(droneBuffer, DRONE_MODEL_INPUT_SIZE);
          
          if (droneDetector.isDrone(confidence)) {
            Serial1.println("drone");
            lastDroneTime = currentTime;
          }
          
          // Optional: Send confidence level to serial for debugging
          if (millis() - lastDroneCheck > 2000) {
            Serial.print("Drone confidence: ");
            Serial.print(confidence, 3);
            Serial.println("");
            lastDroneCheck = millis();
          }
          
          droneBufferIdx = 0;
        }
      }

      if (millis() - lastStatusUpdate > 1000) {
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        lastStatusUpdate = millis();
      }

      // Volume level detection
      long sum = 0;
      for (int i = 0; i < samplesToRead; i++) {
        long sample = audioBuffer[i];
        sum += sample * sample;
      }
      float rms = sqrt(sum / samplesToRead);
      int volumeLevel = (int)rms;

      // Update max volume over last 5 seconds
      if (currentTime - lastMaxReset > 5000) {
        maxVolume = 0;
        lastMaxReset = currentTime;
      }
      if (volumeLevel > maxVolume) {
        maxVolume = volumeLevel;
      }

      // If louder than 75% of max volume in last 5 seconds, send loud signal
      if (volumeLevel > 0.75 * maxVolume && maxVolume > 0) {
        Serial1.println("loud");
      }
    }
  }

  // Heartbeat UART communication
  if (currentTime - lastPingTime > PING_INTERVAL_MS) {
    Serial1.println("ping");
    lastPingTime = currentTime;
  }

  if (Serial1.available()) {
    String msg = Serial1.readStringUntil('\n');
    msg.trim();
    if (msg == "ping") {
      Serial1.println("pong");
    } else if (msg == "pong") {
      lastPongTime = currentTime;
    } else if (msg == "loud") {
      lastLoudTime = currentTime;
    } else if (msg == "drone") {
      lastDroneTime = currentTime;
    }
  }

  // Update NeoPixel color based on detections
  if (currentTime - lastDroneTime < 5000) { // If drone signal received within last 5 seconds
    pixels.setPixelColor(0, pixels.Color(255, 255, 0)); // Yellow
  } else if (currentTime - lastLoudTime < 5000) { // If loud signal received within last 5 seconds
    pixels.setPixelColor(0, pixels.Color(0, 0, 255)); // Blue
  } else if (currentTime - lastPongTime < 2000) { // If pong received within last 2 seconds
    pixels.setPixelColor(0, pixels.Color(0, 255, 0)); // Green
  } else {
    pixels.setPixelColor(0, pixels.Color(255, 0, 0)); // Red
  }
  pixels.show();

  delay(1);
}
