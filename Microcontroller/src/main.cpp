#include <Arduino.h>
#include <I2S.h>

// I2S Pin Configuration for XIAO RP2040
// WS (LRCLK) MUST be SCK + 1
#define I2S_SCK   6   // GPIO6  (D4) - Serial Clock (BCLK) - white
#define I2S_WS    7   // GPIO7  (D5) - Word Select (LRCLK) - blue
#define I2S_SD    4   // GPIO4  (D9) - Serial Data (DIN) - green

// Audio Configuration
#define SAMPLE_RATE 44100      // 44.1kHz sampling rate
#define BITS_PER_SAMPLE 16     // 16-bit samples
#define BUFFER_SIZE 512        // Number of samples per buffer

I2S i2s(INPUT);
int16_t audioBuffer[BUFFER_SIZE];

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
}

bool recording = true;
unsigned long samplesRead = 0;
unsigned long lastStatusUpdate = 0;

void loop() {
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

      if (millis() - lastStatusUpdate > 1000) {
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        lastStatusUpdate = millis();
      }

      /* Volume level debugging
      // Calculate volume level RMS
      long sum = 0;
      for (int i = 0; i < samplesToRead; i++) {
        long sample = audioBuffer[i];
        sum += sample * sample;
      }
      float rms = sqrt(sum / samplesToRead);
      int volumeLevel = (int)rms;

      if (millis() - lastStatusUpdate > 100) {
        Serial.print("Available: ");
        Serial.print(samplesAvailable);
        Serial.print(" | Volume: ");
        Serial.print(volumeLevel);
        Serial.print(" | Raw[0]: ");
        Serial.print(audioBuffer[0]);
        Serial.print(" | Samples read: ");
        Serial.println(samplesRead);

        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        lastStatusUpdate = millis();
      }
      */
    }
  }

  delay(1);
}
