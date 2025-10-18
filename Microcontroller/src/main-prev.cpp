// #include <Arduino.h>
// #include <I2S.h>
// #include <Adafruit_NeoPixel.h>
// #include "drone_inference.h"

// // I2S Pin Configuration for XIAO RP2040
// // WS (LRCLK) MUST be SCK + 1
// #define I2S_SCK   6   // GPIO6  (D4) - Serial Clock (BCLK) - white
// #define I2S_WS    7   // GPIO7  (D5) - Word Select (LRCLK) - blue
// #define I2S_SD    4   // GPIO4  (D9) - Serial Data (DIN) - green

// // Audio Configuration
// #define SAMPLE_RATE 44100      // 44.1kHz sampling rate
// #define BITS_PER_SAMPLE 16     // 16-bit samples
// #define BUFFER_SIZE 512        // Number of samples per buffer

// // NeoPixel Configuration
// #define NEOPIXEL_PIN 12
// #define NUM_PIXELS 1

// // Drone detection configuration (from drone_inference.h)
// // #define DRONE_MODEL_INPUT_SIZE already defined in drone_inference.h
// // #define DRONE_TENSOR_ARENA_SIZE 8*1024

// I2S i2s(INPUT);
// int16_t audioBuffer[BUFFER_SIZE];
// Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

// // Drone detection
// DroneDetector droneDetector;

// // Heartbeat variables
// unsigned long lastPingTime = 0;
// unsigned long lastPongTime = 0;  // ⚠️ WAŻNE: Musi być zainicjalizowane dla początkowego zielonego LED!
// const unsigned long PING_INTERVAL_MS = 250;

// // Loud sound detection variables
// unsigned long lastMaxReset = 0;
// float maxVolume = 0;
// unsigned long lastLoudTime = 0;

// // Drone detection variables
// unsigned long lastDroneTime = 0;
// unsigned long lastDroneSent = 0;

// // Drone inference buffer (accumulates 2048 samples for FFT)
// // 2048 samples @ 44.1kHz = ~46ms of audio
// #define DRONE_INFERENCE_BUFFER_SIZE 2048
// int16_t droneInferenceBuffer[DRONE_INFERENCE_BUFFER_SIZE];
// int droneBufferIdx = 0;
// unsigned long lastDroneInference = 0;

// // Recording variables (must be declared before startAutoRecording function)
// bool recording = false;
// unsigned long samplesRead = 0;
// unsigned long lastStatusUpdate = 0;

// // Start recording automatically on startup
// void startAutoRecording() {
//   recording = true;
//   samplesRead = 0;
//   Serial.println("✓ Recording started automatically");
// }

// void setup() {
//   Serial.begin(921600);
//   while (!Serial && millis() < 3000) {
//     delay(100);
//   }

//   Serial.println("\n\n╔════════════════════════════════════════════╗");
//   Serial.println("║   EUDIS - Drone Detection System v2.0    ║");
//   Serial.println("║   Seeeduino XIAO RP2040 with INMP441    ║");
//   Serial.println("╚════════════════════════════════════════════╝\n");

//   // LED
//   pinMode(LED_BUILTIN, OUTPUT);
//   digitalWrite(LED_BUILTIN, LOW);

//   // I2S for INMP441
//   i2s.setBCLK(I2S_SCK);
//   i2s.setDIN(I2S_SD);
//   i2s.setBitsPerSample(BITS_PER_SAMPLE);
//   if (!i2s.begin(SAMPLE_RATE)) {
//     Serial.println("ERROR: Failed to initialize I2S!");
//     while (1) {
//       digitalWrite(LED_BUILTIN, HIGH);
//       delay(100);
//       digitalWrite(LED_BUILTIN, LOW);
//       delay(100);
//     }
//   }

//   Serial.println("I2S initialized successfully");

//   // Initialize UART for heartbeat
//   Serial1.begin(9600);

//   // Enable NeoPixel power
//   pinMode(11, OUTPUT);
//   digitalWrite(11, HIGH);

//   // Initialize NeoPixel
//   pixels.begin();
//   pixels.setPixelColor(0, pixels.Color(32, 0, 0)); // Red on start
//   pixels.show();
  
//   Serial.println("✓ UART initialized (9600 baud)");
//   Serial.println("✓ NeoPixel initialized");

//   Serial.print("Sample Rate: ");
//   Serial.print(SAMPLE_RATE);
//   Serial.println(" Hz");
//   Serial.print("Bits per Sample: ");
//   Serial.println(BITS_PER_SAMPLE);
//   Serial.print("Buffer Size: ");
//   Serial.print(BUFFER_SIZE);
//   Serial.println(" samples");
//   Serial.println();

//   // Initialize drone detector
//   if (droneDetector.initialize()) {
//     Serial.println("✓ Drone detector initialized successfully");
//     Serial.println("✓ Mel Spectrogram generator running");
//     droneDetector.setThreshold(0.6f);  // 60% confidence threshold for drone detection
//   } else {
//     Serial.println("⚠️ WARNING: Failed to initialize drone detector");
//   }

//   // Start recording automatically
//   startAutoRecording();
  
//   // Initialize heartbeat timers to start with green LED
//   lastPingTime = millis();
//   lastPongTime = millis();
  
//   Serial.println("\n✓✓✓ System ready! ✓✓✓");
//   Serial.println("Looking for audio input from INMP441...\n");
// }

// void loop() {
//   unsigned long currentTime = millis();

//   // Check for serial commands
//   if (Serial.available()) {
//     char cmd = Serial.read();
//     if (cmd == 'r' || cmd == 'R') {
//       recording = true;
//       samplesRead = 0;
//       Serial.println("Recording started");
//       digitalWrite(LED_BUILTIN, HIGH);
//     } else if (cmd == 's' || cmd == 'S') {
//       recording = false;
//       Serial.println("Recording stopped");
//       digitalWrite(LED_BUILTIN, LOW);
//     }
//   }

//   if (recording) {
//     // Read audio samples from I2S
//     int samplesAvailable = i2s.available();

//     if (samplesAvailable > 0) {
//       int samplesToRead = min(samplesAvailable, BUFFER_SIZE);
//       for (int i = 0; i < samplesToRead; i++) {
//         int16_t left = 0, right = 0;
//         i2s.read16(&left, &right);
//         audioBuffer[i] = left;
//       }

//       samplesRead += samplesToRead;

//       // ========================================
//       // DRONE DETECTION PIPELINE (Option 2)
//       // ========================================
//       // Accumulate 2048 samples (46ms @ 44.1kHz)
//       // → Generate Mel Spectrogram (224×224)
//       // → Run TFLite inference
//       // → Output confidence score
//       // ========================================
      
//       for (int i = 0; i < samplesToRead; i++) {
//         if (droneBufferIdx < DRONE_INFERENCE_BUFFER_SIZE) {
//           droneInferenceBuffer[droneBufferIdx++] = audioBuffer[i];
//         }
        
//         // When buffer is full, run inference
//         if (droneBufferIdx >= DRONE_INFERENCE_BUFFER_SIZE) {
//           unsigned long inferenceStart = millis();
          
//           // Step 1: Generate Mel Spectrogram (on-device)
//           // - FFT: 2048-point (optimized with ARM CMSIS-DSP)
//           // - Mel Filterbank: 128 frequency bands
//           // - dB Conversion: Log scale
//           // - Normalization: [0, 1] range
//           // Total: ~75-80ms on RP2040
          
//           float confidence = droneDetector.detectDrone(
//             droneInferenceBuffer, 
//             DRONE_INFERENCE_BUFFER_SIZE
//           );
          
//           unsigned long inferenceTime = millis() - inferenceStart;
          
//           // Check if drone detected
//           if (droneDetector.isDrone(confidence)) {
//             // Only send UART message once per 1000ms to avoid flooding
//             if (currentTime - lastDroneSent > 1000) {
//               Serial1.println("drone");
//               lastDroneSent = currentTime;
//               lastDroneTime = currentTime;
              
//               Serial.print("🚁 DRONE DETECTED! Confidence: ");
//               Serial.print(confidence, 3);
//               Serial.print(" | Processing: ");
//               Serial.print(inferenceTime);
//               Serial.println("ms");
//             }
//           }
          
//           // Debug output every 3 seconds
//           if (millis() - lastDroneInference > 3000) {
//             Serial.print("📊 [DRONE] Confidence: ");
//             Serial.print(confidence, 3);
//             Serial.print(" (threshold: 0.60) | Inference: ");
//             Serial.print(inferenceTime);
//             Serial.print("ms | Status: ");
//             Serial.println(droneDetector.isDrone(confidence) ? "ALERT ⚠️" : "Normal ✓");
//             lastDroneInference = millis();
//           }
          
//           // Reset buffer for next frame
//           droneBufferIdx = 0;
//         }
//       }

//       if (millis() - lastStatusUpdate > 1000) {
//         digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
//         lastStatusUpdate = millis();
//       }

//       // Volume level detection
//       long sum = 0;
//       for (int i = 0; i < samplesToRead; i++) {
//         long sample = audioBuffer[i];
//         sum += sample * sample;
//       }
//       float rms = sqrt(sum / samplesToRead);
//       int volumeLevel = (int)rms;

//       // Update max volume over last 5 seconds
//       if (currentTime - lastMaxReset > 15000) {
//         maxVolume = 0;
//         lastMaxReset = currentTime;
//       }
//       if (volumeLevel > maxVolume) {
//         maxVolume = volumeLevel;
//       }

//       // If louder than 75% of max volume in last 5 seconds, send loud signal
//       if (volumeLevel > 0.75 * maxVolume && maxVolume > 0) {
//         Serial1.println("loud");
//       }
//     }
//   }

//   // Heartbeat UART communication
//   if (currentTime - lastPingTime > PING_INTERVAL_MS) {
//     Serial1.println("ping");
//     // Serial.println("[UART] Sent 'ping'");
//     lastPingTime = currentTime;
//   }

//   // Read UART responses (check regularly for incoming messages)
//   // Important: This must be called frequently to not miss pong responses
//   while (Serial1.available()) {
//     String msg = Serial1.readStringUntil('\n');
//     msg.trim();
    
//     if (msg.length() == 0) continue;  // Skip empty messages
    
//     if (msg == "ping") {
//       Serial1.println("pong");
//       // Serial.println("[UART] Received 'ping', sending 'pong'");
//     } else if (msg == "pong") {
//       lastPongTime = currentTime;
//       // Serial.println("[UART] Received 'pong' - connection OK ✓");
//     } else if (msg == "loud") {
//       lastLoudTime = currentTime;
//       Serial.println("[UART] Received 'loud'");
//     } else if (msg == "drone") {
//       lastDroneTime = currentTime;
//       Serial.println("[UART] Received 'drone'");
//     }
//   }

//   // Update NeoPixel color based on detections (priority: drone > loud > green > red)
//   if (currentTime - lastDroneTime < 5000) {
//     // Drone alert: YELLOW (255, 255, 0)
//     pixels.setPixelColor(0, pixels.Color(255, 255, 0));
//   } else if (currentTime - lastLoudTime < 750) {
//     // Loud sound: BLUE (0, 0, 255)
//     pixels.setPixelColor(0, pixels.Color(0, 0, 127));
//   } else if (currentTime - lastPongTime < 3000) {
//     // Connection OK (pong received): GREEN (0, 255, 0)
//     pixels.setPixelColor(0, pixels.Color(0, 64, 0));
//   } else {
//     // No connection or timeout: RED (255, 0, 0)
//     pixels.setPixelColor(0, pixels.Color(64, 0, 0));
//   }
//   pixels.show();

//   delay(1);
// }
