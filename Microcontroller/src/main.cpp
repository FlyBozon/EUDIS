#include <Arduino.h>
#include <I2S.h>
#include <Adafruit_NeoPixel.h>
#include "arduinoFFT.h"

// I2S Pin Configuration for XIAO RP2040
#define I2S_SCK   6   // GPIO6  (D4) - Serial Clock (BCLK)
#define I2S_WS    7   // GPIO7  (D5) - Word Select (LRCLK)
#define I2S_SD    4   // GPIO4  (D9) - Serial Data (DIN)

// Audio Configuration
#define SAMPLE_RATE 44100
#define BITS_PER_SAMPLE 16
#define BUFFER_SIZE 512

// FFT Configuration
#define FFT_SIZE 1024  // Smaller for faster processing
#define FFT_OVERLAP (FFT_SIZE / 2)  // 50% overlap

// NeoPixel Configuration
#define NEOPIXEL_PIN 12
#define NUM_PIXELS 1

// Detection Configuration
#define NUM_PROFILES 9
#define HISTORY_SIZE 7  // Track last 7 detections
#define MIN_SNR 3.5     // Minimum signal-to-noise ratio
#define MIN_DETECTIONS 4  // Require 4/7 consistent detections
#define NOISE_CALIBRATION_SAMPLES 20

// Timing Configuration
const unsigned long PING_INTERVAL_MS = 250;
const unsigned long DRONE_ALERT_DURATION = 5000;
const unsigned long LOUD_DURATION = 750;
const unsigned long CONNECTION_TIMEOUT = 3000;

// Global Objects
I2S i2s(INPUT);
Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);
ArduinoFFT<float> FFT = ArduinoFFT<float>();

// Audio Buffers
int16_t audioBuffer[BUFFER_SIZE];
float fftBuffer[FFT_SIZE];
float vReal[FFT_SIZE];
float vImag[FFT_SIZE];
int fftBufferIdx = 0;

// Noise Floor
float noiseFloor[FFT_SIZE/2];
float noiseStdDev[FFT_SIZE/2];
bool noiseCalibrated = false;

// Drone Profile Structure
struct DroneProfile {
  const char* name;
  float fundamental_freq;
  float frequency_tolerance;
  float* harmonics;
  int num_harmonics;
  float harmonic_tolerance;
  float confidence_threshold;
  float min_duration_ms;  // Minimum detection duration
};

// Detection History
struct DetectionHistory {
  int profileIdx;
  float confidence;
  float snr;
  unsigned long timestamp;
};
DetectionHistory history[HISTORY_SIZE];
int historyIdx = 0;

// Timing Variables
unsigned long lastPingTime = 0;
unsigned long lastPongTime = 0;
unsigned long lastLoudTime = 0;
unsigned long lastDroneTime = 0;
unsigned long lastDroneSent = 0;
unsigned long lastStatusUpdate = 0;
unsigned long lastMaxReset = 0;
unsigned long detectionStartTime = 0;

// State Variables
bool recording = true;
float maxVolume = 0;
int consecutiveDetections = 0;
int lastDetectedProfile = -1;

// Drone Harmonic Arrays
float harmonics_1[] = {2275.0};
float harmonics_2[] = {1750.0};
float harmonics_3[] = {2046.0};
float harmonics_4[] = {};
float harmonics_5[] = {2326.5};
float harmonics_6[] = {};
float harmonics_7[] = {139.0, 223.5};
float harmonics_8[] = {};
float harmonics_9[] = {250.5, 287.5};

// Drone Profiles with optimized parameters
DroneProfile profiles[NUM_PROFILES] = {
  {"Dron_1", 1143.0, 12.0, harmonics_1, 1, 25.0, 0.65, 500},
  {"Dron_3", 886.5, 12.0, harmonics_2, 1, 25.0, 0.65, 500},
  {"Dron_4", 1028.0, 12.0, harmonics_3, 1, 25.0, 0.65, 500},
  {"Dron_2", 1759.5, 15.0, harmonics_4, 0, 30.0, 0.55, 400},
  {"Dron_5", 1145.5, 12.0, harmonics_5, 1, 25.0, 0.65, 500},
  {"Dron_6", 1124.5, 15.0, harmonics_6, 0, 30.0, 0.55, 400},
  {"Dron_7", 84.5, 10.0, harmonics_7, 2, 20.0, 0.70, 600},
  {"Dron_8", 1801.0, 15.0, harmonics_8, 0, 30.0, 0.55, 400},
  {"Dron_9", 107.5, 10.0, harmonics_9, 2, 20.0, 0.70, 600}
};

// ============================================================================
// NOISE CALIBRATION
// ============================================================================
void calibrateNoiseFloor() {
  Serial.println("\n🔧 Starting noise floor calibration...");
  Serial.println("   Please keep environment QUIET for 10 seconds!");
  
  // Initialize arrays
  for (int i = 0; i < FFT_SIZE/2; i++) {
    noiseFloor[i] = 0;
    noiseStdDev[i] = 0;
  }
  
  // Collect multiple FFT samples
  float samples[NOISE_CALIBRATION_SAMPLES][FFT_SIZE/2];
  
  for (int sample = 0; sample < NOISE_CALIBRATION_SAMPLES; sample++) {
    // Fill buffer with audio
    int samplesCollected = 0;
    while (samplesCollected < FFT_SIZE) {
      if (i2s.available()) {
        int16_t left = 0, right = 0;
        i2s.read16(&left, &right);
        vReal[samplesCollected] = (float)left;
        vImag[samplesCollected] = 0;
        samplesCollected++;
      }
    }
    
    // Apply Hamming window
    for (int i = 0; i < FFT_SIZE; i++) {
      float window = 0.54 - 0.46 * cos(2.0 * PI * i / (FFT_SIZE - 1));
      vReal[i] *= window;
    }
    
    // Compute FFT
    FFT.compute(vReal, vImag, FFT_SIZE, FFTDirection::Forward);
    FFT.complexToMagnitude(vReal, vImag, FFT_SIZE);
    
    // Store magnitudes
    for (int i = 0; i < FFT_SIZE/2; i++) {
      samples[sample][i] = vReal[i];
      noiseFloor[i] += vReal[i];
    }
    
    // Progress indicator
    if (sample % 5 == 0) {
      Serial.print(".");
    }
    
    delay(500);
  }
  
  Serial.println();
  
  // Calculate mean
  for (int i = 0; i < FFT_SIZE/2; i++) {
    noiseFloor[i] /= NOISE_CALIBRATION_SAMPLES;
  }
  
  // Calculate standard deviation
  for (int sample = 0; sample < NOISE_CALIBRATION_SAMPLES; sample++) {
    for (int i = 0; i < FFT_SIZE/2; i++) {
      float diff = samples[sample][i] - noiseFloor[i];
      noiseStdDev[i] += diff * diff;
    }
  }
  
  for (int i = 0; i < FFT_SIZE/2; i++) {
    noiseStdDev[i] = sqrt(noiseStdDev[i] / NOISE_CALIBRATION_SAMPLES);
  }
  
  noiseCalibrated = true;
  
  Serial.println("✓ Noise floor calibration complete!");
  Serial.print("   Average noise level: ");
  
  float avgNoise = 0;
  for (int i = 10; i < 100; i++) {
    avgNoise += noiseFloor[i];
  }
  avgNoise /= 90;
  Serial.println(avgNoise, 2);
}

// ============================================================================
// FFT FREQUENCY ANALYSIS
// ============================================================================
float getFrequencyMagnitude(float targetFreq, float tolerance) {
  if (!noiseCalibrated) return 0;
  
  float freqResolution = (float)SAMPLE_RATE / FFT_SIZE;
  int minBin = max(1, (int)((targetFreq - tolerance) / freqResolution));
  int maxBin = min(FFT_SIZE/2 - 1, (int)((targetFreq + tolerance) / freqResolution));
  
  float maxMagnitude = 0;
  int peakBin = -1;
  
  // Find peak magnitude in range
  for (int i = minBin; i <= maxBin; i++) {
    // Subtract noise floor with 2 sigma margin
    float signal = vReal[i];
    float noise = noiseFloor[i] + 2.0 * noiseStdDev[i];
    float magnitude = max(0.0f, signal - noise);
    
    if (magnitude > maxMagnitude) {
      maxMagnitude = magnitude;
      peakBin = i;
    }
  }
  
  // Parabolic interpolation for sub-bin accuracy
  if (peakBin > 0 && peakBin < FFT_SIZE/2 - 1) {
    float alpha = vReal[peakBin - 1];
    float beta = vReal[peakBin];
    float gamma = vReal[peakBin + 1];
    
    float p = 0.5 * (alpha - gamma) / (alpha - 2.0 * beta + gamma);
    maxMagnitude = beta - 0.25 * (alpha - gamma) * p;
  }
  
  return maxMagnitude;
}

float calculateSNR(float targetFreq) {
  if (!noiseCalibrated) return 0;
  
  float freqResolution = (float)SAMPLE_RATE / FFT_SIZE;
  int bin = (int)(targetFreq / freqResolution);
  bin = constrain(bin, 1, FFT_SIZE/2 - 1);
  
  float signal = vReal[bin];
  float noise = noiseFloor[bin] + 0.001; // Avoid division by zero
  
  return signal / noise;
}

float calculateSpectralFlatness() {
  float geometricMean = 0;
  float arithmeticMean = 0;
  int count = 0;
  
  // Analyze mid-frequency range (100 Hz to 5000 Hz)
  float freqResolution = (float)SAMPLE_RATE / FFT_SIZE;
  int startBin = (int)(100.0 / freqResolution);
  int endBin = (int)(5000.0 / freqResolution);
  
  for (int i = startBin; i < endBin; i++) {
    float mag = vReal[i] + 0.001;
    geometricMean += log(mag);
    arithmeticMean += mag;
    count++;
  }
  
  geometricMean = exp(geometricMean / count);
  arithmeticMean /= count;
  
  return geometricMean / arithmeticMean;
}

// ============================================================================
// DRONE DETECTION ENGINE
// ============================================================================
bool detectDrone(char* detectedName, float* confidence, float* snr) {
  if (!noiseCalibrated) {
    *confidence = 0;
    return false;
  }
  
  // Check spectral flatness (drones have peaks, not flat noise)
  float flatness = calculateSpectralFlatness();
  if (flatness > 0.6) {
    return false; // Too noisy, likely not a drone
  }
  
  float maxScore = 0;
  int bestProfile = -1;
  float bestSNR = 0;
  
  // Calculate overall RMS for normalization
  float rms = 0;
  for (int i = 0; i < FFT_SIZE; i++) {
    rms += fftBuffer[i] * fftBuffer[i];
  }
  rms = sqrt(rms / FFT_SIZE);
  
  if (rms < 50) {
    return false; // Too quiet
  }
  
  // Check each drone profile
  for (int p = 0; p < NUM_PROFILES; p++) {
    DroneProfile* profile = &profiles[p];
    
    // 1. Check fundamental frequency
    float fundamental_mag = getFrequencyMagnitude(
      profile->fundamental_freq, 
      profile->frequency_tolerance
    );
    
    float fundamental_snr = calculateSNR(profile->fundamental_freq);
    
    // Require minimum SNR for fundamental
    if (fundamental_snr < MIN_SNR) {
      continue;
    }
    
    // 2. Check harmonics
    float harmonic_score = 0;
    int valid_harmonics = 0;
    
    if (profile->num_harmonics > 0) {
      for (int h = 0; h < profile->num_harmonics; h++) {
        float harm_mag = getFrequencyMagnitude(
          profile->harmonics[h],
          profile->harmonic_tolerance
        );
        
        float harm_snr = calculateSNR(profile->harmonics[h]);
        
        if (harm_snr > 2.0) { // Lower threshold for harmonics
          harmonic_score += harm_snr;
          valid_harmonics++;
        }
      }
      
      if (valid_harmonics > 0) {
        harmonic_score /= profile->num_harmonics;
      } else {
        // If profile has harmonics but none detected, penalize
        harmonic_score = 0;
      }
    } else {
      // Profile has no harmonics, use only fundamental
      harmonic_score = fundamental_snr * 0.5;
    }
    
    // 3. Calculate weighted confidence score
    float weight_fundamental = profile->num_harmonics > 0 ? 0.7 : 1.0;
    float weight_harmonics = profile->num_harmonics > 0 ? 0.3 : 0.0;
    
    float total_score = (fundamental_snr * weight_fundamental + 
                        harmonic_score * weight_harmonics) / 10.0;
    
    // Normalize to 0-1 range
    total_score = constrain(total_score, 0, 1);
    
    // Check if this profile matches
    if (total_score > profile->confidence_threshold && total_score > maxScore) {
      maxScore = total_score;
      bestProfile = p;
      bestSNR = fundamental_snr;
    }
  }
  
  // Return best match if found
  if (bestProfile >= 0) {
    strcpy(detectedName, profiles[bestProfile].name);
    *confidence = maxScore;
    *snr = bestSNR;
    return true;
  }
  
  return false;
}

// ============================================================================
// TEMPORAL CONSISTENCY CHECK
// ============================================================================
bool checkTemporalConsistency(int currentProfile, float currentConfidence, float currentSNR) {
  // Add current detection to history
  history[historyIdx].profileIdx = currentProfile;
  history[historyIdx].confidence = currentConfidence;
  history[historyIdx].snr = currentSNR;
  history[historyIdx].timestamp = millis();
  historyIdx = (historyIdx + 1) % HISTORY_SIZE;
  
  // Count votes for each profile in recent history (last 3 seconds)
  int votes[NUM_PROFILES] = {0};
  float avgConfidence[NUM_PROFILES] = {0};
  int validCount[NUM_PROFILES] = {0};
  
  unsigned long currentTime = millis();
  for (int i = 0; i < HISTORY_SIZE; i++) {
    if (currentTime - history[i].timestamp < 3000) {
      int idx = history[i].profileIdx;
      if (idx >= 0 && idx < NUM_PROFILES) {
        votes[idx]++;
        avgConfidence[idx] += history[i].confidence;
        validCount[idx]++;
      }
    }
  }
  
  // Calculate average confidence for profiles with votes
  for (int i = 0; i < NUM_PROFILES; i++) {
    if (validCount[i] > 0) {
      avgConfidence[i] /= validCount[i];
    }
  }
  
  // Check if current profile has enough consistent detections
  if (votes[currentProfile] >= MIN_DETECTIONS && 
      avgConfidence[currentProfile] > profiles[currentProfile].confidence_threshold) {
    
    // Check minimum duration requirement
    if (currentProfile != lastDetectedProfile) {
      detectionStartTime = millis();
      lastDetectedProfile = currentProfile;
      consecutiveDetections = 1;
      return false;
    } else {
      consecutiveDetections++;
      unsigned long detectionDuration = millis() - detectionStartTime;
      
      if (detectionDuration >= profiles[currentProfile].min_duration_ms) {
        return true;
      }
    }
  } else {
    // Reset if detection changed
    if (currentProfile != lastDetectedProfile) {
      lastDetectedProfile = -1;
      consecutiveDetections = 0;
    }
  }
  
  return false;
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(921600);
  while (!Serial && millis() < 3000) {
    delay(100);
  }

  Serial.println("\n\n╔═══════════════════════════════════════════════╗");
  Serial.println("║   EUDIS - Advanced Drone Detection System   ║");
  Serial.println("║   Real-time FFT + Multi-stage Validation    ║");
  Serial.println("║   Seeeduino XIAO RP2040 + INMP441 + ML      ║");
  Serial.println("╚═══════════════════════════════════════════════╝\n");

  // Initialize LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Initialize I2S
  i2s.setBCLK(I2S_SCK);
  i2s.setDIN(I2S_SD);
  i2s.setBitsPerSample(BITS_PER_SAMPLE);
  
  if (!i2s.begin(SAMPLE_RATE)) {
    Serial.println("❌ ERROR: Failed to initialize I2S!");
    while (1) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(100);
      digitalWrite(LED_BUILTIN, LOW);
      delay(100);
    }
  }
  Serial.println("✓ I2S initialized (44.1kHz, 16-bit)");

  // Initialize UART
  Serial1.begin(9600);
  Serial.println("✓ UART initialized (9600 baud)");

  // Initialize NeoPixel
  pinMode(11, OUTPUT);
  digitalWrite(11, HIGH);
  pixels.begin();
  pixels.setPixelColor(0, pixels.Color(32, 0, 32)); // Purple during calibration
  pixels.show();
  Serial.println("✓ NeoPixel initialized");

  // Print loaded profiles
  Serial.println("\n📋 LOADED DRONE PROFILES:");
  for (int i = 0; i < NUM_PROFILES; i++) {
    Serial.print("  ");
    Serial.print(i + 1);
    Serial.print(". ");
    Serial.print(profiles[i].name);
    Serial.print(" - ");
    Serial.print(profiles[i].fundamental_freq, 1);
    Serial.print(" Hz (±");
    Serial.print(profiles[i].frequency_tolerance, 0);
    Serial.print(" Hz)");
    
    if (profiles[i].num_harmonics > 0) {
      Serial.print(" | ");
      Serial.print(profiles[i].num_harmonics);
      Serial.print(" harmonics");
    }
    Serial.println();
  }

  // Calibrate noise floor
  delay(1000);
  calibrateNoiseFloor();

  // Initialize timers
  lastPingTime = millis();
  lastPongTime = millis();
  lastMaxReset = millis();
  
  pixels.setPixelColor(0, pixels.Color(0, 64, 0)); // Green - ready
  pixels.show();
  
  Serial.println("\n✓✓✓ System ready! Listening for drones... ✓✓✓\n");
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
  unsigned long currentTime = millis();

  // Handle serial commands
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'r' || cmd == 'R') {
      recording = true;
      Serial.println("▶ Recording started");
      digitalWrite(LED_BUILTIN, HIGH);
    } else if (cmd == 's' || cmd == 'S') {
      recording = false;
      Serial.println("⏸ Recording stopped");
      digitalWrite(LED_BUILTIN, LOW);
    } else if (cmd == 'c' || cmd == 'C') {
      calibrateNoiseFloor();
    }
  }

  if (recording) {
    // Read audio from I2S
    int samplesAvailable = i2s.available();

    if (samplesAvailable > 0) {
      int samplesToRead = min(samplesAvailable, BUFFER_SIZE);
      
      for (int i = 0; i < samplesToRead; i++) {
        int16_t left = 0, right = 0;
        i2s.read16(&left, &right);
        audioBuffer[i] = left;
      }

      // Accumulate samples for FFT with overlap
      for (int i = 0; i < samplesToRead; i++) {
        if (fftBufferIdx < FFT_SIZE) {
          fftBuffer[fftBufferIdx++] = (float)audioBuffer[i];
        }
        
        // Process when buffer full or at overlap point
        if (fftBufferIdx >= FFT_SIZE) {
          // Copy to FFT working arrays and apply Hamming window
          for (int j = 0; j < FFT_SIZE; j++) {
            float window = 0.54 - 0.46 * cos(2.0 * PI * j / (FFT_SIZE - 1));
            vReal[j] = fftBuffer[j] * window;
            vImag[j] = 0;
          }
          
          // Compute FFT
          unsigned long fftStart = micros();
          FFT.compute(vReal, vImag, FFT_SIZE, FFTDirection::Forward);
          FFT.complexToMagnitude(vReal, vImag, FFT_SIZE);
          unsigned long fftTime = micros() - fftStart;
          
          // Detect drone
          char detectedName[32] = "";
          float confidence = 0;
          float snr = 0;
          
          unsigned long detectStart = micros();
          bool droneDetected = detectDrone(detectedName, &confidence, &snr);
          unsigned long detectTime = micros() - detectStart;
          
          if (droneDetected) {
            // Get profile index
            int profileIdx = -1;
            for (int p = 0; p < NUM_PROFILES; p++) {
              if (strcmp(detectedName, profiles[p].name) == 0) {
                profileIdx = p;
                break;
              }
            }
            
            // Check temporal consistency
            if (profileIdx >= 0 && checkTemporalConsistency(profileIdx, confidence, snr)) {
              // Confirmed detection - send UART message
              if (currentTime - lastDroneSent > 1000) {
                Serial1.println("drone");
                lastDroneSent = currentTime;
                lastDroneTime = currentTime;
                
                Serial.print("🚁 [CONFIRMED] ");
                Serial.print(detectedName);
                Serial.print(" | Confidence: ");
                Serial.print(confidence * 100, 1);
                Serial.print("% | SNR: ");
                Serial.print(snr, 1);
                Serial.print(" | Detections: ");
                Serial.print(consecutiveDetections);
                Serial.println();
              }
            }
          }
          
          // Debug output every 5 seconds
          static unsigned long lastDebug = 0;
          if (millis() - lastDebug > 5000) {
            Serial.print("📊 FFT: ");
            Serial.print(fftTime);
            Serial.print("µs | Detect: ");
            Serial.print(detectTime);
            Serial.print("µs | Status: ");
            Serial.println(droneDetected ? "ANALYZING ⚠️" : "Monitoring ✓");
            lastDebug = millis();
          }
          
          // Shift buffer for overlap (50%)
          memmove(fftBuffer, fftBuffer + FFT_OVERLAP, 
                  FFT_OVERLAP * sizeof(float));
          fftBufferIdx = FFT_OVERLAP;
        }
      }

      // Status LED heartbeat
      if (millis() - lastStatusUpdate > 1000) {
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        lastStatusUpdate = millis();
      }

      // Volume level detection for "loud" signal
      long sum = 0;
      for (int i = 0; i < samplesToRead; i++) {
        long sample = audioBuffer[i];
        sum += sample * sample;
      }
      float rms = sqrt(sum / samplesToRead);
      int volumeLevel = (int)rms;

      // Track max volume over 15 seconds
      if (currentTime - lastMaxReset > 15000) {
        maxVolume = 0;
        lastMaxReset = currentTime;
      }
      if (volumeLevel > maxVolume) {
        maxVolume = volumeLevel;
      }

      // Send loud signal if above threshold
      if (volumeLevel > 0.75 * maxVolume && maxVolume > 100) {
        Serial1.println("loud");
        lastLoudTime = currentTime;
      }
    }
  }

  // UART heartbeat
  if (currentTime - lastPingTime > PING_INTERVAL_MS) {
    Serial1.println("ping");
    lastPingTime = currentTime;
  }

  // Read UART responses
  while (Serial1.available()) {
    String msg = Serial1.readStringUntil('\n');
    msg.trim();
    
    if (msg.length() == 0) continue;
    
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

  // Update NeoPixel (priority: drone > loud > connected > disconnected)
  if (currentTime - lastDroneTime < DRONE_ALERT_DURATION) {
    // Drone detected: YELLOW
    pixels.setPixelColor(0, pixels.Color(255, 255, 0));
  } else if (currentTime - lastLoudTime < LOUD_DURATION) {
    // Loud sound: BLUE
    pixels.setPixelColor(0, pixels.Color(0, 0, 127));
  } else if (currentTime - lastPongTime < CONNECTION_TIMEOUT) {
    // Connected: GREEN
    pixels.setPixelColor(0, pixels.Color(0, 64, 0));
  } else {
    // Disconnected: RED
    pixels.setPixelColor(0, pixels.Color(64, 0, 0));
  }
  pixels.show();

  delay(1);
}