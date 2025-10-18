#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include "esp_heap_caps.h"
#include "audio_input.h"
#include "drone_detector.h"
#include "mel_spectrogram.h"
#if USE_DRONE_MODEL
// Użyj małego nagłówka modelu umieszczonego w katalogu głównym projektu ESPaiFFT
// (duży plik w folderze model/ powoduje przepełnienie pamięci Flash)
#include "../drone_model.h"
#endif

// ============ KONFIGURACJA SPRZĘTU ============

// I2S Pins (INMP441 microphone)
#define I2S_BCLK  9
#define I2S_WS    8
#define I2S_DIN   7

// UART dla drugiego ESP32-S3 (HeartBeat + Alerty)
#define UART_TX   17
#define UART_RX   18
#define UART_BAUD 115200

// NeoPixel
#define NEOPIXEL_PIN 12
#define NUM_PIXELS 1

// ============ AUDIO CONFIGURATION ============
#define SAMPLE_RATE 44100
#define AUDIO_BUFFER_SIZE 512
#define DRONE_INFERENCE_BUFFER_SIZE 2048  // 2048 samples = ~46ms

// ============ GLOBALNE ZMIENNE ============

AudioInput audio_input(I2S_BCLK, I2S_WS, I2S_DIN);
DroneDetector drone_detector;
MelSpectrogramGenerator* mel_spec_gen = nullptr;

int16_t audio_buffer[AUDIO_BUFFER_SIZE];
int16_t drone_inference_buffer[DRONE_INFERENCE_BUFFER_SIZE];
int drone_buffer_idx = 0;

Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

// ============ HEARTBEAT CONFIGURATION ============
const unsigned long PING_INTERVAL_MS = 500;  // Co 500ms
unsigned long last_ping_time = 0;
unsigned long last_pong_time = 0;
const unsigned long PONG_TIMEOUT_MS = 3000;

// ============ DETEKCJA GŁOŚNOŚCI ============
unsigned long last_max_reset = 0;
float max_volume = 0;
unsigned long last_loud_time = 0;
const unsigned long VOLUME_WINDOW_MS = 5000;

// ============ DETEKCJA DRONÓW ============
unsigned long last_drone_detection_time = 0;
unsigned long last_drone_sent = 0;
float drone_threshold = 0.6f;
const unsigned long DRONE_ALERT_DURATION_MS = 5000;

// ============ TIMERY ============
unsigned long last_status_update = 0;

// ============ FORWARD DECLARATIONS ============
void setup_uart();
void handle_uart_messages();
void send_uart_message(const char* message);
void update_neopixel();
void detect_loud_sound(const int16_t* buffer, size_t size, unsigned long current_time);
void detect_drone(unsigned long current_time);
static void print_memory(const char* tag) {
    size_t free_int = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    Serial.printf("[MEM] %-16s | Free INT: %u bytes (%.1f KB) | Free PSRAM: %u bytes (%.1f KB)\n",
                  tag, (unsigned)free_int, free_int/1024.0, (unsigned)free_psram, free_psram/1024.0);
}

// ============ SETUP ============

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) delay(100);
    
    Serial.println("\n\n╔════════════════════════════════════════╗");
    Serial.println("║  EUDIS - Drone Detector ESP32-S3       ║");
    Serial.println("║  TensorFlow Lite + Mel Spectrogram     ║");
    Serial.println("╚════════════════════════════════════════╝\n");
    
    // LED status
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    
    // NeoPixel
    pixels.begin();
    pixels.setPixelColor(0, pixels.Color(255, 0, 0));  // Czerwony
    pixels.show();
    Serial.println("[LED] NeoPixel initialized (Red)");
    
    // I2S Audio Input
    Serial.println("[AUDIO] Initializing I2S...");
    if (!audio_input.begin()) {
        Serial.println("[ERROR] Failed to initialize I2S!");
        while (true) {
            pixels.setPixelColor(0, pixels.Color(255, 0, 0));
            pixels.show();
            delay(200);
        }
    }
    Serial.println("[AUDIO] ✓ I2S ready");
    print_memory("before model");
    
    // Load Drone Model to PSRAM (opcjonalnie)
    #if USE_DRONE_MODEL
    Serial.println("[MODEL] Loading TFLite model to PSRAM...");
    uint8_t* model_buffer = load_drone_model_to_psram();
    if (!model_buffer) {
        Serial.println("[ERROR] Failed to load model to PSRAM!");
        while (true) {
            pixels.setPixelColor(0, pixels.Color(255, 0, 0));
            pixels.show();
            delay(100);
            pixels.setPixelColor(0, pixels.Color(0, 0, 0));
            pixels.show();
            delay(100);
        }
    }
    Serial.printf("[MODEL] ✓ Model loaded to PSRAM: %p\n", model_buffer);
    print_memory("after model");
    #endif
    
    // UART dla drugiego ESP
    Serial.println("[UART] Initializing UART1...");
    setup_uart();
    
    // Inicjalizuj detektor dronów z modelem (jeśli dostępny)
    Serial.println("[MODEL] Loading TFLite model...");
    #if USE_DRONE_MODEL
    if (!drone_detector.initialize(model_buffer)) {
        Serial.println("[WARNING] Drone detector not available");
    } else {
        Serial.println("[MODEL] ✓ TFLite model loaded");
    }
    print_memory("after tf init");
    #else
    if (!drone_detector.initialize(nullptr)) {
        Serial.println("[WARNING] Drone detector not available");
    } else {
        Serial.println("[MODEL] ✓ Heuristic detector loaded");
    }
    print_memory("after tf init");
    #endif
    
    // Inicjalizacja timerów
    last_ping_time = millis();
    last_pong_time = millis();
    last_max_reset = millis();
    
    Serial.println("\n╔════════════════════════════════════════╗");
    Serial.println("║  ✓✓✓ System Ready!                     ║");
    Serial.println("╚════════════════════════════════════════╝\n");
    Serial.println("Waiting for audio input...\n");
}

// ============ MAIN LOOP ============

void loop() {
    unsigned long current_time = millis();
    
    // ========== HEARTBEAT: Wysłij PING co 500ms ==========
    if (current_time - last_ping_time >= PING_INTERVAL_MS) {
        send_uart_message("ping");
        last_ping_time = current_time;
    }
    
    // ========== ODCZYT UART ==========
    handle_uart_messages();
    
    // ========== AUDIO PROCESSING ==========
    if (audio_input.isInitialized()) {
        size_t samples_read = audio_input.read(audio_buffer, AUDIO_BUFFER_SIZE);
        
        if (samples_read > 0) {
            // ===== DETEKCJA GŁOŚNOŚCI =====
            detect_loud_sound(audio_buffer, samples_read, current_time);
            
            // ===== DRONE DETECTION PIPELINE =====
            // Zbierz 2048 próbek dla spektrogramu
            for (size_t i = 0; i < samples_read; i++) {
                if (drone_buffer_idx < DRONE_INFERENCE_BUFFER_SIZE) {
                    drone_inference_buffer[drone_buffer_idx++] = audio_buffer[i];
                }
                
                // Gdy mamy pełen buffer, analizuj
                if (drone_buffer_idx >= DRONE_INFERENCE_BUFFER_SIZE) {
                    detect_drone(current_time);
                    drone_buffer_idx = 0;
                }
            }
            
            // Status LED (blink co 1s)
            if (current_time - last_status_update > 1000) {
                digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
                last_status_update = current_time;
            }
        }
    }
    
    // ========== UPDATE NEOPIXEL ==========
    update_neopixel();
    
    delay(1);
}

// ============ FUNKCJE POMOCNICZE ============

void setup_uart() {
    Serial1.begin(UART_BAUD, SERIAL_8N1, UART_RX, UART_TX);
    delay(100);
    Serial.println("[UART] ✓ UART1 ready");
}

void send_uart_message(const char* message) {
    Serial1.println(message);
}

void handle_uart_messages() {
    while (Serial1.available()) {
        String msg = Serial1.readStringUntil('\n');
        msg.trim();
        
        if (msg.length() == 0) continue;
        
        if (msg == "ping") {
            send_uart_message("pong");
        } 
        else if (msg == "pong") {
            last_pong_time = millis();
        }
        else if (msg == "loud") {
            last_loud_time = millis();
        }
        else if (msg == "drone") {
            last_drone_detection_time = millis();
        }
    }
}

void detect_loud_sound(const int16_t* buffer, size_t size, unsigned long current_time) {
    // Oblicz RMS
    long sum = 0;
    for (size_t i = 0; i < size; i++) {
        long sample = buffer[i];
        sum += sample * sample;
    }
    float rms = sqrt(sum / (float)size);
    int volume_level = (int)rms;
    
    // Reset max co 5 sekund
    if (current_time - last_max_reset > VOLUME_WINDOW_MS) {
        max_volume = 0;
        last_max_reset = current_time;
    }
    
    // Update max
    if (volume_level > max_volume) {
        max_volume = volume_level;
    }
    
    // Jeśli głośny dźwięk (>75% max), wyślij alert
    if (volume_level > 0.75 * max_volume && max_volume > 0) {
        send_uart_message("loud");
        last_loud_time = current_time;
    }
}

void detect_drone(unsigned long current_time) {
    float confidence = drone_detector.detectDrone(
        drone_inference_buffer, 
        DRONE_INFERENCE_BUFFER_SIZE
    );
    
    if (drone_detector.isDrone(drone_threshold)) {
        send_uart_message("drone");
        last_drone_detection_time = current_time;
    }
}

void update_neopixel() {
    unsigned long current_time = millis();
    uint32_t color;
    
    // Priorytet: drone > loud > connected > disconnected
    if (current_time - last_drone_detection_time < DRONE_ALERT_DURATION_MS) {
        // ŻÓŁTY: Drone detected
        color = pixels.Color(255, 255, 0);
    }
    else if (current_time - last_loud_time < 750) {
        // NIEBIESKI: Loud sound detected
        color = pixels.Color(0, 0, 255);
    }
    else if (current_time - last_pong_time < PONG_TIMEOUT_MS) {
        // ZIELONY: Connected
        color = pixels.Color(0, 255, 0);
    }
    else {
        // CZERWONY: Disconnected
        color = pixels.Color(255, 0, 0);
    }
    
    pixels.setPixelColor(0, color);
    pixels.show();
}