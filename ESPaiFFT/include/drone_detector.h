#ifndef DRONE_DETECTOR_H
#define DRONE_DETECTOR_H

#include <Arduino.h>
#include "mel_spectrogram.h"

#ifdef USE_TFLITE
#include <tflm_esp32.h>
// Różne dystrybucje biblioteki mają różne nazwy nagłówków
#if __has_include(<EloquentTinyML.h>)
    #include <EloquentTinyML.h>
    #if __has_include("eloquent_tinyml/tf.h")
        #include "eloquent_tinyml/tf.h"
        #define HAS_ELOQUENT_TINYML 1
    #else
        #define HAS_ELOQUENT_TINYML 0
    #endif
#elif __has_include(<eloquent_tinyml.h>)
    #include <eloquent_tinyml.h>
    #if __has_include("eloquent_tinyml/tf.h")
        #include "eloquent_tinyml/tf.h"
        #define HAS_ELOQUENT_TINYML 1
    #else
        #define HAS_ELOQUENT_TINYML 0
    #endif
#else
    #warning "EloquentTinyML not found, TFLite integration will be disabled"
    #define HAS_ELOQUENT_TINYML 0
#endif
#endif

/**
 * Detektor Dronów - ESP32-S3
 * 
 * Pipeline:
 * 1. Zbierz 2048 próbek audio (46ms @ 44.1kHz)
 * 2. Generuj Mel Spectrogram (96x96)
 * 3. Analiza heurystyczna spektrogramu
 * 4. Zwróć confidence score [0, 1]
 * 
 * NOTATKA: Gdy TFLite będzie dostępny, zamień heurystykę na model
 */
class DroneDetector {
private:
    // State
    bool initialized;
    float drone_confidence;
    MelSpectrogramGenerator* mel_spec_gen;
    
    // Konfiguracja
    static constexpr int FFT_SIZE = 2048;
    
public:
    DroneDetector() 
        : initialized(false), drone_confidence(0.0f), mel_spec_gen(nullptr) {
    }
    
    ~DroneDetector() {
        cleanup();
    }
    
    bool initialize(const uint8_t* model_data = nullptr) {
        Serial.println("[DRONE] Inicjalizacja detektora dronów...");
        
        // Inicjalizuj generator spektrogramu
        mel_spec_gen = new MelSpectrogramGenerator();
        if (mel_spec_gen == nullptr || !mel_spec_gen->isInitialized()) {
            Serial.println("[ERROR] Brak pamięci na spektrogram!");
            cleanup();
            return false;
        }
        Serial.println("[DRONE] Generator spektrogramu zainicjalizowany");
        
    #if defined(USE_TFLITE) && HAS_ELOQUENT_TINYML
        // Spróbuj zainicjalizować TFLite (EloquentTinyML)
        if (model_data != nullptr) {
            Serial.println("[DRONE] Inicjalizuję TensorFlow Lite Micro...");
            // Konfiguracja: przewidujemy wyjście 2 klasy, wejście 224x224=50176 floatów
            // Uwaga: Arena musi być duża (PSRAM). EloquentTinyML wykorzysta lokalną arenę,
            // ale na ESP32-S3 z PSRAM to i tak limit. Przy dużym modelu może się nie zmieścić.
            tflm_enabled = tf_begin(model_data);
            if (tflm_enabled) {
                Serial.println("[DRONE] ✓ TFLite gotowe (będzie użyte do inferencji)");
            } else {
                Serial.println("[DRONE] ⚠ Nie udało się zainicjalizować TFLite. Fallback do heurystyki");
            }
        } else {
            Serial.println("[DRONE] Brak danych modelu TFLite → używam heurystyki");
        }
    #else
        Serial.println("[DRONE] Używam analizy heurystycznej");
        Serial.println("[DRONE] TODO: Załadować model TFLite...");
        #endif
        
        initialized = true;
        Serial.println("[DRONE] ✓ Detektor gotowy!");
        return true;
    }
    
    /**
     * Analizuj próbki audio i zwróć confidence dla drona
     * 
     * Heurystyka oparta na cechach spektrogramu drona:
     * - Energia w pasmach niskofrequencyjnych (80-250 Hz)
     * - Wąskie szczyty w spektrogramie
     * - Energia skoncentrowana w specyficznych pasmach
     */
    float detectDrone(const int16_t* audio_buffer, uint16_t buffer_size) {
        if (!initialized || mel_spec_gen == nullptr) {
            drone_confidence = 0.0f;
            return 0.0f;
        }
        
        if (buffer_size < FFT_SIZE) {
            return 0.0f;
        }
        
    // Generuj Mel Spectrogram (MEL_SPEC_HEIGHT x MEL_SPEC_WIDTH)
        unsigned long start = millis();
        const float* spectrogram = mel_spec_gen->generateMelSpectrogram(audio_buffer, buffer_size);
        unsigned long spec_time = millis() - start;
        
        if (spectrogram == nullptr) {
            Serial.println("[ERROR] Spektrogram nie wygenerowany!");
            return 0.0f;
        }
        
        #if defined(USE_TFLITE) && HAS_ELOQUENT_TINYML
        if (tflm_enabled) {
            // Uwaga: nasz docelowy model oczekuje 96x96x1 wejścia INT8 (0..1 skalowane do kwantyzacji).
            // W prostym wariancie spłaszczamy do wektora i przekazujemy do modelu jako float (predictInt8 zrobi kwantyzację).
            drone_confidence = run_tflite_inference(spectrogram);
        } else {
            drone_confidence = analyzeSpectrogram(spectrogram);
        }
        #else
        // Analiza heurystyczna spektrogramu
        drone_confidence = analyzeSpectrogram(spectrogram);
        #endif
        
        // Debug
        if (false) {  // Set to true dla debug
            Serial.printf("[DRONE] Spec gen: %ld ms, Conf: %.2f%%\n",
                         spec_time, drone_confidence * 100);
        }
        
        return drone_confidence;
    }
    
    float getLastConfidence() const {
        return drone_confidence;
    }
    
    bool isDrone(float threshold = 0.6f) const {
        return drone_confidence >= threshold;
    }
    
private:
    #if defined(USE_TFLITE) && HAS_ELOQUENT_TINYML
    // Minimalna implementacja integracji z EloquentTinyML
    static constexpr uint8_t TF_NUM_OPS = 10; // dostosuj wg potrzeby (Conv, DepthwiseConv, Add, Relu, MaxPool, FC, Softmax)
    static constexpr size_t TF_TENSOR_ARENA = 200 * 1024; // ~200KB arena: bezpieczniejsze dla DRAM
    Eloquent::TF::Sequential<TF_NUM_OPS, TF_TENSOR_ARENA> tf;
    bool tflm_enabled = false;

    bool tf_begin(const unsigned char* model_data) {
        tf.setNumInputs(MelSpectrogramGenerator::MEL_SPEC_WIDTH * MelSpectrogramGenerator::MEL_SPEC_HEIGHT); // spłaszczone 2D → 1D
        tf.setNumOutputs(2);

        auto &err = tf.begin(model_data);
        if (err) {
            Serial.printf("[TFLM] Init error: %s\n", err.toCString());
            return false;
        }
        return true;
    }

    float run_tflite_inference(const float* spectrogram) {
        // Spłaszcz dane [224x224] do bufora wejściowego
        // Uwaga: to kopiuje 50176 floatów. Można zoptymalizować poprzez bezpośrednie zapisanie do tf.in->data.f
        static float input_buffer[MelSpectrogramGenerator::MEL_SPEC_WIDTH * MelSpectrogramGenerator::MEL_SPEC_HEIGHT];
        for (int i = 0; i < MelSpectrogramGenerator::MEL_SPEC_WIDTH * MelSpectrogramGenerator::MEL_SPEC_HEIGHT; i++) input_buffer[i] = spectrogram[i];

    auto &err = tf.predictInt8(input_buffer);
        if (err) {
            Serial.printf("[TFLM] Invoke error: %s\n", err.toCString());
            return analyzeSpectrogram(spectrogram); // fallback
        }

        // Załóż: wyjścia [drone, not_drone] lub odwrotnie
        float p0 = tf.output(0);
        float p1 = tf.output(1);
        // Zwróć prawdopodobieństwo drona (przyjmijmy indeks 0 jako dron)
        float conf = p0;
        // sanity clamp
        if (isnan(conf) || conf < 0.0f || conf > 1.0f) conf = 0.0f;
        return conf;
    }
    #endif
    /**
     * Analiza heurystyczna spektrogramu dla detekacji drona
     * Szuka charakterystycznych cech dźwięku drona:
     * - Niska częstotliwość (rzędy propelerów)
     * - Okresowe modulacje
     * - Wąskie pasma energii
     */
    float analyzeSpectrogram(const float* spectrogram) {
        if (spectrogram == nullptr) return 0.0f;
        
        float confidence = 0.0f;
        
    // Cecha 1: Energia w pasmach niskofrequencyjnych (dolne 50% rzędów)
        // Dron = silna energia w 50-500 Hz
        float low_freq_energy = 0.0f;
    int low_freq_rows = MelSpectrogramGenerator::MEL_SPEC_HEIGHT / 2;  // dolne 50%
        
        for (int i = 0; i < low_freq_rows * MelSpectrogramGenerator::MEL_SPEC_WIDTH; i++) {
            low_freq_energy += spectrogram[i];
        }
        low_freq_energy /= (low_freq_rows * MelSpectrogramGenerator::MEL_SPEC_WIDTH);
        
        // Normalizuj do [0, 1]
        low_freq_energy = fminf(1.0f, low_freq_energy);
        confidence += low_freq_energy * 0.3f;  // 30% wagi
        
        // Cecha 2: Wariancja energii (szczyty vs doliny)
        // Drony mają wysoki kontrast w spektrogramie
        float mean_val = 0.0f;
        for (int i = 0; i < low_freq_rows * MelSpectrogramGenerator::MEL_SPEC_WIDTH; i++) {
            mean_val += spectrogram[i];
        }
        mean_val /= (low_freq_rows * MelSpectrogramGenerator::MEL_SPEC_WIDTH);
        
        float variance = 0.0f;
        for (int i = 0; i < low_freq_rows * MelSpectrogramGenerator::MEL_SPEC_WIDTH; i++) {
            float diff = spectrogram[i] - mean_val;
            variance += diff * diff;
        }
        variance /= (low_freq_rows * MelSpectrogramGenerator::MEL_SPEC_WIDTH);
        float stddev = sqrtf(variance);
        
        // Normalizuj wariancję
        float variance_score = fminf(1.0f, stddev * 2);
        confidence += variance_score * 0.3f;  // 30% wagi
        
        // Cecha 3: Energia w specyficznym pasme dla dronów (100-250 Hz)
        // Drony mają dominantę w tym paśmie
        float drone_band_energy = 0.0f;
    int drone_band_start = low_freq_rows / 2;   // ~100 Hz przy nowej skali (heurystyka)
    int drone_band_end = low_freq_rows - 1;    // ~250 Hz
        
        for (int row = drone_band_start; row < drone_band_end; row++) {
            for (int col = 0; col < MelSpectrogramGenerator::MEL_SPEC_WIDTH; col++) {
                drone_band_energy += spectrogram[row * MelSpectrogramGenerator::MEL_SPEC_WIDTH + col];
            }
        }
        drone_band_energy /= ((drone_band_end - drone_band_start) * MelSpectrogramGenerator::MEL_SPEC_WIDTH);
        
        float band_score = fminf(1.0f, drone_band_energy);
        confidence += band_score * 0.4f;  // 40% wagi
        
        // Zablokuj do [0, 1]
        drone_confidence = fmaxf(0.0f, fminf(1.0f, confidence));
        
        return drone_confidence;
    }
    
    void cleanup() {
        if (mel_spec_gen != nullptr) {
            delete mel_spec_gen;
            mel_spec_gen = nullptr;
        }
        initialized = false;
    }
};

#endif // DRONE_DETECTOR_H
