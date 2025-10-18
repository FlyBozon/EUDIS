#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <Arduino.h>
#include <cmath>
#include <cstring>

/**
 * MelSpectrogramGenerator - OPTIMIZED for ESP32-S3 (8MB PSRAM)
 * 
 * Generuje Mel Spectrogram z próbek audio
 * Wymiary wyjścia: 224x224 (zgodne z modelem CNN)
 * Obsługuje PSRAM dla bufora spektrogramu
 */
class MelSpectrogramGenerator {
public:
    static constexpr int SAMPLE_RATE = 44100;
    static constexpr int FFT_SIZE = 2048;
    static constexpr int N_MELS = 64;             // było 128
    static constexpr int MEL_SPEC_WIDTH = 96;     // było 224
    static constexpr int MEL_SPEC_HEIGHT = 96;    // było 224
    static constexpr float FMIN = 50.0f;
    static constexpr float FMAX = 1000.0f;
    
private:
    // FFT buffers
    float fft_real[FFT_SIZE];
    float fft_imag[FFT_SIZE];
    
    // Power spectrum
    float magnitude[FFT_SIZE / 2 + 1];
    
    // Mel bands dla bieżącego ramki
    float mel_bands[N_MELS];
    
    // Output spectrogram (96x96 = 9216 floatów)
    float* spectrogram;
    bool owns_spectrogram;
    
    int frame_count;
    
public:
    MelSpectrogramGenerator() : frame_count(0), owns_spectrogram(true) {
        // Alokuj spektrogram w PSRAM (8MB dostępnego)
    spectrogram = (float*)ps_malloc(MEL_SPEC_HEIGHT * MEL_SPEC_WIDTH * sizeof(float));
        
        if (spectrogram == nullptr) {
            Serial.println("[ERROR] Brak pamięci na spektrogram!");
            owns_spectrogram = false;
        } else {
            memset(spectrogram, 0, MEL_SPEC_HEIGHT * MEL_SPEC_WIDTH * sizeof(float));
        }
        
        memset(fft_real, 0, sizeof(fft_real));
        memset(fft_imag, 0, sizeof(fft_imag));
        memset(magnitude, 0, sizeof(magnitude));
        memset(mel_bands, 0, sizeof(mel_bands));
    }
    
    ~MelSpectrogramGenerator() {
        if (spectrogram != nullptr && owns_spectrogram) {
            free(spectrogram);
        }
    }
    
    bool isInitialized() const {
        return spectrogram != nullptr;
    }
    
    /**
     * Generuj Mel Spectrogram z 2048 próbek audio (46ms @ 44.1kHz)
     */
    const float* generateMelSpectrogram(const int16_t* audio_samples, int num_samples) {
        if (spectrogram == nullptr || num_samples < FFT_SIZE) {
            return spectrogram;
        }
        
        // Krok 1: Okno Hanna
        applyHannWindow(audio_samples);
        
        // Krok 2: FFT
        computeFFT();
        
        // Krok 3: Magnitude spectrum
        computeMagnitude();
        
        // Krok 4: Mel filterbank
        applyMelFilterbank();
        
        // Krok 5: Konwersja do dB
        convertToDb();
        
        // Krok 6: Normalizacja do zakresu [0, 1]
        normalizeToRange();
        
        frame_count++;
        return spectrogram;
    }
    
    const float* getSpectrogram() const {
        return spectrogram;
    }
    
    int getFrameCount() const {
        return frame_count;
    }
    
private:
    // ============ KONWERSJA CZĘSTOTLIWOŚCI ============
    
    float hzToMel(float hz) {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
    }
    
    float melToHz(float mel) {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }
    
    // ============ OKNO ============
    
    void applyHannWindow(const int16_t* input) {
        for (int n = 0; n < FFT_SIZE; n++) {
            float window = 0.5f * (1.0f - cosf(2.0f * M_PI * n / (FFT_SIZE - 1)));
            fft_real[n] = (float)input[n] / 32768.0f * window;
            fft_imag[n] = 0.0f;
        }
    }
    
    // ============ FFT (Cooley-Tukey) ============
    
    void computeFFT() {
        int n = FFT_SIZE;
        
        // Permutacja bit-reversal
        for (int i = 0, j = 0; i < n; i++) {
            if (i < j) {
                float tmp = fft_real[i];
                fft_real[i] = fft_real[j];
                fft_real[j] = tmp;
                
                tmp = fft_imag[i];
                fft_imag[i] = fft_imag[j];
                fft_imag[j] = tmp;
            }
            
            int k = n / 2;
            while (k <= j) {
                j -= k;
                k /= 2;
            }
            j += k;
        }
        
        // Cooley-Tukey FFT
        for (int s = 1; s <= (int)log2f(n); s++) {
            int m = 1 << s;
            float angle = 2.0f * M_PI / m;
            
            for (int k = 0; k < m / 2; k++) {
                float wr = cosf(k * angle);
                float wi = -sinf(k * angle);
                
                for (int j = 0; j < n; j += m) {
                    int idx1 = j + k;
                    int idx2 = j + k + m / 2;
                    
                    float tr = wr * fft_real[idx2] - wi * fft_imag[idx2];
                    float ti = wr * fft_imag[idx2] + wi * fft_real[idx2];
                    
                    fft_real[idx2] = fft_real[idx1] - tr;
                    fft_imag[idx2] = fft_imag[idx1] - ti;
                    
                    fft_real[idx1] += tr;
                    fft_imag[idx1] += ti;
                }
            }
        }
    }
    
    // ============ MAGNITUDE ============
    
    void computeMagnitude() {
        int n_freqs = FFT_SIZE / 2 + 1;
        
        for (int k = 0; k < n_freqs; k++) {
            float real = fft_real[k];
            float imag = fft_imag[k];
            magnitude[k] = sqrtf(real * real + imag * imag);
        }
    }
    
    // ============ MEL FILTERBANK ============
    
    void applyMelFilterbank() {
        float mel_min = hzToMel(FMIN);
        float mel_max = hzToMel(FMAX);
        
        int n_freqs = FFT_SIZE / 2 + 1;
        
        // Punkty Mel scale
        float mel_points[N_MELS + 2];
        float hz_points[N_MELS + 2];
        
        for (int i = 0; i < N_MELS + 2; i++) {
            mel_points[i] = mel_min + (mel_max - mel_min) * i / (N_MELS + 1);
            hz_points[i] = melToHz(mel_points[i]);
        }
        
        // Dla każdego Mel pasma, zastosuj trójkątny filtr
        for (int m = 0; m < N_MELS; m++) {
            float sum = 0.0f;
            
            float left_hz = hz_points[m];
            float center_hz = hz_points[m + 1];
            float right_hz = hz_points[m + 2];
            
            // Konwolucja z trójkątnym filtrem
            for (int k = 0; k < n_freqs; k++) {
                float hz = (float)k * SAMPLE_RATE / FFT_SIZE;
                
                float weight = 0.0f;
                if (hz > left_hz && hz < right_hz) {
                    if (hz < center_hz) {
                        weight = (hz - left_hz) / (center_hz - left_hz);
                    } else {
                        weight = (right_hz - hz) / (right_hz - center_hz);
                    }
                }
                
                sum += weight * magnitude[k];
            }
            
            // Power spectrum
            mel_bands[m] = sum * sum;
        }
    }
    
    // ============ KONWERSJA dB ============
    
    void convertToDb() {
        for (int m = 0; m < N_MELS; m++) {
            float db = 10.0f * log10f(mel_bands[m] + 1e-10f);
            mel_bands[m] = db;
        }
    }
    
    // ============ NORMALIZACJA ============
    
    void normalizeToRange() {
        // Min/max
        float current_min = mel_bands[0];
        float current_max = mel_bands[0];
        
        for (int m = 1; m < N_MELS; m++) {
            current_min = fminf(current_min, mel_bands[m]);
            current_max = fmaxf(current_max, mel_bands[m]);
        }
        
        float range = current_max - current_min + 1e-10f;
        for (int m = 0; m < N_MELS; m++) {
            mel_bands[m] = (mel_bands[m] - current_min) / range;
            mel_bands[m] = fmaxf(0.0f, fminf(1.0f, mel_bands[m]));
        }
        
        // Wypełnij spektrogram 224x224
        for (int row = 0; row < MEL_SPEC_HEIGHT; row++) {
            int mel_idx = (row * N_MELS) / MEL_SPEC_HEIGHT;
            if (mel_idx >= N_MELS) mel_idx = N_MELS - 1;
            
            for (int col = 0; col < MEL_SPEC_WIDTH; col++) {
                spectrogram[row * MEL_SPEC_WIDTH + col] = mel_bands[mel_idx];
            }
        }
    }
};

#endif // MEL_SPECTROGRAM_H
