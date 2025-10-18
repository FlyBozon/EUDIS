#ifndef AUDIO_INPUT_H
#define AUDIO_INPUT_H

#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_err.h>

/**
 * Adapter audio input dla ESP32-S3 z I2S
 * Obsługuje INMP441 lub podobne mikrofony I2S
 */
class AudioInput {
private:
    static constexpr int SAMPLE_RATE = 44100;
    static constexpr int BITS_PER_SAMPLE = 16;
    static constexpr int CHANNELS = 1;
    static constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
    
    // Piny I2S
    int8_t pin_bclk;
    int8_t pin_ws;
    int8_t pin_din;
    
    bool initialized;
    
public:
    AudioInput(int8_t bclk = 9, int8_t ws = 8, int8_t din = 7)
        : pin_bclk(bclk), pin_ws(ws), pin_din(din), initialized(false) {
    }
    
    bool begin() {
        const i2s_config_t i2s_config = {
            .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
            .sample_rate = SAMPLE_RATE,
            .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
            .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
            .communication_format = I2S_COMM_FORMAT_STAND_I2S,
            .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
            .dma_buf_count = 8,
            .dma_buf_len = 1024,
            .use_apll = false,
            .tx_desc_auto_clear = false,
            .fixed_mclk = 0
        };
        
        const i2s_pin_config_t pin_config = {
            .mck_io_num = -1,
            .bck_io_num = pin_bclk,
            .ws_io_num = pin_ws,
            .data_out_num = -1,
            .data_in_num = pin_din
        };
        
        esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, nullptr);
        if (err != ESP_OK) {
            Serial.printf("[ERROR] I2S driver install failed: %s\n", esp_err_to_name(err));
            return false;
        }
        
        err = i2s_set_pin(I2S_PORT, &pin_config);
        if (err != ESP_OK) {
            Serial.printf("[ERROR] I2S set pin failed: %s\n", esp_err_to_name(err));
            i2s_driver_uninstall(I2S_PORT);
            return false;
        }
        
        initialized = true;
        Serial.println("[INFO] I2S audio input initialized");
        return true;
    }
    
    void end() {
        if (initialized) {
            i2s_driver_uninstall(I2S_PORT);
            initialized = false;
        }
    }
    
    /**
     * Odczytaj próbki audio
     * @param buffer Wskaźnik na bufor int16_t
     * @param samples Liczba próbek do odczytania
     * @return Liczba odczytanych próbek
     */
    size_t read(int16_t* buffer, size_t samples) {
        if (!initialized || buffer == nullptr) {
            return 0;
        }
        
        size_t bytes_read = 0;
        esp_err_t err = i2s_read(I2S_PORT, buffer, samples * sizeof(int16_t), &bytes_read, 100);
        
        if (err == ESP_OK) {
            return bytes_read / sizeof(int16_t);
        }
        
        return 0;
    }
    
    bool isInitialized() const {
        return initialized;
    }
};

#endif // AUDIO_INPUT_H
