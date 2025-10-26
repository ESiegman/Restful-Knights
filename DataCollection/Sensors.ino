/**
 * @file Sensors.ino
 * @brief Arduino sketch for reading EEG, ECG, and SpO₂ data and streaming as CSV.
 */

#include <Wire.h>

// --- MAX30102 Definitions ---
#define MAX30102_ADDR 0x57
#define REG_FIFO_WR_PTR  0x04
#define REG_OVF_COUNTER  0x05
#define REG_FIFO_RD_PTR  0x06
#define REG_FIFO_DATA    0x07
#define REG_FIFO_CONFIG  0x08
#define REG_MODE_CONFIG  0x09
#define REG_SPO2_CONFIG  0x0A
#define REG_LED1_PA      0x0C
#define REG_LED2_PA      0x0D

#define SAMPLE_RATE_HZ 100
#define BUFFER_SIZE 100

uint32_t redBuffer[BUFFER_SIZE];
uint32_t irBuffer[BUFFER_SIZE];
int bufferIndex = 0;
unsigned long lastSampleTime = 0;

float spo2 = 0;

/**
 * @brief Writes a value to a MAX30102 register.
 * @param reg Register address.
 * @param value Value to write.
 */
void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

/**
 * @brief Reads FIFO data from MAX30102 sensor.
 * @param red Reference to store red LED value.
 * @param ir Reference to store IR LED value.
 * @return True if read successful, false otherwise.
 */
bool readFIFO(uint32_t &red, uint32_t &ir) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(REG_FIFO_DATA);
  Wire.endTransmission(false);
  Wire.requestFrom(MAX30102_ADDR, (uint8_t)6);
  if (Wire.available() < 6) return false;
  uint32_t rawRed = ((uint32_t)Wire.read() << 16) |
                    ((uint32_t)Wire.read() << 8) |
                    ((uint32_t)Wire.read());
  uint32_t rawIR  = ((uint32_t)Wire.read() << 16) |
                    ((uint32_t)Wire.read() << 8) |
                    ((uint32_t)Wire.read());
  rawRed &= 0x3FFFF;
  rawIR  &= 0x3FFFF;
  red = rawRed;
  ir = rawIR;
  return true;
}

/**
 * @brief Initializes the MAX30102 sensor.
 */
void setupMAX30102() {
  writeRegister(REG_MODE_CONFIG, 0x40); delay(100);
  writeRegister(REG_FIFO_WR_PTR, 0x00);
  writeRegister(REG_OVF_COUNTER, 0x00);
  writeRegister(REG_FIFO_RD_PTR, 0x00);
  writeRegister(REG_FIFO_CONFIG, 0x4F);   // sample avg = 4
  writeRegister(REG_MODE_CONFIG, 0x03);   // SpO2 mode
  writeRegister(REG_SPO2_CONFIG, 0x27);   // 100 Hz, 18-bit
  writeRegister(REG_LED1_PA, 0x24);       // RED LED current
  writeRegister(REG_LED2_PA, 0x24);       // IR LED current
}

/**
 * @brief Computes SpO₂ value from buffer data.
 */
void computeSpO2() {
  // Compute DC components
  double meanRed = 0, meanIR = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    meanRed += redBuffer[i];
    meanIR  += irBuffer[i];
  }
  meanRed /= BUFFER_SIZE;
  meanIR  /= BUFFER_SIZE;

  // Compute AC components (RMS deviation)
  double acRed = 0, acIR = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    acRed += pow(redBuffer[i] - meanRed, 2);
    acIR  += pow(irBuffer[i] - meanIR, 2);
  }
  acRed = sqrt(acRed / BUFFER_SIZE);
  acIR  = sqrt(acIR / BUFFER_SIZE);

  // Ratio of ratios
  double R = (acRed / meanRed) / (acIR / meanIR);
  spo2 = 110.0 - 25.0 * R;
  if (spo2 > 100) spo2 = 100;
  if (spo2 < 0) spo2 = 0;
}

// --- EEG/ECG pins ---
const int eegPin = A0;
const int ecgPin = A1;

/**
 * @brief Arduino setup function. Initializes serial, analog, and sensor.
 */
void setup() {
  Serial.begin(152000); // High baud rate for fast streaming
  analogReadResolution(12);
  analogSetPinAttenuation(A1, ADC_11db);
  analogSetPinAttenuation(A0, ADC_11db);
  Wire.begin();
  setupMAX30102();
  Serial.println("timestamp,EEG,ECG,SpO2");
}

/**
 * @brief Arduino loop function. Samples sensors and prints CSV data.
 */
void loop() {
  unsigned long now = micros();

  // Only sample every 10ms (100Hz)
  if (now - lastSampleTime < 10000) return;
  lastSampleTime = now;

  // Read analog sensors
  int eeg = analogRead(eegPin);
  int ecg = analogRead(ecgPin);

  // Read MAX30102 FIFO
  uint32_t red, ir;
  if (readFIFO(red, ir)) {
    redBuffer[bufferIndex] = red;
    irBuffer[bufferIndex]  = ir;
    bufferIndex++;

    if (bufferIndex >= BUFFER_SIZE) {
      bufferIndex = 0;
      computeSpO2();
    }
  }

  // Print CSV: timestamp,EEG,ECG,SpO2
  Serial.print(now);      // Print full microseconds timestamp
  Serial.print(",");
  Serial.print(eeg);
  Serial.print(",");
  Serial.print(ecg);
  Serial.print(",");
  Serial.println(spo2, 1);
}