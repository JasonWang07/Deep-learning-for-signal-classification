# Modulation Classification with MATLAB

This implements a system for modulation classification using synthetic data and neural networks:

---

## **1. Overview**
This code generates synthetic data for various modulation types, simulates channel effects, and processes these signals for training a neural network to classify modulation types. It also includes an evaluation on a test dataset.

---

## **2. Key Components**

### **Modulation Types**
This supports several modulation schemes:
- **Digital:** BPSK, QPSK, 8PSK, 16QAM, 32QAM, 64QAM
- **Analog:** GFSK, B-FM, DSB-AM, SSB-AM

### **Data Generation**
- Random bit sequences are generated for PAM-4 modulation.
- Channel impairments are simulated, including:
  - Multipath fading
  - Doppler shift
  - Frequency and clock offsets
  - Noise

### **Data Storage**
- Signals are saved in a temporary directory to allow reuse without regenerating the dataset.

### **Data Preparation**
- Frames are split into training, validation, and test sets to ensure robust evaluation.

---

## **3. Model Training**

### **Supported Architectures**
The script supports multiple neural network architectures for modulation classification:
- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- ResNet (Residual Network)
- DenseNet
- CLDNN (Convolutional, LSTM, and Dense Network)

The default model is CNN, which can be changed by modifying this line:
```matlab
modClassNet = modelCNN(modulationTypes, sps, spf);
