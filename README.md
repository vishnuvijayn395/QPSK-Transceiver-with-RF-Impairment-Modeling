# QPSK Transceiver with RF Impairment Modeling

## Overview
This project implements a waveform-level QPSK baseband transceiver and demonstrates the impact of practical RF impairments on digital communication signals. The focus is on realistic signal generation, pulse shaping, and visualization of RF non-idealities commonly encountered in Software Defined Radio (SDR) systems.
Unlike theoretical BER simulations, this project emphasizes **signal behavior and distortion analysis** at the waveform level.

## Objectives
- Generate a QPSK baseband signal using Gray coding
- Apply Root Raised Cosine (RRC) pulse shaping
- Model common RF impairments
- Visualize constellation distortion and waveform behavior
- Understand real-world receiver challenges

## System Flow
Random Bits → QPSK Modulation → RRC Pulse Shaping → RF Impairments → Visualization

### QPSK Modulation
- Gray-coded QPSK symbol mapping
- Unit-energy constellation normalization

### Pulse Shaping
- Root Raised Cosine (RRC) filtering
- Oversampled waveform generation
- Bandwidth-controlled baseband signal modeling

### RF Impairments Modeled
- **IQ Imbalance**: Models gain and phase mismatch between I and Q paths
- **Phase Noise**: Introduces random phase variations
- **Carrier Frequency Offset (CFO)**: Causes continuous constellation rotation

### Visualization
- Ideal QPSK constellation
- RF-impaired QPSK constellation
- Time-domain waveform plots

## Key Concepts Demonstrated
- Baseband signal modeling
- Oversampling and pulse shaping
- RF front-end non-idealities
- Constellation-based signal analysis
- Practical SDR-oriented DSP concepts

## Tools & Technologies
- Python
- NumPy
- Matplotlib
- Digital Signal Processing (DSP)

## Notes
This project focuses on waveform-level RF impairment analysis. Bit Error Rate (BER) performance and theoretical 
Validation is handled separately in an independent project to ensure correct Eb/N0 modeling.

## Future Enhancements
- Carrier frequency offset estimation and correction
- IQ imbalance compensation
- Timing recovery
- Extension to higher-order modulation schemes (e.g., 16-QAM)
