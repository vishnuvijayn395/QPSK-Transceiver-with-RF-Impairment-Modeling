import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt


# =============================
# IQ Imbalance Compensation (Receiver)
# =============================

def compensate_iq_imbalance(signal):
    """
    Blind IQ imbalance compensation using second-order statistics
    """
    I = np.real(signal)
    Q = np.imag(signal)

    # Remove DC offsets
    I -= np.mean(I)
    Q -= np.mean(Q)

    # Gain normalization
    I /= np.std(I)
    Q /= np.std(Q)

    # Remove I/Q correlation
    rho = np.mean(I * Q)
    Q = Q - rho * I

    return I + 1j * Q


# =============================
# Frequency Offset Estimation (QPSK)
# =============================

def estimate_freq_offset_qpsk(signal):
    """
    Estimate normalized carrier frequency offset using 4th-power method.
    Returns phase increment per sample (rad/sample).
    """
    s4 = signal ** 4
    phase_diff = np.angle(np.mean(s4[1:] * np.conj(s4[:-1])))
    return phase_diff / 4


# =============================
# RF IMPAIRMENT FUNCTIONS
# =============================

def apply_iq_imbalance(signal, gain_imbalance=0.05, phase_imbalance_deg=5):
    phase = np.deg2rad(phase_imbalance_deg)
    i = np.real(signal)
    q = np.imag(signal)

    i_new = (1 + gain_imbalance) * i
    q_new = (1 - gain_imbalance) * (q * np.cos(phase) + i * np.sin(phase))

    return i_new + 1j * q_new


def apply_phase_noise(signal, phase_noise_std=0.01):
    phase_noise = np.random.normal(0, phase_noise_std, len(signal))
    return signal * np.exp(1j * phase_noise)


def apply_frequency_offset(signal, freq_offset, fs):
    n = np.arange(len(signal))
    return signal * np.exp(1j * 2 * np.pi * freq_offset * n / fs)


# Number of QPSK symbols
N = 100000

# Generate random bits (2 bits per symbol)
bits = np.random.randint(0, 2, 2 * N)

# QPSK modulation (Gray coding)
symbols = []
for i in range(0, len(bits), 2):
    b1, b2 = bits[i], bits[i + 1]

    if b1 == 0 and b2 == 0:
        sym = 1 + 1j
    elif b1 == 0 and b2 == 1:
        sym = -1 + 1j
    elif b1 == 1 and b2 == 1:
        sym = -1 - 1j
    else:
        sym = 1 - 1j

    symbols.append(sym)

# Normalize power
symbols = np.array(symbols) / np.sqrt(2)

# Plot constellation
plt.figure()
plt.scatter(symbols.real, symbols.imag)
plt.axhline(0)
plt.axvline(0)
plt.grid(True)
plt.title("QPSK Constellation Diagram")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")


# plt.show()
# =============================
# STEP 2: RRC Pulse Shaping
# =============================

def rrc_filter(beta, span, sps):
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    h = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(t[i]) == 1 / (4 * beta):
            h[i] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (np.sin(np.pi * t[i] * (1 - beta)) +
                   4 * beta * t[i] * np.cos(np.pi * t[i] * (1 + beta)))
            den = (np.pi * t[i] * (1 - (4 * beta * t[i]) ** 2))
            h[i] = num / den

    return h / np.sqrt(np.sum(h ** 2))  # energy normalization


# RRC parameters
beta = 0.25
sps = 8
span = 8

# Generate RRC filter
rrc = rrc_filter(beta, span, sps)
rrc_delay = (len(rrc) - 1) // 2

# Upsample symbols
upsampled = np.zeros(len(symbols) * sps, dtype=complex)
upsampled[::sps] = symbols

# Pulse shaping
tx_signal = np.convolve(upsampled, rrc, mode='same')
# Normalize TX signal power so Es = 1
tx_signal /= np.sqrt(np.mean(np.abs(tx_signal)**2))

fs = sps  # normalized sampling frequency

tx_rf = tx_signal.copy()

tx_rf = apply_iq_imbalance(tx_rf, gain_imbalance=0.08, phase_imbalance_deg=7)
tx_rf = apply_phase_noise(tx_rf, phase_noise_std=0.02)
tx_rf = apply_frequency_offset(tx_rf, freq_offset=0.01, fs=fs)

# Plot waveform
plt.figure()
plt.plot(np.real(tx_signal[:400]))
plt.title("Pulse-Shaped QPSK Signal (Real Part)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)


# plt.show()


# =============================
# STEP 3: AWGN Channel + BER (FINAL)
# =============================
# Mapping sanity check


def add_awgn_symbols(symbols, ebn0_db):
    """
    Add AWGN to QPSK symbols using Eb/N0 (symbol-rate noise)
    """
    ebn0 = 10 ** (ebn0_db / 10)

    bits_per_symbol = 2
    Es = np.mean(np.abs(symbols)**2)        # should be 1
    Eb = Es / bits_per_symbol

    noise_var = Eb / ebn0

    noise = np.sqrt(noise_var / 2) * (
            np.random.randn(len(symbols)) +
            1j * np.random.randn(len(symbols))
    )

    return symbols + noise




ebn0_db_range = np.arange(0, 13, 2)
ber = []

# Gray-coded QPSK (counter-clockwise)
constellation = np.array([
    +1 + 1j,   # 00
    -1 + 1j,   # 01
    -1 - 1j,   # 11
    +1 - 1j    # 10
]) / np.sqrt(2)

bit_map = np.array([
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
])
for i in range(4):
    print(constellation[i], "→", bit_map[i])
print("bit_map shape:", bit_map.shape)
rrc_delay = (len(rrc) - 1) // 2
offset = sps // 2



for ebn0_db in ebn0_db_range:
    if ebn0_db == 12:
        print("TX:", symbols[:5])
        print("RX:", rx_samples[:5])
    # Add noise at SYMBOL RATE
    noisy_symbols = add_awgn_symbols(symbols, ebn0_db)

    # Upsample
    up = np.zeros(len(noisy_symbols) * sps, dtype=complex)
    up[::sps] = noisy_symbols

    # Pulse shaping + matched filter
    tx = np.convolve(up, rrc, mode='same')
    rx = np.convolve(tx, rrc, mode='same')

    # Sample (group delay compensated)
    rrc_delay = (len(rrc) - 1) // 2
    rx_samples = rx[rrc_delay+ sps//2::sps][:len(symbols)]

    # Minimum-distance detection
    detected_bits = []
    for sym in rx_samples:
        idx = np.argmin(np.abs(sym - constellation))
        detected_bits.extend(bit_map[idx])

    detected_bits = np.array(detected_bits)

    min_len = min(len(bits), len(detected_bits))
    errors = np.sum(bits[:min_len] != detected_bits[:min_len])
    ber.append(errors / min_len)






print("Eb/N0 points:", len(ebn0_db_range))
print("BER points:", len(ber))
print("BER:", ber)

print("TX symbols:", symbols[:5])
print("RX samples:", rx_samples[:5])
print("TX bits:", bits[:10])
print("RX bits:", detected_bits[:10])



# 4. Plot BER vs Eb/N0
plt.figure()
plt.semilogy(ebn0_db_range, ber, 'o-', linewidth=2)
plt.grid(True, which='both')
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs Eb/N0 for QPSK over AWGN Channel")
plt.show()


# =============================
# STEP 4: RF IMPAIRMENT MODELS
# =============================

def apply_iq_imbalance(signal, gain_imbalance=0.05, phase_imbalance_deg=5):
    phase = np.deg2rad(phase_imbalance_deg)
    i = np.real(signal)
    q = np.imag(signal)

    i_new = (1 + gain_imbalance) * i
    q_new = (1 - gain_imbalance) * (q * np.cos(phase) + i * np.sin(phase))

    return i_new + 1j * q_new


def apply_phase_noise(signal, phase_noise_std=0.01):
    phase_noise = np.random.normal(0, phase_noise_std, len(signal))
    return signal * np.exp(1j * phase_noise)


def apply_frequency_offset(signal, freq_offset, fs):
    n = np.arange(len(signal))
    return signal * np.exp(1j * 2 * np.pi * freq_offset * n / fs)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(symbols.real, symbols.imag, s=5)
plt.title("Ideal QPSK Constellation")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(tx_rf.real[::sps], tx_rf.imag[::sps], s=5)
plt.title("With RF Impairments")
plt.grid(True)

plt.show()


# =============================
# Frequency Offset Estimation (QPSK)
# =============================

def estimate_freq_offset_qpsk(signal):
    """
    Estimates normalized frequency offset using 4th-power method.
    Returns phase increment per sample (radians/sample).
    """
    s4 = signal ** 4
    phase_diff = np.angle(np.mean(s4[1:] * np.conj(s4[:-1])))
    return phase_diff / 4
