## 1. Explain & Implement DFT IDFT

# import numpy as np
# import matplotlib.pyplot as plt

# plt.style.use('ggplot')

# def dft(x):
#     N = len(x)
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     e = np.exp(-2j * np.pi * k * n / N)
#     X = np.dot(e, x)
#     return X

# def idft(X):
#     N = len(X)
#     k = np.arange(N)
#     n = k.reshape((N, 1))
#     e = np.exp(2j * np.pi * k * n / N)
#     x = np.dot(e, X) / N
#     return x

# N = 1024
# fs = 100
# T = 1 / fs
# k = np.arange(N)
# f = 0.25 + 2 * np.sin(2 * np.pi * 5 * k * T) + 1 * np.sin(2 * np.pi * 12.5 * k * T) + 1.5 * np.sin(2 * np.pi * 20 * k * T) + 0.5 * np.sin(2 * np.pi * 35 * k * T)

# X = dft(f)
# plt.figure(figsize=(8, 6))

# plt.subplot(3, 1, 1)
# plt.plot(k, f)
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.title('Original Signal')

# plt.subplot(3, 2, 3)
# plt.plot(np.abs(X))
# plt.xlabel('Frequency Bin')
# plt.ylabel('Magnitude')
# plt.title('Magnitude Spectrum')

# plt.subplot(3, 2, 4)
# plt.plot(np.angle(X))
# plt.xlabel('Frequency Bin')
# plt.ylabel('Phase (radians)')
# plt.title('Phase Spectrum')

# x_reconstructed = idft(X)
# plt.subplot(3, 1, 3)
# plt.plot(k, x_reconstructed.real)
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.title('IDFT')

# plt.tight_layout()
# plt.show()

#-------------------------------------------------------------------------
## 2. Design an FIR filter to meet the following s
# pecification passband edge= 2 KHz, StopBand edge=5KHz,
# Fs=20KHz, Filter length=21, use Hanning window in the design

# import matplotlib.pyplot as plt
# from scipy.signal import firwin, freqz
# import numpy as np

# # Filter specifications
# passband_edge = 2e3  # Passband edge in Hz
# stopband_edge = 5e3  # Stopband edge in Hz
# Fs = 20e3  # Sampling frequency in Hz
# filter_length = 21  # Filter length
# window_type = 'hann'  # Hanning window

# # Calculate normalized frequencies
# passband_edge_normalized = passband_edge / (Fs / 2)
# stopband_edge_normalized = stopband_edge / (Fs / 2)

# # Design the FIR filter
# taps = firwin(filter_length, [passband_edge_normalized, stopband_edge_normalized], pass_zero=False, window=window_type)

# # Frequency response of the filter
# w, h = freqz(taps, worN=8000, fs=Fs)

# # Plot the filter response
# plt.plot(0.5 * Fs * w / np.pi, np.abs(h), 'b')
# plt.title('FIR Filter Frequency Response')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Gain')
# plt.grid()
# plt.show()

#----------------------------------------------------------------------------
## 3.  Create a signal 's' with three sinusoidal components (at 5,15,30 Hz) 
# and a time vector 't' of 100 samples with a sampling rate of 100 Hz, and displaying it 
# in the time domain. Design an IIR filter to suppress frequencies of 5 Hz and 30 Hz 
# from given signal

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, lfilter

# # Create a signal with three sinusoidal components
# Fs = 100  # Sampling frequency in Hz
# t = np.linspace(0, 1, 100, endpoint=False)  # Time vector
# s = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 30 * t)

# # Plot the original signal in the time domain
# plt.figure(figsize=(10, 4))
# plt.subplot(2, 1, 1)
# plt.plot(t, s)
# plt.title('Original Signal')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')

# # Design an IIR filter to suppress frequencies of 5 Hz and 30 Hz
# cutoff_frequency = [4, 31]  # Hz
# order = 4  # Filter order

# # Design the Butterworth filter
# b, a = butter(order, cutoff_frequency, btype='bandstop', fs=Fs)

# # Apply the filter to the signal
# filtered_signal = lfilter(b, a, s)

# # Plot the filtered signal in the time domain
# plt.subplot(2, 1, 2)
# plt.plot(t, filtered_signal)
# plt.title('Filtered Signal')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.show()

#--------------------------------------------------------------------------------
## 01. Sampling

# import numpy as np
# import matplotlib.pyplot as plt

# # Generate the sinusoidal signal
# f = 10 # signal frequency in Hz
# fs = 200 # sampling frequency in Hz
# t = np.arange(0, 1, 1/fs)
# x = np.sin(2*np.pi*f*t)

# # Plot the original signal
# plt.subplot(3,1,1)
# plt.plot(t, x)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Original Signal')

# # Sample the signal
# Ts = 1/fs  # Sampling interval (in seconds)
# n = np.arange(0, 1, Ts)
# xn = np.sin(2*np.pi*f*n)

# # Plot the sampled signal
# plt.subplot(3,1,2)
# plt.stem(n, xn)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Sampled Signal')

# # Reconstruct the analog signal using ideal reconstruction
# xr = np.zeros_like(t)  # Initialize the reconstructed signal
# for i in range(len(n)):
#     xr += xn[i] * np.sinc((t - i*Ts) / Ts)

# # Plot the reconstructed signal
# plt.subplot(3,1,3)
# plt.plot(t, xr)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Reconstructed Signal')

# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------
## 02. Z Transform

# import scipy.signal as sig
# import numpy as np
# import matplotlib.pyplot as plt
# from lcapy.discretetime import n

# x = 1 / 16 ** n
# Z = x.ZT()
# print(Z)

# x = Z.IZT()
# print(x)

# num = [1, 0, 0, 1]
# den = [1, 0, 2, 0, 1]
# [z, p, k] = sig.tf2zpk(num, den)
# plt.subplot(1, 1, 1)
# plt.scatter(np.real(z), np.imag(z), edgecolors='b', marker='o')
# plt.scatter(np.real(p), np.imag(p), color='b', marker='x')
# plt.show()

# ---------------------------------------------------------------------
## 03. DFT-IDFT

# import numpy as np
# import matplotlib.pyplot as plt

# # MATLAB-style plot formatting
# plt.style.use('ggplot')

# # Define variables
# n = np.arange(-1, 4)
# x = np.arange(1, 6)
# N = len(n)
# k = np.arange(len(n))

# # Calculate Fourier transform
# X = np.sum(x * np.exp(-2j * np.pi * np.outer(n, k) / N), axis=1)
# magX = np.abs(X)
# angX = np.angle(X)
# realX = np.real(X)
# imagX = np.imag(X)

# # Plot Fourier transform components
# fig, axs = plt.subplots(2, 2, figsize=(8, 6))
# axs[0, 0].plot(k, magX)
# axs[0, 0].grid(True)
# axs[0, 0].set_xlabel('Frequency in pi units')
# axs[0, 0].set_title('Magnitude part')

# axs[0, 1].plot(k, angX)
# axs[0, 1].grid(True)
# axs[0, 1].set_xlabel('Frequency in pi units')
# axs[0, 1].set_title('Angle part')

# axs[1, 0].plot(k, realX)
# axs[1, 0].grid(True)
# axs[1, 0].set_xlabel('Frequency in pi units')
# axs[1, 0].set_title('Real part')

# axs[1, 1].plot(k, imagX)
# axs[1, 1].grid(True)
# axs[1, 1].set_xlabel('Frequency in pi units')
# axs[1, 1].set_title('Imaginary part')

# plt.tight_layout()
# plt.show()

# # Perform FFT on signal
# fs = 128
# N = 256
# T = 1 / fs
# k = np.arange(N)
# time = k * T
# f = 0.25 + 2 * np.sin(2 * np.pi * 5 * k * T) + 1 * np.sin(2 * np.pi * 12.5 * k * T) + 1.5 * np.sin(
#     2 * np.pi * 20 * k * T) + 0.5 * np.sin(2 * np.pi * 35 * k * T)

# # Plot original signal
# fig, axs = plt.subplots(2, 1, figsize=(8, 6))
# axs[0].plot(time, f)
# axs[0].set_title('Signal sampled at 128Hz')

# # Calculate FFT and plot frequency components
# F = np.fft.fft(f)
# magF = np.abs(np.hstack((F[0] / N, F[1:N // 2] / (N / 2))))
# hertz = k[0:N // 2] * (1 / (N * T))
# axs[1].stem(hertz, magF)
# axs[1].set_title('Frequency Components')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------
## 04. FIR(LowPass) 

# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# fs = 8000  # sampling rate
# N = 50  # order of filer
# fc = 1200  # cutoff frequency
# b = sig.firwin(N + 1, fc, fs=fs, window='hamming', pass_zero='lowpass')
# w, h_freq = sig.freqz(b, fs=fs)
# z, p, k = sig.tf2zpk(b, 1)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h_freq))  # magnitude
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Magnitude')

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h_freq)))  # phase
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Phase(angel)')

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.show()

# --------------------------------------------------------------------
## 04. FIR(HighPass)

# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# fs = 8000  # sampling rate
# N = 50  # order of filer
# fc = 1200  # cutoff frequency
# b = sig.firwin(N + 1, fc, fs=fs, window='hamming', pass_zero='highpass')
# w, h_freq = sig.freqz(b, fs=fs)
# z, p, k = sig.tf2zpk(b, 1)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h_freq))  # magnitude
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Magnitude')

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h_freq)))  # phase
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Phase(angel)')

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.show()

# --------------------------------------------------------------------
## 04. FIR(BandPass)

# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# fs = 8000  # sampling rate
# N = 50  # order of filer
# fc = np.array([1200, 1800])  # cutoff frequency
# b = sig.firwin(N + 1, fc, fs=fs, window='hamming', pass_zero='bandpass')
# w, h_freq = sig.freqz(b, fs=fs)
# z, p, k = sig.tf2zpk(b, 1)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h_freq))  # magnitude
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Magnitude')

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h_freq)))  # phase
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Phase(angel)')

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.tight_layout()
# plt.show()

# -------------------------------------------------------------------
## 04. FIR(BandStop)

# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# fs = 8000  # sampling rate
# N = 50  # order of filer
# fc = np.array([1200, 2800])  # cutoff frequency
# # wc = 2 * fc / fs  # normalized cutoff frequency to the nyquist frequency
# b = sig.firwin(N + 1, fc, fs=fs, window='hamming', pass_zero='bandstop')
# w, h_freq = sig.freqz(b, fs=fs)
# z, p, k = sig.tf2zpk(b, 1)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h_freq))  # magnitude
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Magnitude')

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h_freq)))  # phase
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Phase(angel)')

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.tight_layout()
# plt.show()

# ----------------------------------------------------------------------
## 04. FIR (MultiBand)

# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# N = 50
# fs = 8000
# fc = np.array([1200, 1400, 2500, 2600])
# b = sig.firwin(N + 1, fc, fs=fs, window='hamming', pass_zero='bandpass')
# w, h_freq = sig.freqz(b, fs=fs)
# z, p, k = sig.tf2zpk(b, 1)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h_freq))  # magnitude
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Magnitude')

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h_freq)))  # phase
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Phase(angel)')

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.show()

# -------------------------------------------------------------------
## 04. FIR (Notch)

# import scipy.signal as sig
# import matplotlib.pyplot as plt
# import numpy as np

# fs = 8000
# N = 50
# fc = np.array([2000, 2050])
# b = sig.firwin(N + 1, fc, fs=fs, window='hamming', pass_zero='bandstop')
# z, p, k = sig.tf2zpk(b, 1)
# w, h_freq = sig.freqz(b, 1, fs=fs)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h_freq))
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Magnitude')

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h_freq)))
# plt.xlabel('frequency(Hz)')
# plt.ylabel('Phase(angel)')

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------
## 04. IIR(LowPass)

# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

# fs = 8000
# n, w = sig.buttord(1200 / 4000, 1500 / 4000, 1, 50)
# [b, a] = sig.butter(n, w)
# w, h = sig.freqz(b, a, 512, fs=8000)
# z, p, k = sig.tf2zpk(b, a)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h))

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h)))

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.tight_layout()
# plt.show()

# -------------------------------------------------------------------
## 04. IIR(HighPass) 

# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# fs = 8000
# [n, w] = sig.buttord(1200 / 4000, 1500 / 4000, 1, 50)
# [b, a] = sig.butter(n, w, btype='highpass')
# w, h = sig.freqz(b, a, 512, fs=fs)
# z, p, k = sig.tf2zpk(b, a)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h))

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h)))

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------
## 04. IIR (BandPass)

# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# fs = 8000
# [n, w] = sig.buttord([1000 / 4000, 2500 / 4000], [400 / 4000, 3200 / 4000], 1, 50)
# [b, a] = sig.butter(n, w, btype='bandpass')
# w, h = sig.freqz(b, a, 512, fs=fs)
# z, p, k = sig.tf2zpk(b, a)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h))

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h)))

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.tight_layout()
# plt.show()

# ----------------------------------------------------------------------
## 04. IIR(BandStop)

# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# fs = 8000
# [n, w] = sig.buttord([1000 / 4000, 2500 / 4000], [400 / 4000, 3200 / 4000], 1, 50)
# [b, a] = sig.butter(n, w, btype='bandstop')
# w, h = sig.freqz(b, a, 512, fs=fs)
# z, p, k = sig.tf2zpk(b, a)

# plt.subplot(3, 1, 1)
# plt.plot(w, np.abs(h))

# plt.subplot(3, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h)))

# plt.subplot(3, 1, 3)
# plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
# plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------

