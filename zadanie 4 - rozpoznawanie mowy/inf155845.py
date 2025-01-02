#Jakub Kamieniarz 155845 - program testowy w komentarzach

import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate
from scipy.fftpack import fft, fftfreq
import sys
import os
#import shutil #biblioteka do czyszczenia folderów
import matplotlib.pyplot as plt

male_range = (65, 171.5)
female_range = (171.5, 300)

def harmonic_product_spectrum(fy, repeat=4):
    hps = fy.copy()
    components = [fy]
    for k in range(2, repeat + 1):
        downsampled = decimate(fy, k)
        hps[:len(downsampled)] *= downsampled
        components.append(downsampled)
    return hps, components

def predict(path):
    samplerate, signal = wavfile.read(path)
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)
    signal = signal * np.kaiser(len(signal), 10*round(len(signal) / samplerate))
    fx = fftfreq(len(signal), d=1 / samplerate)
    fy = np.abs(fft(signal))
    fy[:male_range[0] // 2] = 0
    hps, components = harmonic_product_spectrum(fy)
    human_range = (fx > male_range[0]) & (fx < female_range[1])
    dominant_freq = fx[human_range][np.argmax(hps[human_range])]
    if male_range[0] < dominant_freq < male_range[1]:
        return 'M', fx, fy, hps, components
    return 'K', fx, fy, hps, components

def draw(freqs, signal_fft, hps, components, save_path=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(freqs[np.argmax(freqs > male_range[0]):np.argmax(freqs > female_range[-1])], 
             signal_fft[np.argmax(freqs > male_range[0]):np.argmax(freqs > female_range[-1])], label="FFT Spectrum")
    ax1.set_title("FFT Spectrum")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid()

    ax2.plot(freqs[np.argmax(freqs > male_range[0]):np.argmax(freqs > female_range[-1])], 
             hps[np.argmax(freqs > male_range[0]):np.argmax(freqs > female_range[-1])], label="HPS Spectrum")
    ax2.set_title("Harmonic Product Spectrum")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid()

    for idx, component in enumerate(components):
        ax3.plot(freqs[np.argmax(freqs > male_range[0]):np.argmax(freqs > female_range[-1])], 
                 component[:len(freqs[np.argmax(freqs > male_range[0]):np.argmax(freqs > female_range[-1])])], label=f"Stage {idx+1}")
    ax3.set_title("HPS Stages")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def clear_directories():
    for folder in ["good_plots", "bad_plots"]:
        #if os.path.exists(folder):
        #    shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def calculate_accuracy_and_draw(directory):
    clear_directories()
    correct = 0
    total = 0
    for file in os.scandir(directory):
        if file.is_file():
            predicted_label, freqs, signal_fft, hps, components = predict(file.path)
            if predicted_label == file.name[-5]:
                correct += 1
                save_path = f"good_plots/{file.name}.png"
            else: 
                f"bad_plots/{file.name}.png"
            total += 1
            draw(freqs, signal_fft, hps, components, save_path=save_path)
    return correct / total if total > 0 else 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        label, *_ = predict(path)
        print(f"{label}", end="")
    else:
        accuracy = calculate_accuracy_and_draw('train/')
        print(f"Accuracy: {accuracy:.4f}")
