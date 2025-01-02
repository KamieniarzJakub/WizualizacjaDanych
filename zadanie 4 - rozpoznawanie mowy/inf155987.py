import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate
from scipy.fftpack import fft, fftfreq
import os
import sys
import warnings

def read_wav(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samplerate, signal = wavfile.read(path)
    if len(signal.shape) == 2:  # Konwersja do mono
        signal = signal.mean(axis=1)
    return signal, samplerate

def harmonic_product_spectrum(fy, repeat=4):
    fy_copy = fy.copy()
    for k in range(2, repeat+1):
        downsampled = decimate(fy, k)
        fy_copy[:len(downsampled)] *= downsampled
    fy_copy[:10] = 0  # Usunięcie niskich częstotliwości
    return fy_copy

def process_signal(wav):
    fy = wav * np.kaiser(len(wav), 50)
    fy = np.abs(fft(fy))
    return harmonic_product_spectrum(fy)

def predict(path, male_freqs, female_freqs):
    wav, rate = read_wav(path)
    fx = fftfreq(len(wav), d=1/rate)
    fy = process_signal(wav)

    male_range = (85, 171)
    female_range = (171, 250)

    valid_indices = (fx >= male_range[0]) & (fx <= female_range[1])
    dominant_freq = fx[valid_indices][np.argmax(fy[valid_indices])]
    
    if male_range[0] <= dominant_freq < male_range[1]:
        male_freqs.append(dominant_freq)
        return 'M'
    if female_range[0] <= dominant_freq < female_range[1]:
        female_freqs.append(dominant_freq)
        return 'K'
    return 'Unknown'

def extract_label_from_filename(filename):
    for char in filename:
        if char in ('K', 'M'):
            return char
    return 'Unknown'

def calculate_accuracy(directory, male_freqs, female_freqs):
    correct = 0
    total = 0

    for file in os.scandir(directory):
        if file.is_file():
            predicted_label = predict(file.path, male_freqs, female_freqs)
            actual_label = extract_label_from_filename(file.name)
            if predicted_label == actual_label:
                correct += 1
            else:
                print(file.path)
            total += 1

    return correct / total if total > 0 else 0

if __name__ == "__main__":
    male_freqs = []
    female_freqs = []

    if len(sys.argv) > 1:
        path = sys.argv[1]
        label = predict(path, male_freqs, female_freqs)
        print(f"Predicted label: {label}")
    else:
        accuracy = calculate_accuracy('train/', male_freqs, female_freqs)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Male Frequencies: {sorted(male_freqs)}")
        print(f"Female Frequencies: {sorted(female_freqs)}")
