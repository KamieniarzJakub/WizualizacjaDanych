import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.io import wavfile
import os
from warnings import catch_warnings, simplefilter
import sys

def apply_window(signal):
    """Zastosowanie okna do sygnału przed obliczeniem FFT."""
    window_type = 'hamming'
    if window_type == 'hann':
        window = np.hanning(len(signal))  # Okno Hann
    elif window_type == 'hamming':
        window = np.hamming(len(signal))  # Okno Hamming
    elif window_type == 'blackman':
        window = np.blackman(len(signal))  # Okno Blackman-Harris
    else:
        raise ValueError(f"Nieobsługiwany typ okna: {window_type}")
    
    return signal * window  # Mnożenie sygnału przez okno

def calculate_fft(signal, samplerate, min_f0=70, max_freq=250):
    """Oblicza FFT i HPS sygnału oraz wykrywa płeć na podstawie analizy częstotliwości."""
    n = len(signal)
    
    # Zastosowanie okna przed FFT
    signal_windowed = apply_window(signal)

    # FFT sygnału
    signal_fft = fft(signal_windowed)
    signal_fft = np.abs(signal_fft)  # Moduł widma
    freqs = fftfreq(n, 1 / samplerate)  # Częstotliwości odpowiadające FFT

    # Ograniczamy zakres częstotliwości do min_f0 i max_freq
    freq_limit_min = int(n * min_f0 / samplerate)  # Próg minimalnej częstotliwości
    freq_limit_max = int(n * max_freq / samplerate)  # Próg maksymalnej częstotliwości

    # Obliczenie HPS
    hps = harmonic_product_spectrum(signal_fft, n)

    # Wykrycie płci na podstawie HPS
    return detect_gender(hps, freqs, samplerate, freq_limit_min, freq_limit_max)

def harmonic_product_spectrum(signal_fft, n, n_harmonics=4):
    """Oblicza Harmonic Product Spectrum (HPS) dla sygnału."""  
    hps = np.copy(signal_fft)
    
    for h in range(2, n_harmonics + 1):
        # Wydzielanie harmonik i mnożenie widm
        harmonic = np.roll(signal_fft, int(n / h))  # Przesuwanie widma o n/h
        hps[:len(harmonic)] *= harmonic[:len(harmonic)]  # Mnożenie widm
        
    return hps

def draw(freqs, signal_fft, hps, freq_limit):
    """Rysuje wykresy: widmo FFT i HPS dla sygnału."""  
    fig = plt.figure(figsize=(15, 10), dpi=80)

    # Widmo sygnału w skali liniowej
    ax1 = fig.add_subplot(311)
    ax1.plot(freqs[:freq_limit], signal_fft[:freq_limit], '-', label="Spektrum (skala liniowa)")
    ax1.set_title("Spektrum skala liniowa")
    ax1.set_xlabel("Częstotliwość [Hz]")
    ax1.set_ylabel("Amplituda")
    ax1.legend()
    ax1.grid()

    # Widmo HPS
    ax2 = fig.add_subplot(312)
    ax2.plot(freqs[:freq_limit], hps[:freq_limit], '-', label="Spektrum HPS")
    ax2.set_title("Spektrum Harmonic Product Spectrum (HPS)")
    ax2.set_xlabel("Częstotliwość [Hz]")
    ax2.set_ylabel("Amplituda")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()

def detect_gender(signal_fft, freqs, samplerate, freq_limit_min, freq_limit_max):
    """Analizuje HPS i wykrywa płeć na podstawie podstawowej częstotliwości F0."""  
    # Ograniczenie do zakresu częstotliwości między min_f0 a max_freq
    relevant_freqs = freqs[freq_limit_min:freq_limit_max]
    relevant_fft = signal_fft[freq_limit_min:freq_limit_max]

    # Znajdź maksymalny punkt w widmie w tym zakresie
    f0 = relevant_freqs[np.argmax(relevant_fft)]  # Podstawowa częstotliwość w określonym zakresie

    # Sprawdzenie, czy F0 jest większe niż minimalna wartość (np. 50 Hz)
    if f0 < 50:
        print(f"F0 jest zbyt niskie ({f0} Hz), traktujemy to jako sygnał nieprawidłowy.")
        return "Unknown"  # Sygnał uznany za nieprawidłowy

    # Zakładamy, że średnia częstotliwość F0 dla mężczyzn to około 120 Hz, dla kobiet to około 200 Hz
    if f0 < 180:
        return "M"  # Mężczyzna
    else:
        return "K"  # Kobieta

def read_wave(file_path):
    """Wczytuje plik WAV i wywołuje funkcję do analizy FFT oraz wykrywania płci."""
    with catch_warnings():
        simplefilter("ignore")
        # Wczytywanie pliku wav
        samplerate, signal = wavfile.read(file_path)

    # Sprawdzenie liczby kanałów (mono lub stereo)
    if len(np.shape(signal)) > 1:
        signal = signal.mean(axis=1)  # Zmiana na mono, jeśli plik jest stereo

    # Analiza FFT i wykrycie płci
    return calculate_fft(signal, samplerate)

def extract_label_from_filename(filename):
    """Ekstrahuje etykietę (płeć) z nazwy pliku na podstawie litery 'M' lub 'K'."""  
    for char in filename:
        if char in ('K', 'M'):
            return char
    return 'Unknown'

def calculate_accuracy(directory):
    """Oblicza dokładność rozpoznawania płci na podstawie plików WAV w folderze."""
    correct = 0
    total = 0

    for file in os.scandir(directory):
        if file.is_file():
            try:
                predicted_label = read_wave(file.path)
                actual_label = extract_label_from_filename(file.name)
                if predicted_label == actual_label:
                    correct += 1
                else:
                    print(f"{file.name} Rozpoznana: {predicted_label}/{actual_label}")
                    print()
                total += 1
            except Exception as e:
                print(f"Error with file {file.name}: {e}")
                continue

    return correct / total if total > 0 else 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        label = read_wave(path)
        print(f"Predicted label: {label}")
    else:
        # Testowanie dokładności na zbiorze treningowym z oknem Hamming
        print(f"Dokładność: {calculate_accuracy('train/') * 100:.2f}%")
