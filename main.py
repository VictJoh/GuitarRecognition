#!/usr/bin/env python3
"""
main.py

Demonstration script that:
1) Reads a WAV file (mono or stereo).
2) Uses the YIN algorithm for pitch detection (from YIN_algorithm.py).
3) Computes short-time energy & detects note onsets (from STE.py).
4) Converts frequencies to MIDI note numbers, then plots them over time with note names.

Author: (Your Name)
Date: (Today)
"""

import os
import wave
import math
import numpy as np
import matplotlib.pyplot as plt

# Import your own modules
from YIN_algorithm import yin_pitch_detection
from STE import compute_short_time_energy, detect_note_onsets


# -----------------------------------------------------------------------------
#                  WAV Reading (Mono Conversion if Stereo)
# -----------------------------------------------------------------------------
def read_wav(file_path):
    """
    Reads a WAV file and returns:
      - sample_rate : int
      - audio_data  : 1D numpy array (mono, int16)

    If the file is stereo, it is converted to mono by averaging channels.
    """
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        audio_data = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16)

    # Convert stereo → mono if needed
    if num_channels == 2:
        audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)

    return sample_rate, audio_data


# -----------------------------------------------------------------------------
#                 Frequency → MIDI / Note Name Helpers
# -----------------------------------------------------------------------------
def frequency_to_midi(frequency):
    """
    Converts a frequency in Hz to the nearest MIDI note number.
    Returns None if frequency <= 0.
    """
    if frequency <= 0:
        return None
    # MIDI formula: MIDI = 69 + 12*log2(freq / 440)
    return int(round(69 + 12 * math.log2(frequency / 440.0)))


def midi_to_note_name(midi_number):
    """
    Converts a MIDI note number into note name & octave, e.g. 'A4', 'C#5'.
    """
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_index = (midi_number - 12) % 12
    octave = (midi_number // 12) - 1
    return f"{NOTE_NAMES[note_index]}{octave}"


# -----------------------------------------------------------------------------
#                     Plotting Notes (MIDI) Over Time
# -----------------------------------------------------------------------------
def plot_notes_over_time(times, pitches):
    """
    Creates a scatter plot of the pitch data in terms of MIDI note numbers,
    labeling each y-tick with the corresponding note name.

    Parameters
    ----------
    times   : 1D numpy array
        Time stamps (seconds) for each pitch frame.
    pitches : 1D numpy array
        Estimated pitch in Hz for each frame.
    """
    # Convert all frequencies to MIDI
    midi_numbers = []
    for f in pitches:
        m = frequency_to_midi(f)
        if m is None:
            midi_numbers.append(np.nan)  # use NaN to indicate "no pitch"
        else:
            midi_numbers.append(m)
    midi_numbers = np.array(midi_numbers, dtype=float)

    # Build the figure
    plt.figure(figsize=(10, 4))
    plt.title("Detected Notes Over Time (MIDI numbers with note names)")
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Note Number")

    # Plot each (time, MIDI) point
    plt.scatter(times, midi_numbers, c="b", marker='o', label="Detected Notes")

    # If there is at least one valid MIDI note, configure y-ticks with note names
    valid_midi = midi_numbers[~np.isnan(midi_numbers)]
    if len(valid_midi) > 0:
        min_m = int(np.nanmin(valid_midi)) - 1
        max_m = int(np.nanmax(valid_midi)) + 1
        y_ticks = list(range(min_m, max_m + 1))
        y_tick_labels = [midi_to_note_name(m) for m in y_ticks]
        plt.yticks(y_ticks, y_tick_labels)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
#                                   MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # ----------------------------------------------------
    # 1) Pick a WAV file using a path relative to this file
    # ----------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Folder containing main.py
    data_dir = os.path.join(script_dir, "data")             # 'data' subfolder
    file_path = os.path.join(data_dir, "d_major_scale.wav") # Our example file

    sample_rate, audio_data = read_wav(file_path)
    print(f"Loaded '{file_path}' at {sample_rate} Hz, {len(audio_data)} samples.")

    # -------------------------------------------
    # 2) Run YIN pitch detection over the signal
    # -------------------------------------------
    frame_size = 8192
    hop_size   = frame_size // 2  # 50% overlap
    times, pitches = yin_pitch_detection(audio_data,
                                         sample_rate=sample_rate,
                                         frame_size=frame_size,
                                         hop_size=hop_size,
                                         fmin=50,
                                         fmax=1000,
                                         threshold=0.1)

    # ------------------------------
    # 3) Short-Time Energy (STE)
    # ------------------------------
    energies = compute_short_time_energy(audio_data, frame_size, hop_size)

    # Because we want the same number of frames for STE and pitch:
    if len(energies) != len(pitches):
        min_len = min(len(energies), len(pitches))
        energies = energies[:min_len]
        pitches  = pitches[:min_len]
        times    = times[:min_len]

    # ------------------------------
    # 4) Detect Note Onsets
    # ------------------------------
    onset_times = detect_note_onsets(
        pitches,
        energies,
        times,
        pitch_change_threshold=20.0,
        energy_ratio_threshold=2.0,
        min_silence_frames=2
    )

    print("\nDetected Onset Times (s):")
    for t_onset in onset_times:
        print(f"  {t_onset:.3f}")

    # ------------------------------
    # 5) Print Example Note Names
    # ------------------------------
    print("\nFirst 10 pitch frames => note names:")
    for i, freq in enumerate(pitches[:10]):
        midi_val = frequency_to_midi(freq)
        if midi_val is None:
            note_str = "No pitch"
        else:
            note_str = midi_to_note_name(midi_val)
        print(f"  Frame {i} at {times[i]:.3f}s => {freq:.2f} Hz => {note_str}")

    # ------------------------------
    # 6) Plot the pitch in raw Hz
    # ------------------------------
    plt.figure(figsize=(10, 6))

    # Plot raw pitch
    plt.subplot(2, 1, 1)
    plt.title("Pitch Over Time (Hz)")
    plt.plot(times, pitches, 'b.-', label="Pitch (Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()

    # Plot STE
    plt.subplot(2, 1, 2)
    ste_times = np.arange(len(energies)) * (hop_size / sample_rate)
    plt.title("Short-Time Energy")
    plt.plot(ste_times, energies, 'g.-', label="Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------
    # 7) Plot the pitch as MIDI notes
    # ------------------------------
    plot_notes_over_time(times, pitches)
