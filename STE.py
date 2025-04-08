#!/usr/bin/env python3
"""
STE.py

Implements functions for short-time energy (STE) computation and a simple
note-onset detection based on pitch & energy changes.
"""

import numpy as np

def compute_short_time_energy(signal, frame_size, hop_size):
    """
    Computes the short-time energy (STE) of 'signal' using frames.

    STE(n) = sum over m of [ x(n*hop_size + m) ]^2

    Parameters
    ----------
    signal : 1D numpy array
        Audio samples (mono).
    frame_size : int
        Number of samples per analysis frame.
    hop_size : int
        Number of samples between consecutive frames.

    Returns
    -------
    energies : 1D numpy array
        The STE for each frame.
    """
    num_frames = (len(signal) - frame_size) // hop_size + 1
    energies = np.empty(num_frames)
    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size]
        energies[i] = np.sum(frame.astype(float)**2)
    return energies


def detect_note_onsets(pitches, energies, times,
                       pitch_change_threshold=20.0,
                       energy_ratio_threshold=2.0,
                       min_silence_frames=2):
    """
    Detects onset times of new notes by analyzing changes in pitch and energy.

    Simple heuristic:
      - If energy dips (silence) then rises by a factor of 'energy_ratio_threshold',
        and the pitch changes significantly (by 'pitch_change_threshold' Hz),
        => new note onset.

      - Sudden large pitch jumps can also be treated as new onsets,
        even if energy is relatively stable.

    Parameters
    ----------
    pitches : 1D numpy array
        Detected fundamental frequencies (Hz) per frame.
    energies : 1D numpy array
        Short-time energy values (arbitrary units) per frame.
    times : 1D numpy array
        Time stamps (seconds) for each frame.
    pitch_change_threshold : float
        Minimum absolute pitch difference (Hz) to consider for new note onset.
    energy_ratio_threshold : float
        Factor by which energy must jump (after a dip) to mark new onset.
    min_silence_frames : int
        Minimum consecutive frames of low energy to treat as silence.

    Returns
    -------
    onset_times : list of float
        List of time stamps (seconds) where new notes are detected.
    """
    onset_times = []
    if len(pitches) == 0:
        return onset_times

    # We can consider the very first frame as an onset
    onset_times.append(times[0])

    low_energy = False
    silence_counter = 0

    for i in range(1, len(pitches)):
        pitch_diff = abs(pitches[i] - pitches[i - 1])

        # Check if we have a drop in energy compared to previous frame
        if energies[i] < 0.8 * energies[i - 1]:
            low_energy = True
            silence_counter += 1
        else:
            if low_energy and silence_counter >= min_silence_frames:
                # If energy is coming back up and pitch changes,
                # or if there's a large ratio jump in energy,
                # we mark a new note.
                if pitch_diff > pitch_change_threshold or \
                   energies[i] > (energies[i - 1] * energy_ratio_threshold):
                    onset_times.append(times[i])
            else:
                # Even without silence, a large pitch jump alone can signal a new note
                if pitch_diff > pitch_change_threshold:
                    onset_times.append(times[i])
            # Reset the silence flags
            low_energy = False
            silence_counter = 0

    return onset_times
