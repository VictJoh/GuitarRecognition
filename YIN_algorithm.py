#!/usr/bin/env python3
"""
YIN_algorithm.py

Implements the YIN algorithm for fundamental frequency estimation
plus a convenience function for applying it to an entire signal.

Reference:
- de Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator
  for speech and sounds. The Journal of the Acoustical Society of America, 111(4).
"""

import numpy as np

def yin_pitch(frame, sample_rate, fmin=50, fmax=1000, threshold=0.1):
    """
    Estimates the fundamental frequency of a given audio frame using the YIN algorithm.
    
    Parameters
    ----------
    frame : 1D numpy array
        Audio samples for one frame (windowed).
    sample_rate : int
        Sampling rate in Hz.
    fmin : float
        Minimum expected frequency in Hz.
    fmax : float
        Maximum expected frequency in Hz.
    threshold : float
        Threshold in the cumulative mean normalized difference function.

    Returns
    -------
    f0 : float
        Estimated fundamental frequency (Hz). Returns 0 if no pitch is detected.
    """
    # Determine the valid range of lag values from fmin / fmax.
    tau_min = int(np.floor(sample_rate / fmax))
    tau_max = int(np.floor(sample_rate / fmin))
    if tau_max > len(frame):
        tau_max = len(frame)

    # 1. Calculate squared difference d(tau).
    d = np.zeros(tau_max)
    for tau in range(tau_min, tau_max):
        shifted = frame[tau:]
        unshifted = frame[:len(shifted)]
        d[tau] = np.sum((unshifted - shifted)**2)

    # 2. Cumulative mean normalized difference function d'(tau).
    d_tilde = np.zeros(tau_max)
    d_tilde[0] = 1  # by definition
    running_sum = 0.0
    for tau in range(1, tau_max):
        running_sum += d[tau]
        if running_sum == 0:
            d_tilde[tau] = 1
        else:
            d_tilde[tau] = d[tau] * tau / running_sum

    # 3. Find the first tau where d'(tau) < threshold.
    tau_est = None
    for tau in range(tau_min, tau_max):
        if d_tilde[tau] < threshold:
            # Move to local minimum to avoid subharmonic errors.
            while tau + 1 < tau_max and d_tilde[tau + 1] < d_tilde[tau]:
                tau += 1
            tau_est = tau
            break

    if tau_est is None:
        return 0.0  # no pitch detected

    # 4. Parabolic interpolation around tau_est for refinement.
    if tau_est <= 0 or tau_est >= tau_max - 1:
        refined_tau = tau_est
    else:
        alpha = d_tilde[tau_est - 1]
        beta  = d_tilde[tau_est]
        gamma = d_tilde[tau_est + 1]
        denominator = alpha - 2*beta + gamma
        if denominator == 0:
            delta = 0
        else:
            delta = 0.5 * (alpha - gamma) / denominator
        refined_tau = tau_est + delta

    # Convert period to frequency
    f0 = sample_rate / refined_tau if refined_tau != 0 else 0.0
    return f0


def yin_pitch_detection(signal, sample_rate,
                        frame_size=2048, hop_size=1024,
                        fmin=50, fmax=1000, threshold=0.1):
    """
    Applies the YIN algorithm frame-by-frame to estimate pitch (fundamental frequency) over time.
    
    Parameters
    ----------
    signal : 1D numpy array
        Audio samples (mono).
    sample_rate : int
        Sampling rate in Hz.
    frame_size : int
        Number of samples per frame.
    hop_size : int
        Number of samples between consecutive frames.
    fmin : float
        Minimum expected pitch frequency in Hz.
    fmax : float
        Maximum expected pitch frequency in Hz.
    threshold : float
        Threshold for YIN.

    Returns
    -------
    times : 1D numpy array
        Time stamps (seconds) for each detected pitch.
    pitches : 1D numpy array
        Fundamental frequency (Hz) per frame.
    """
    num_frames = (len(signal) - frame_size) // hop_size + 1
    pitches = []
    times = np.arange(num_frames) * (hop_size / sample_rate)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size].astype(np.float32)

        # Optional: apply a window (like Hanning) to each frame to reduce edge artifacts.
        window = np.hanning(len(frame))
        frame_windowed = frame * window

        f0 = yin_pitch(frame_windowed,
                       sample_rate,
                       fmin=fmin,
                       fmax=fmax,
                       threshold=threshold)
        pitches.append(f0)

    return times, np.array(pitches)
