#!/usr/bin/env python3
"""THE LAST MEMORY OF RAIN — original score + sound design synthesis.
Stereo 44.1 kHz. No dialogue. Near-silence is a designed element."""
import math, os, sys, wave
import numpy as np

SR = 44100
DUR = 165.0
N = int(SR * DUR)
stereo = np.zeros((N, 2), dtype=np.float64)

def pan_note(freq, t0, dur, amp=0.30, pan=0.0, bright=1.0, decay=2.2):
    n0 = int(t0 * SR); n1 = min(N, int((t0 + dur) * SR))
    if n1 <= n0: return
    t = np.arange(n1 - n0) / SR
    env = np.exp(-decay * t / max(dur, 0.5)) * np.minimum(1.0, t * 180.0 + 0.02)
    sig = np.zeros(n1 - n0)
    for h, w in [(1, 1.0), (2, 0.52 * bright), (3, 0.28 * bright), (4, 0.15 * bright),
                 (5, 0.085 * bright), (6, 0.05), (7, 0.03), (8, 0.018)]:
        sig += w * np.sin(2 * np.pi * freq * h * t + h * 0.7)
    sig *= env * amp
    g = pan * 0.5
    gl, gr = math.cos((g + 0.5) * math.pi / 2), math.sin((g + 0.5) * math.pi / 2)
    L = sig * gl * 0.94 + 0.06 * np.roll(sig, 3)
    R = sig * gr * 0.94 + 0.06 * np.roll(sig, -3)
    stereo[n0:n1, 0] += L; stereo[n0:n1, 1] += R

def low_drone(t0, t1, freq=55.0, amp=0.16):
    n0, n1 = int(t0 * SR), min(N, int(t1 * SR))
    t = np.arange(n1 - n0) / SR
    lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t)
    sig = (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * 1.5 * t + 1.0)
           + 0.25 * np.sin(2 * np.pi * freq * 2.02 * t + 2.1)) * lfo * amp
    stereo[n0:n1, 0] += sig; stereo[n0:n1, 1] += sig

def sub_pulse(t0, t1, freq=41.2, bpm=52, amp=0.22):
    beat = 60.0 / bpm
    t = t0
    while t < t1:
        n0 = int(t * SR); n1 = min(N, n0 + int(1.1 * SR))
        tt = np.arange(n1 - n0) / SR
        env = np.exp(-4.5 * tt) * np.minimum(1.0, tt * 90 + 0.05)
        s = (np.sin(2 * np.pi * freq * tt) + 0.35 * np.sin(2 * np.pi * freq * 2 * tt)) * env * amp
        stereo[n0:n1, 0] += s; stereo[n0:n1, 1] += s
        t += beat

def noise_buf(t0, t1, amp=0.1, kind="brown", attack=0.5, release=2.0):
    n0, n1 = int(t0 * SR), min(N, int(t1 * SR))
    n = n1 - n0
    if n <= 0: return
    rng = np.random.default_rng(int(t0 * 977) & 0xffffffff)
    x = rng.standard_normal(n)
    if kind == "brown":
        x = np.cumsum(x); x /= (np.max(np.abs(x)) + 1e-9)
        x = np.convolve(x, np.ones(64) / 64, mode="same")
    elif kind == "pinkish":
        x = np.convolve(x, np.ones(18) / 18, mode="same")
    t = np.arange(n) / SR
    env = np.minimum(1.0, t / max(attack, 0.01)) * np.minimum(1.0, (n - np.arange(n)) / SR / max(release, 0.01))
    stereo[n0:n1, 0] += x * env * amp; stereo[n0:n1, 1] += x * env * amp * 0.92

def thunder(t0, amp=0.5):
    n0 = int(t0 * SR); n = int(4.2 * SR); n1 = min(N, n0 + n)
    rng = np.random.default_rng(1234)
    x = np.cumsum(rng.standard_normal(n1 - n0))
    x = np.convolve(x, np.ones(256) / 256, mode="same")
    x /= (np.max(np.abs(x)) + 1e-9)
    t = np.arange(x.size) / SR
    env = np.exp(-1.1 * t) * np.minimum(1.0, t * 30 + 0.03)
    sub = np.sin(2 * np.pi * 36.0 * t * (1.0 - 0.18 * np.exp(-3 * t))) * np.exp(-1.6 * t)
    s = (x * 0.65 + sub * 0.85) * env * amp
    stereo[n0:n1, 0] += s * 0.9; stereo[n0:n1, 1] += s

def click(t0, amp=0.35, freq=2600.0, dur=0.055):
    n0 = int(t0 * SR); n = int(dur * SR); n1 = min(N, n0 + n)
    t = np.arange(n1 - n0) / SR
    s = np.sin(2 * np.pi * freq * t) * np.exp(-t * 70) * amp
    s += np.random.default_rng(7).standard_normal(s.size) * np.exp(-t * 120) * amp * 0.4
    stereo[n0:n1, 0] += s; stereo[n0:n1, 1] += s

def hiss(t0, t1, amp=0.05):
    n0, n1 = int(t0 * SR), min(N, int(t1 * SR))
    x = np.random.default_rng(99).standard_normal(n1 - n0)
    x = x - np.convolve(x, np.ones(64) / 64, mode="same")
    t = np.arange(x.size) / SR
    env = np.minimum(1.0, t * 8) * np.minimum(1.0, (x.size - np.arange(x.size)) / SR / 0.4)
    stereo[n0:n1, 0] += x * env * amp; stereo[n0:n1, 1] += x * env * amp

def chime(t0, amp=0.22):
    for f, a, d in [(1318.5, 1.0, 5.5), (1975.5, 0.55, 3.6), (2637.0, 0.3, 2.2)]:
        pan_note(f, t0, 4.0, amp=amp * a, pan=0.15, bright=0.6, decay=d)

def rain_bed(t0, t1, amp=0.085):
    n0, n1 = int(t0 * SR), min(N, int(t1 * SR))
    n = n1 - n0
    rng = np.random.default_rng(20250607)
    x = rng.standard_normal(n)
    x = np.convolve(x, np.ones(10) / 10, mode="same") * 0.6 + np.convolve(x, np.ones(120) / 120, mode="same") * 0.9
    x /= (np.max(np.abs(x)) + 1e-9)
    t = np.arange(n) / SR
    attack = np.clip(t / 6.0, 0, 1)
    ease = np.clip((n / SR - t) / 18.0, 0, 1)
    amp_mod = 1.0 + 0.22 * np.sin(2 * np.pi * 0.35 * t) + 0.13 * np.sin(2 * np.pi * 0.09 * t + 1.3)
    s = x * attack * ease * amp_mod * amp
    stereo[n0:n1, 0] += s; stereo[n0:n1, 1] += s * 0.95

def rocket_rumble(t0, t1, amp=0.4):
    n0, n1 = int(t0 * SR), min(N, int(t1 * SR))
    n = n1 - n0
    rng = np.random.default_rng(555)
    x = np.cumsum(rng.standard_normal(n)); x = x - np.convolve(x, np.ones(512) / 512, mode="same")
    x /= (np.max(np.abs(x)) + 1e-9)
    t = np.arange(n) / SR
    env = np.tanh(t * 3.0) * np.clip((n / SR - t) / 1.6, 0, 1)
    s = x * 0.75 * env * amp
    crack = (rng.standard_normal(n) > 2.6) * rng.standard_normal(n) * np.exp(-((np.arange(n) / SR) % 0.08) * 40) * 0.35
    s = (s + crack * env) * 1.15
    stereo[n0:n1, 0] += s; stereo[n0:n1, 1] += s * 0.93

D4, E4, F4, G4, A4, C5, D5, F5, D6 = 293.66, 329.63, 349.23, 392.00, 440.00, 523.25, 587.33, 698.46, 1174.66
A2, D3, F3, G3 = 110.0, 146.83, 174.61, 196.00

low_drone(0.0, 42.0, 55.0, 0.10)
low_drone(96.0, 118.0, 55.0, 0.06)
low_drone(150.0, 165.0, 73.42, 0.055)
sub_pulse(37.0, 88.0, 41.2, bpm=50, amp=0.16)

pan_note(D4, 4.2, 6.0, 0.16, pan=-0.35, bright=0.9, decay=1.6)
pan_note(A4, 10.0, 5.5, 0.12, pan=0.3, bright=0.8, decay=1.4)

motif = [(D4, 17.0, 1.5), (F4, 19.1, 1.4), (A4, 21.3, 2.2), (G4, 25.6, 1.2),
         (F4, 27.2, 1.2), (E4, 29.0, 1.5), (D4, 31.0, 3.4)]
for f, t0, d in motif: pan_note(f, t0, d, 0.24, pan=-0.15 + (t0 % 1) * 0.5 - 0.25, bright=1.0)
low_drone(16.5, 42.0, D3 / 2, 0.075)
pan_note(D3, 17.0, 8.0, 0.10, pan=0.0, bright=0.5, decay=1.1)
pan_note(F3, 25.6, 7.0, 0.09, pan=-0.2, bright=0.5, decay=1.1)
pan_note(G3, 29.0, 6.0, 0.085, pan=0.2, bright=0.5, decay=1.1)

pan_note(A4, 39.0, 2.6, 0.20, pan=0.25, bright=0.85)
pan_note(F4, 43.5, 2.0, 0.19, pan=-0.25, bright=0.8)
pan_note(D4, 47.5, 4.5, 0.17, pan=0.0, bright=0.7, decay=1.5)
pan_note(F5, 51.5, 3.0, 0.10, pan=0.35, bright=1.2, decay=2.0)

pan_note(A4, 57.5, 4.0, 0.16, pan=0.0, bright=0.8, decay=0.9)
pan_note(A4, 71.0, 6.5, 0.17, pan=0.0, bright=0.85, decay=0.7)
sub_pulse(56.0, 88.0, 38.89, bpm=56, amp=0.13)
click(42.05, 0.30, 1500.0, 0.07)     # drawer relay
hiss(57.8, 58.6, 0.045)             # terminal wake
click(58.55, 0.22, 2400.0, 0.04)    # terminal chime click
chime(67.2, 0.20)                   # 61% resolves
click(87.6, 0.40, 900.0, 0.09)      # core seated
noise_buf(88.6, 96.0, amp=0.10, kind="pinkish", attack=0.25, release=3.0)
rocket_rumble(88.4, 100.6, 0.42)
pan_note(D6, 110.2, 3.2, 0.30, pan=0.0, bright=1.5, decay=3.2)
click(110.22, 0.18, 3400.0, 0.03)
thunder(110.9, 0.5)
noise_buf(115.0, 121.5, amp=0.16, kind="brown", attack=4.0, release=8.0)
rain_bed(111.5, 165.0, amp=0.075)
for f, t0, d in [(D4, 150.5, 2.0), (F4, 152.7, 2.0), (A4, 154.9, 2.4), (D5, 157.6, 5.5)]:
    pan_note(f, t0, d, 0.20, pan=0.0, bright=1.0, decay=1.5)
low_drone(150.0, 165.0, D3 / 2, 0.08)

peak = np.max(np.abs(stereo)) + 1e-9
if peak > 0.94:
    stereo *= 0.94 / peak
stereo = np.tanh(stereo * 1.12) * 0.92
fade_in = int(1.2 * SR); fade_out = int(3.5 * SR)
stereo[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
stereo[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None]
out = (stereo * 32767.0).astype(np.int16)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio.wav")
with wave.open(OUT, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(out.tobytes())
print("AUDIO OK -> %s  (%.1f s, stereo, %d Hz)" % (OUT, N / SR, SR))
