#!/usr/bin/env python3
"""Тёплый эмоциональный мелодичный трек на 3:45 в стиле референса
(«Everything Works Out In The End»): си-бемоль минор, Bbm-Db-Fm-Eb,
128 BPM с half-time битом, протяжный саб-бас, супер-сo аккорды, колокольная мелодия.
Мелодия и аранжировка оригинальные.

128 BPM -> такт 1.875 c, 120 тактов = 225 c (3:45).

Запуск:  python3 generate_warm_track.py [out.wav]   (нужен только numpy)
"""
import sys
import wave
from functools import lru_cache

import numpy as np

SR = 44100
BPM = 128
BEAT = 60 / BPM
BAR = BEAT * 4
BARS = 120
TOTAL = BARS * BAR  # 225 c
N = int(round(TOTAL * SR))
rng = np.random.default_rng(11)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tb(bar, beat=0.0):
    return bar * BAR + beat * BEAT


class Bus:
    def __init__(self):
        self.x = np.zeros((N, 2), dtype=np.float32)

    def add(self, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= N:
            return
        if sig.ndim == 1:
            a = (pan + 1) * np.pi / 4
            sig = np.stack([sig * np.cos(a), sig * np.sin(a)], axis=1) * np.sqrt(2)
        j = min(N, i + len(sig))
        self.x[i:j] += (sig[: j - i] * gain).astype(np.float32)


def adsr(n, a, d, s, r):
    e = np.full(n, float(s))
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    if a:
        k = min(a, n)
        e[:k] = np.linspace(0, 1, a)[:k]
    if d and a < n:
        k = min(d, n - a)
        e[a:a + k] = np.linspace(1, s, d)[:k]
    if r:
        k = min(r, n)
        e[n - k:] *= np.linspace(1, 0, k)
    return e


def hp_noise(n, order=2):
    x = rng.standard_normal(n + order)
    for _ in range(order):
        x = np.diff(x)
    return x / 2 ** order


def smooth(x, w):
    c = np.cumsum(np.concatenate([[0.0], x]))
    y = (c[w:] - c[:-w]) / w
    return np.concatenate([y, np.zeros(len(x) - len(y))])


# ----------------------------------------------------------------- звуки (кэшируются)

@lru_cache(None)
def supersaw_chord(notes, dur, bright):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    detunes = (-0.18, -0.11, -0.05, 0.0, 0.05, 0.11, 0.18)
    for m in notes:
        for vi, dt in enumerate(detunes):
            f = hz(m + dt)
            ph = rng.uniform(0, 2 * np.pi)
            v = np.zeros(n)
            for k in range(1, 13):
                if k * f > 12000:
                    break
                v += np.sin(2 * np.pi * k * f * t + ph * k) / k * np.exp(-(k - 1) / bright)
            pan = (vi / (len(detunes) - 1)) * 2 - 1
            a = (pan * 0.9 + 1) * np.pi / 4
            out[:, 0] += v * np.cos(a)
            out[:, 1] += v * np.sin(a)
    out *= adsr(n, 0.02, 0.3, 0.8, 0.25)[:, None]
    return out / (len(notes) * len(detunes)) * 2.2


@lru_cache(None)
def warm_pad(notes, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for m in notes:
        for dt, pan in ((-0.06, -0.8), (0.06, 0.8), (0.0, 0.0)):
            f = hz(m + dt)
            v = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
            a = (pan + 1) * np.pi / 4
            out[:, 0] += v * np.cos(a)
            out[:, 1] += v * np.sin(a)
    out *= adsr(n, 0.8, 0.5, 0.85, 1.0)[:, None]
    return out / (len(notes) * 3)


@lru_cache(None)
def sub808(m, dur, glide_to=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = np.full(n, hz(m))
    if glide_to is not None:  # глайд в последние 0.12 с
        g = int(0.12 * SR)
        f[-g:] = np.geomspace(hz(m), hz(glide_to), g)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    s = np.tanh(s * 1.8) / np.tanh(1.8)
    return s * adsr(n, 0.008, 0.6, 0.75, 0.06)


@lru_cache(None)
def bell(m, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 3)
         + 0.45 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 5)
         + 0.25 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t * 8)
         + 0.18 * np.sin(2 * np.pi * 4.2 * f * t) * np.exp(-t * 12))
    s += 0.25 * np.sin(2 * np.pi * f * t + 2.5 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 6)) * np.exp(-t * 4)
    return s * adsr(n, 0.003, 0, 1, 0.08)


@lru_cache(None)
def lead(m, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t) * np.clip(t * 3 - 0.3, 0, 1)
    out = np.zeros(n)
    for dt in (-0.07, 0.07):
        f = hz(m + dt)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        for k in range(1, 9):
            out += np.sin(k * ph) / k * np.exp(-(k - 1) / 2.2)
    return out * adsr(n, 0.02, 0.2, 0.7, 0.15) * 0.5


@lru_cache(None)
def kick():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 48 + 140 * np.exp(-t * 35)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8)
    s += hp_noise(n) * np.exp(-t * 400) * 0.35
    return np.tanh(s * 2.0)


@lru_cache(None)
def clap():
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    e = np.exp(-t * 14)
    for off in (0.0, 0.009, 0.019):
        k = int(off * SR)
        e[k:k + 250] += 0.7 * np.exp(-np.arange(250) / 50)
    s = hp_noise(n, 1) * e * 0.9 + np.sin(2 * np.pi * 210 * t) * np.exp(-t * 25) * 0.35
    return s


@lru_cache(None)
def hat(open_=False):
    n = int((0.22 if open_ else 0.05) * SR)
    t = np.arange(n) / SR
    return hp_noise(n, 3) * np.exp(-t * (16 if open_ else 80)) * 1.4


@lru_cache(None)
def snap():
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    return hp_noise(n, 2) * np.exp(-t * 45)


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    p = t / dur
    s = hp_noise(n, 1) * p ** 2 * 0.6 + np.sin(2 * np.pi * np.cumsum(220 * 8 ** p) / SR) * p ** 3 * 0.25
    return s


@lru_cache(None)
def impact():
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    f = 32 + 60 * np.exp(-t * 7)
    return np.tanh((np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6)
                    + smooth(rng.standard_normal(n), 5) * np.exp(-t * 6) * 0.4) * 1.4)


@lru_cache(None)
def reverse_cymbal(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (hp_noise(n, 2) * np.exp(-(dur - t) * 3.5))


# ----------------------------------------------------------------- гармония и мелодия
# Bbm - Db - Fm - Eb, по 2 такта
CHORDS = [
    ((58, 61, 65, 70), 46),  # Bbm
    ((56, 61, 65, 68), 49),  # Db  (Ab Db F Ab)
    ((56, 60, 65, 68), 41),  # Fm  (Ab C F Ab)
    ((55, 58, 63, 67), 39),  # Eb  (G Bb Eb G)
]
PAD = [(46, 53, 58, 61), (49, 56, 61, 65), (41, 48, 56, 60), (39, 46, 55, 58)]

# мелодия: 8 тактов, восьмые (64 шага); None = пауза, '-' = продлить
R, H = None, '-'
MELODY = [
    77, H, 73, H, 70, H, 72, 73,   H, H, 72, H, 70, H, 68, H,     # Bbm
    68, H, 73, H, 77, H, 75, 73,   H, H, 75, H, 77, H, 80, H,     # Db
    80, H, 77, H, 75, H, 72, H,    73, H, 72, H, 68, H, H, H,     # Fm
    70, H, 75, H, 77, H, 79, H,    77, H, 75, H, 73, H, 72, H,    # Eb
]
# ответная (вторая) мелодия — выше и спокойнее, для поздних частей
MELODY2 = [
    82, H, H, H, 80, H, 77, H,     H, H, H, H, 77, H, 80, H,
    80, H, H, H, 77, H, 85, H,     H, H, 84, H, 82, H, H, H,
    84, H, H, H, 80, H, 77, H,     H, H, 80, H, 84, H, H, H,
    82, H, H, H, 79, H, 82, H,     H, H, 84, H, 82, H, H, H,
]


def melody_events(seq):
    ev = []
    i = 0
    while i < len(seq):
        m = seq[i]
        if m is None or m == H:
            i += 1
            continue
        j = i + 1
        while j < len(seq) and seq[j] == H:
            j += 1
        ev.append((i * 0.5, m, (j - i) * 0.5))  # (доля, нота, длительность в долях)
        i = j
    return ev


MEL_EV = melody_events(MELODY)
MEL2_EV = melody_events(MELODY2)

# ----------------------------------------------------------------- аранжировка
SECTIONS = [
    (0, 8, "intro"),       # 0:00 пэд + колокольчики
    (8, 24, "verse"),      # 0:15 саб, хэты, снэпы
    (24, 40, "drop1"),     # 0:45 полный бит + суперсо
    (40, 48, "break1"),    # 1:15 провал, только пэд и мелодия
    (48, 56, "build"),     # 1:30 нарастание
    (56, 80, "drop2"),     # 1:45 полный + вторая мелодия
    (80, 88, "break2"),    # 2:30 тихо
    (88, 112, "final"),    # 2:45 кульминация
    (112, 120, "outro"),   # 3:30 затухание
]


def sec(bar):
    for s, e, name in SECTIONS:
        if s <= bar < e:
            return name
    return "outro"


FULL = ("drop1", "drop2", "final")

drums, bass, chords, melody, fx, send = (Bus() for _ in range(6))
kick_times = []

for bar in range(BARS):
    s = sec(bar)
    ci = (bar // 2) % 4
    notes, root = CHORDS[ci]
    first = bar % 2 == 0
    phrase_bar = bar % 8

    # --- пэд (везде, кроме полных частей там тише)
    if first:
        pg = {"intro": 0.55, "verse": 0.5, "break1": 0.65, "build": 0.55, "break2": 0.65,
              "outro": 0.55}.get(s, 0.32)
        p = warm_pad(PAD[ci], 2 * BAR + 1.0)
        chords.add(p, tb(bar), pg)
        send.add(p, tb(bar), pg * 0.4)

    # --- суперсо аккорды: ритмичные стабы в полных частях, длинные в build
    if s in FULL:
        g = 0.42 if s != "final" else 0.48
        for st in (0, 3, 6, 8, 11, 14):  # синкопа на 16-х
            c = supersaw_chord(notes, BEAT * 0.7, 7.0)
            chords.add(c, tb(bar, st / 4), g)
            send.add(c, tb(bar, st / 4), g * 0.25)
    elif s == "build" and first:
        c = supersaw_chord(notes, 2 * BAR, 3.0 + (bar - 48) * 0.6)
        chords.add(c, tb(bar), 0.25 + 0.04 * (bar - 48))

    # --- саб-бас 808: длинная нота на 2 такта с глайдом в следующий аккорд
    if s not in ("intro", "break1", "break2", "outro") and first:
        nxt = CHORDS[(ci + 1) % 4][1]
        gap = phrase_bar == 6  # провал перед новым кругом, как в референсе
        dur = 2 * BAR - (BEAT if gap else 0.02)
        b = sub808(root, round(dur, 4), None if gap else nxt)
        bass.add(b, tb(bar), 0.85 if s in FULL else 0.6)

    # --- барабаны (half-time: кик 1 и «2.75», клэп на 3)
    if s in FULL or s in ("verse", "build"):
        full = s in FULL
        if full or s == "build":
            for st in (0, 10) + ((6,) if phrase_bar % 2 else ()):
                kick_times.append(tb(bar, st / 4))
                drums.add(kick(), tb(bar, st / 4), 1.0)
        if full:
            drums.add(clap(), tb(bar, 2), 0.6)
            send.add(clap(), tb(bar, 2), 0.3)
        elif s == "verse":
            drums.add(snap(), tb(bar, 2), 0.5)
            send.add(snap(), tb(bar, 2), 0.4)
        # хэты
        step = 0.5 if s == "verse" else 0.25 if full else 0.5
        k = 0
        b = 0.0
        while b < 4:
            accent = 0.22 if (b * 2) % 2 == 0 else 0.13
            drums.add(hat(), tb(bar, b), accent, pan=0.25 if k % 2 else -0.15)
            b += step
            k += 1
        # трап-роллы хэтов в конце каждых 4 тактов
        if full and phrase_bar % 4 == 3:
            for r in range(6):
                drums.add(hat(), tb(bar, 3 + r / 6), 0.12 + 0.02 * r, pan=0.3)
        if full and phrase_bar % 2 == 1:
            drums.add(hat(True), tb(bar, 3.5), 0.15, pan=-0.3)

    # --- мелодия (8-тактовая фраза)
    if phrase_bar == 0:
        if s in ("intro", "break1", "break2", "outro", "verse", "build") or s in FULL:
            g = {"intro": 0.30, "verse": 0.30, "break1": 0.38, "build": 0.3, "break2": 0.38,
                 "outro": 0.28}.get(s, 0.34)
            for beat, m, dl in MEL_EV:
                t = tb(bar, beat)
                if t > TOTAL - 1:
                    continue
                bl = bell(m, dl * BEAT + 1.2)
                melody.add(bl, t, g, pan=0.15)
                melody.add(bl, t + BEAT * 0.75, g * 0.3, pan=-0.6)   # дилей 3/16
                melody.add(bl, t + BEAT * 1.5, g * 0.12, pan=0.6)
                send.add(bl, t, g * 0.6)
                if s in FULL:  # подкладка лидом в полных частях
                    ld = lead(m, dl * BEAT)
                    melody.add(ld, t, 0.13, pan=-0.1)
                    send.add(ld, t, 0.08)
        if s in ("drop2", "final", "break2"):
            g = 0.2 if s != "break2" else 0.16
            for beat, m, dl in MEL2_EV:
                t = tb(bar, beat)
                ld = lead(m, dl * BEAT + 0.1)
                melody.add(ld, t, g, pan=0.35)
                send.add(ld, t, g * 0.7)

# --- переходы
for bar in (24, 56, 88):
    fx.add(impact(), tb(bar), 0.7)
    send.add(impact(), tb(bar), 0.25)
for bar in (24, 40, 56, 80, 88, 112):
    rc = reverse_cymbal(BAR)
    fx.add(np.stack([rc, np.roll(rc, 150)], 1), tb(bar) - BAR, 0.35)
r = riser(8 * BAR - BEAT)
fx.add(r, tb(48), 0.45)
send.add(r, tb(48), 0.3)
# снэйр-ролл в build
t0, t1 = tb(52), tb(56) - BEAT
t = t0
while t < t1:
    p = (t - t0) / (t1 - t0)
    drums.add(clap(), t, 0.12 + 0.4 * p)
    t += BEAT / (1 if p < 0.25 else 2 if p < 0.5 else 4 if p < 0.8 else 8)

# ----------------------------------------------------------------- микс
duck = np.ones(N, dtype=np.float32)
dn = int(0.3 * SR)
shape = (1 - 0.65 * np.exp(-np.arange(dn) / SR * 11)).astype(np.float32)
for t in kick_times:
    i = int(t * SR)
    j = min(N, i + dn)
    duck[i:j] = np.minimum(duck[i:j], shape[: j - i])
bass.x *= duck[:, None]
chords.x *= (0.3 + 0.7 * duck)[:, None]


def reverb(x, seconds=2.8):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.stack([smooth(rng.standard_normal(n), 4) * np.exp(-t * 2.4) for _ in range(2)], 1)
    ir[: int(0.025 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(0))
    nfft = 1 << int(np.ceil(np.log2(N + n)))
    out = np.zeros_like(x)
    for ch in range(2):
        out[:, ch] = np.fft.irfft(np.fft.rfft(x[:, ch], nfft) * np.fft.rfft(ir[:, ch], nfft), nfft)[:N]
    return out


wet = reverb(send.x.astype(np.float64))
mix = (drums.x * 0.85 + bass.x * 1.5 + chords.x * 0.7 + melody.x * 0.85 + fx.x * 0.8).astype(np.float64) + wet * 0.5

master = np.interp(np.arange(N) / SR, [0, 2.0, TOTAL - 8, TOTAL], [0.0, 1.0, 1.0, 0.0])
mix *= master[:, None]

# мастеринг: мягкая компрессия (tanh) + нормализация к -1 dBFS
mix /= np.percentile(np.abs(mix), 99.9) / 1.3
mix = np.tanh(mix)
mix *= 0.891 / np.max(np.abs(mix))

out = sys.argv[1] if len(sys.argv) > 1 else "warm_track.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print(f"Готово: {out} ({TOTAL:.2f} c)")
