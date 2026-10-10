#!/usr/bin/env python3
"""Генератор интригующего саундтрека (3:45) для видео.

D minor, 96 BPM, 90 тактов по 2.5 с = 225 с.
Структура с перепадами: тихое интро -> нарастание -> дроп -> провал ->
повторное нарастание -> большой дроп -> резкая остановка -> кульминация -> аутро.

Запуск:  python3 generate_track.py [out.wav]
Нужен только numpy.
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 96
BEAT = 60 / BPM            # 0.625 c
BAR = BEAT * 4             # 2.5 c
TOTAL = 225.0              # 3:45
N = int(TOTAL * SR)
rng = np.random.default_rng(7)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def tb(bar, beat=0.0):
    """Время в секундах по такту и доле."""
    return bar * BAR + beat * BEAT


class Bus:
    def __init__(self):
        self.x = np.zeros((N, 2), dtype=np.float64)

    def add(self, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= N or len(sig) == 0:
            return
        if sig.ndim == 1:
            l = np.cos((pan + 1) * np.pi / 4)
            r = np.sin((pan + 1) * np.pi / 4)
            sig = np.stack([sig * l, sig * r], axis=1) * np.sqrt(2)
        j = min(N, i + len(sig))
        self.x[i:j] += sig[: j - i] * gain


def env_adsr(n, a, d, s, r):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    e = np.full(n, float(s))
    if a:
        e[: min(a, n)] = np.linspace(0, 1, a)[: min(a, n)]
    if d and a < n:
        k = min(d, n - a)
        e[a: a + k] = np.linspace(1, s, d)[:k]
    if r:
        k = min(r, n)
        e[n - k:] *= np.linspace(1, 0, k)
    return e


def saw_add(f, dur, harm=10, bright=6.0, phase=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    ph = rng.uniform(0, 2 * np.pi) if phase is None else phase
    for k in range(1, harm + 1):
        if k * f > 16000:
            break
        out += np.sin(2 * np.pi * k * f * t + ph * k) / k * np.exp(-(k - 1) / bright)
    return out


def highpass_noise(n, order=2):
    x = rng.standard_normal(n + order)
    for _ in range(order):
        x = np.diff(x)
    return x / (2 ** order)


def smooth(x, w):
    """Грубый low-pass скользящим средним."""
    if w <= 1:
        return x
    c = np.cumsum(np.concatenate([[0.0], x]))
    y = (c[w:] - c[:-w]) / w
    return np.concatenate([y, np.zeros(len(x) - len(y))])


# ---------------------------------------------------------------- инструменты

def kick(gain=1.0):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 7)
    click = highpass_noise(n) * np.exp(-t * 300) * 0.3
    return np.tanh((body + click) * 1.6) * gain


def heartbeat():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 40 + 40 * np.exp(-t * 20)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def clap():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    nz = highpass_noise(n, 1)
    e = np.exp(-t * 18)
    for off in (0.0, 0.011, 0.022):  # тройной "хлопок"
        k = int(off * SR)
        e[k:k + 300] += 0.6 * np.exp(-np.arange(min(300, n - k)) / 60)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.4
    return nz * e * 0.8 + tone


def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    return highpass_noise(n, 3) * np.exp(-t * (14 if open_ else 70)) * 1.3


def tick():
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 3200 * t) * 0.5 + highpass_noise(n, 2)) * np.exp(-t * 160)


def bass_note(midi, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(midi)
    sig = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    return np.tanh(sig * 1.4) * env_adsr(n, 0.004, 0.12, 0.55, 0.05)


def pluck(midi, dur=0.35):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(midi)
    sig = np.zeros(n)
    for k in range(1, 9):
        sig += np.sin(2 * np.pi * k * f * t) / k ** 1.3 * np.exp(-t * (6 + 5 * k))
    return sig * env_adsr(n, 0.002, 0.0, 1.0, 0.02)


def piano(midi, dur=2.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(midi)
    sig = np.zeros(n)
    for k, a in zip(range(1, 7), (1, 0.5, 0.25, 0.15, 0.08, 0.05)):
        sig += a * np.sin(2 * np.pi * k * f * (1 + 0.0004 * k * k) * t) * np.exp(-t * (1.2 + 0.9 * k))
    return sig * env_adsr(n, 0.003, 0, 1, 0.2)


def pad_chord(notes, dur, bright=4.0):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    for m in notes:
        for det, pan in ((-0.07, -0.7), (0.0, 0.0), (0.07, 0.7)):
            v = saw_add(hz(m + det), dur, harm=8, bright=bright)
            l = np.cos((pan + 1) * np.pi / 4)
            r = np.sin((pan + 1) * np.pi / 4)
            out[:, 0] += v * l
            out[:, 1] += v * r
    t = np.arange(n) / SR
    lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6))
    e = env_adsr(n, 1.2, 0.5, 0.85, 1.4) * lfo
    return out * e[:, None] / (len(notes) * 3)


def braam(root=26, dur=4.5):
    """Большой «кинематографичный» низкий удар-бас."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for m, a in ((root, 1.0), (root + 12, 0.7), (root + 19, 0.45), (root + 24, 0.3)):
        f = hz(m)
        for k in range(1, 18):
            if k * f > 9000:
                break
            # верхние гармоники раскрываются и затухают — эффект «рёва»
            hk = np.exp(-t * (0.4 + 0.12 * k)) * (1 - np.exp(-t * (30 / k)))
            out += a * np.sin(2 * np.pi * k * f * t * (1 + 0.002 * np.sin(5 * t))) / k * hk
    return np.tanh(out * 0.9) * env_adsr(n, 0.02, 0, 1, 1.5)


def impact(dur=3.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 30 + 70 * np.exp(-t * 6)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.3)
    nz = smooth(rng.standard_normal(n), 6) * np.exp(-t * 5) * 0.5
    return np.tanh((boom + nz) * 1.5)


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    p = t / dur
    nz = highpass_noise(n, 1) * p ** 2.2
    f = 180 * (12 ** p)
    sweep = np.sin(2 * np.pi * np.cumsum(f) / SR) * p ** 3 * 0.35
    trem = 1 + 0.3 * np.sin(2 * np.pi * (2 + 14 * p) * t)
    return (nz * 0.6 + sweep) * trem


def downlifter(dur=3.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    p = t / dur
    f = 1600 * (0.06 ** p)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3 + highpass_noise(n, 1) * 0.4) * (1 - p) ** 2


def drone(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for m, a, pan in ((26, 1.0, -0.3), (26.08, 0.8, 0.3), (33, 0.45, 0.0), (38, 0.25, 0.5)):
        f = hz(m)
        sw = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t + m)
        v = a * (np.sin(2 * np.pi * f * t) + 0.3 * sw * np.sin(4 * np.pi * f * t) + 0.12 * sw * np.sin(6 * np.pi * f * t))
        out[:, 0] += v * np.cos((pan + 1) * np.pi / 4)
        out[:, 1] += v * np.sin((pan + 1) * np.pi / 4)
    return out


# ---------------------------------------------------------------- аранжировка
# Dm - Bb - Gm - A (по 2 такта)
PROG = [
    (50, 53, 57),  # Dm
    (46, 50, 53),  # Bb
    (43, 46, 50),  # Gm
    (45, 49, 52),  # A
]
ROOTS = [38, 34, 31, 33]


def chord_idx(bar):
    return (bar // 2) % 4


drums, bass, music, fx, rev_send = Bus(), Bus(), Bus(), Bus(), Bus()

# Громкость секций (кривая автоматизации по тактам) для мелодических партий
SECTIONS = [
    # (start_bar, end_bar, name)
    (0, 8, "intro"), (8, 16, "tension"), (16, 24, "build1"), (24, 40, "drop1"),
    (40, 48, "breakdown"), (48, 56, "build2"), (56, 72, "drop2"),
    (72, 74, "stop"), (74, 86, "climax"), (86, 90, "outro"),
]


def sec(bar):
    for s, e, name in SECTIONS:
        if s <= bar < e:
            return name
    return "outro"


# --- дрон на всю длину с автоматизацией
d = drone(TOTAL)
drone_gain = np.interp(
    np.arange(N) / SR,
    [0, tb(4), tb(24), tb(40), tb(48), tb(72), tb(74), tb(86), TOTAL - 4, TOTAL],
    [0.0, 0.22, 0.16, 0.26, 0.18, 0.28, 0.14, 0.2, 0.12, 0.0],
)
music.x += d * drone_gain[:, None]

# --- пэды
for bar in range(0, 90, 2):
    s = sec(bar)
    g = {"intro": 0.0 if bar < 4 else 0.35, "tension": 0.45, "build1": 0.55, "drop1": 0.7,
         "breakdown": 0.5, "build2": 0.55, "drop2": 0.8, "stop": 0.0, "climax": 0.85, "outro": 0.6}[s]
    if g == 0:
        continue
    bright = 3.0 if s in ("intro", "breakdown", "outro") else 6.0
    p = pad_chord(PROG[chord_idx(bar)], BAR * 2 + 1.4, bright)
    music.add(p, tb(bar), g)
    rev_send.add(p, tb(bar), g * 0.5)

# --- тиканье часов (интрига)
for bar in range(2, 24):
    for b in range(4):
        tk = tick()
        fx.add(tk, tb(bar, b), 0.25 if b % 2 == 0 else 0.15, pan=-0.4 if b % 2 else 0.4)
for bar in list(range(40, 48)) + list(range(72, 74)):
    for b in range(4):
        fx.add(tick(), tb(bar, b), 0.28, pan=-0.4 if b % 2 else 0.4)
# замедляющиеся тики в аутро
tt = tb(87)
step = BEAT
while tt < TOTAL - 1.5:
    fx.add(tick(), tt, 0.2)
    tt += step
    step *= 1.18

# --- сердцебиение
for bar in list(range(8, 16)) + list(range(48, 52)) + [72, 73]:
    drums.add(heartbeat(), tb(bar, 0), 0.9)
    drums.add(heartbeat(), tb(bar, 0.6), 0.6)

# --- кик / клэп / хэты
kick_times = []
for bar in range(90):
    s = sec(bar)
    if s in ("drop1", "drop2", "climax"):
        for b in range(4):
            kick_times.append(tb(bar, b))
            drums.add(kick(), tb(bar, b), 1.0)
        for b in (1, 3):
            c = clap()
            drums.add(c, tb(bar, b), 0.55)
            rev_send.add(c, tb(bar, b), 0.25)
        for h in range(16):
            if h % 4 == 2:
                drums.add(hat(True), tb(bar, h / 4), 0.22, pan=0.25)
            else:
                drums.add(hat(), tb(bar, h / 4), 0.16 if h % 2 else 0.24, pan=-0.2 if h % 2 else 0.2)
    elif s in ("build1", "build2") and bar not in (48, 49, 50, 51):
        # полутемп, нарастает
        for b in (0, 2):
            kick_times.append(tb(bar, b))
            drums.add(kick(0.8), tb(bar, b), 1.0)
        for h in range(8):
            drums.add(hat(), tb(bar, h / 2 + 0.5 * 0), 0.12 + 0.01 * (bar % 8), pan=0.2)

# --- снэйр-роллы перед дропами (ускоряются)
for start_bar, length in ((22, 2), (54, 2), (84, 2)):
    t0, t1 = tb(start_bar), tb(start_bar + length) - BEAT  # пауза на последнюю долю
    t = t0
    while t < t1:
        p = (t - t0) / (t1 - t0)
        c = clap()
        drums.add(c, t, 0.15 + 0.45 * p)
        t += BEAT / (2 if p < 0.25 else 4 if p < 0.6 else 8 if p < 0.85 else 16)

# --- райзеры
for start_bar, length in ((20, 4), (52, 4), (82, 4)):
    r = riser(length * BAR - BEAT)
    fx.add(np.stack([r, np.roll(r, 200)], axis=1), tb(start_bar), 0.55)
    rev_send.add(r, tb(start_bar), 0.3)

# --- удары и брэмы на ключевых точках
for bar, big in ((0, False), (24, True), (32, False), (40, False), (56, True), (60, False), (64, True),
                 (68, False), (72, True), (74, True), (78, False), (82, False), (86, True)):
    im = impact(4.0 if big else 3.0)
    fx.add(im, tb(bar), 0.9 if big else 0.6)
    rev_send.add(im, tb(bar), 0.35)
    if bar in (24, 56, 64, 72, 74, 86):
        br = braam(26 if bar != 64 else 22, 6.5 if bar == 86 else 4.5)
        music.add(br, tb(bar), 0.55)
        rev_send.add(br, tb(bar), 0.35)
for bar in (40,):
    fx.add(downlifter(), tb(bar), 0.6)
    rev_send.add(downlifter(), tb(bar), 0.3)

# --- бас (восьмые, по корню аккорда)
for bar in range(90):
    s = sec(bar)
    if s in ("intro", "breakdown", "stop", "outro") or bar in (48, 49):
        continue
    root = ROOTS[chord_idx(bar)]
    g = {"tension": 0.45, "build1": 0.55, "build2": 0.55, "drop1": 0.75, "drop2": 0.8, "climax": 0.85}[s]
    for e in range(8):
        if bar in (23, 55, 85) and e >= 6:  # пауза перед дропом
            continue
        m = root + (12 if e % 4 == 3 and s in ("drop2", "climax") else 0)
        bass.add(bass_note(m, BEAT / 2), tb(bar, e / 2), g)

# --- арпеджио (шестнадцатые)
ARP_PATTERN = [0, 1, 2, 1, 0, 2, 1, 2]
for bar in range(12, 90):
    s = sec(bar)
    if s in ("breakdown", "stop", "outro") or bar in (48, 49, 50, 51):
        continue
    g = {"tension": 0.18, "build1": 0.3, "build2": 0.3, "drop1": 0.4, "drop2": 0.45, "climax": 0.5}[s]
    notes = PROG[chord_idx(bar)]
    octv = 24 if s in ("drop2", "climax") else 12
    for h in range(16):
        if bar in (23, 55, 85) and h >= 12:
            continue
        m = notes[ARP_PATTERN[h % 8]] + octv
        pl = pluck(m)
        t = tb(bar, h / 4)
        pan = -0.5 if h % 2 else 0.5
        music.add(pl, t, g, pan=pan)
        # пинг-понг дилей на 3/16
        music.add(pl, t + 3 * BEAT / 4, g * 0.35, pan=-pan)
        music.add(pl, t + 6 * BEAT / 4, g * 0.15, pan=pan)
        rev_send.add(pl, t, g * 0.3)

# --- мотив пианино (интро, провал, аутро) и лид во 2-м дропе / кульминации
MOTIF = [  # (доля от начала 4-тактовой фразы, midi, длительность в долях)
    (0, 62, 2), (2, 65, 2), (4, 64, 3), (7, 57, 1),
    (8, 62, 2), (10, 65, 1), (11, 67, 1), (12, 69, 3), (15, 61, 1),
]
for phrase_bar in (4, 40, 44, 86):
    for beat, m, dl in MOTIF:
        if tb(phrase_bar, beat) > TOTAL - 2:
            continue
        pn = piano(m + (0 if phrase_bar != 86 else -12), dl * BEAT + 1.5)
        g = 0.32 if phrase_bar in (40, 44) else 0.22
        music.add(pn, tb(phrase_bar, beat), g, pan=0.1)
        rev_send.add(pn, tb(phrase_bar, beat), g * 0.9)

for phrase_bar in list(range(60, 72, 4)) + list(range(74, 86, 4)):
    for beat, m, dl in MOTIF:
        n = int((dl * BEAT + 0.3) * SR)
        lead = saw_add(hz(m + 12), n / SR, harm=7, bright=2.5) * env_adsr(n, 0.03, 0.2, 0.7, 0.3)
        lead += saw_add(hz(m + 12.08), n / SR, harm=7, bright=2.5) * env_adsr(n, 0.03, 0.2, 0.7, 0.3)
        g = 0.12 if phrase_bar < 72 else 0.15
        music.add(lead, tb(phrase_bar, beat), g, pan=-0.15)
        rev_send.add(lead, tb(phrase_bar, beat), g * 0.8)

# ---------------------------------------------------------------- микс

# сайдчейн «пампинг» от кика на бас и музыку
duck = np.ones(N)
dn = int(0.32 * SR)
shape = 1 - 0.6 * np.exp(-np.arange(dn) / SR * 12)
for t in kick_times:
    i = int(t * SR)
    j = min(N, i + dn)
    duck[i:j] = np.minimum(duck[i:j], shape[: j - i])
bass.x *= duck[:, None]
music.x *= (0.35 + 0.65 * duck)[:, None]


def reverb(x, seconds=3.2):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    for ch in range(2):
        nz = rng.standard_normal(n)
        nz = smooth(nz, 3)
        ir[:, ch] = nz * np.exp(-t * 2.2)
    ir[: int(0.02 * SR)] = 0  # pre-delay
    ir /= np.sqrt(np.sum(ir ** 2, axis=0))
    nfft = 1 << int(np.ceil(np.log2(N + n)))
    out = np.zeros_like(x)
    for ch in range(2):
        y = np.fft.irfft(np.fft.rfft(x[:, ch], nfft) * np.fft.rfft(ir[:, ch], nfft), nfft)
        out[:, ch] = y[:N]
    return out


wet = reverb(rev_send.x)

mix = drums.x * 0.9 + bass.x * 0.8 + music.x * 1.0 + fx.x * 0.8 + wet * 0.55

# общая огибающая: плавный вход, «провал» в стопе, финальное затухание
master = np.interp(
    np.arange(N) / SR,
    [0, 1.5, tb(72) - 0.05, tb(72) + 0.05, tb(74) - 0.05, tb(74), TOTAL - 6, TOTAL],
    [0.0, 1.0, 1.0, 0.85, 0.85, 1.0, 1.0, 0.0],
)
mix *= master[:, None]

# мастер: мягкий лимитер
mix /= np.max(np.abs(mix)) / 1.6
mix = np.tanh(mix)
mix *= 0.891 / np.max(np.abs(mix))  # -1 dBFS

out_path = sys.argv[1] if len(sys.argv) > 1 else "intrigue_track.wav"
pcm = (mix * 32767).astype(np.int16)
with wave.open(out_path, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"Готово: {out_path}  ({TOTAL:.0f} c)")
