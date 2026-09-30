import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.io import wavfile

st.set_page_config(page_title='Voice Fourier Analyzer', page_icon='🎙️', layout='wide')

st.markdown('''
<style>
.main-title{font-size:2.5rem;font-weight:800;color:#132238;margin-bottom:0}
.subtitle{font-size:1.05rem;color:#607286;margin-bottom:1rem}
.hero{padding:1.5rem;border-radius:20px;background:linear-gradient(135deg,#eaf3ff,#f8fbff);border:1px solid #d8e7f5;margin-bottom:1.2rem}
.card{padding:1rem;border:1px solid #e3e9f1;border-radius:15px;background:white}
</style>
''', unsafe_allow_html=True)

st.markdown('''<div class="hero">
<div class="main-title">🎙️ Voice Signal Frequency Spectrum</div>
<div class="subtitle">Harmonic Extractor using Fourier Series</div>
Analyze a periodic signal, calculate Fourier coefficients, visualize its harmonic spectrum, and reconstruct it using selected harmonics.
</div>''', unsafe_allow_html=True)


def trapz_integral(x, y):
    """Compatibility-safe trapezoidal numerical integration."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def normalize(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    m = np.max(np.abs(x)) if x.size else 0
    return x / m if m > 0 else x


def make_synthetic(f0, duration, fs, harmonics, noise):
    t = np.arange(int(duration * fs)) / fs
    x = np.zeros_like(t)
    for n in range(1, harmonics + 1):
        x += np.sin(2 * np.pi * n * f0 * t) / n
    if noise:
        rng = np.random.default_rng(42)
        x += rng.normal(0, noise, len(x))
    return t, normalize(x)


def load_wav(file):
    fs, data = wavfile.read(io.BytesIO(file.read()))
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.astype(float).mean(axis=1)
    else:
        data = data.astype(float)
    if len(data) == 0:
        raise ValueError('The WAV file contains no samples.')
    return np.arange(len(data)) / fs, normalize(data), float(fs)


def estimate_f0(x, fs):
    if len(x) < 16:
        return max(1.0, 1.0 / max(len(x) / fs, 1e-6))
    y = x - np.mean(x)
    window = np.hanning(len(y))
    mag = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(len(y), 1 / fs)
    if len(mag) <= 2:
        return 1.0
    mag[:2] = 0
    idx = int(np.argmax(mag))
    return max(0.5, float(freqs[idx]))


def fourier_coefficients(t, x, f0, N):
    """Numerically calculate a0, an, bn without np.trapz/np.trapezoid."""
    T = 1.0 / f0
    w0 = 2 * np.pi * f0

    # Use one period for clean Fourier integration.
    end = t[0] + T
    mask = t <= end
    if np.count_nonzero(mask) >= 10:
        tt = t[mask]
        xx = x[mask]
    else:
        tt = np.linspace(t[0], t[-1], min(len(t), 4000))
        xx = np.interp(tt, t, x)

    a0 = (2 / T) * trapz_integral(tt, xx)
    an = np.zeros(N)
    bn = np.zeros(N)
    for n in range(1, N + 1):
        an[n - 1] = (2 / T) * trapz_integral(tt, xx * np.cos(n * w0 * tt))
        bn[n - 1] = (2 / T) * trapz_integral(tt, xx * np.sin(n * w0 * tt))
    return a0, an, bn


def reconstruct(t, f0, a0, an, bn, N):
    result = np.full_like(t, a0 / 2, dtype=float)
    w0 = 2 * np.pi * f0
    for n in range(1, N + 1):
        result += an[n - 1] * np.cos(n * w0 * t)
        result += bn[n - 1] * np.sin(n * w0 * t)
    return result

# Sidebar
st.sidebar.header('⚙️ Controls')
source = st.sidebar.radio('Signal source', ['Synthetic periodic signal', 'Upload WAV audio'])

if source == 'Synthetic periodic signal':
    f0 = st.sidebar.slider('Fundamental frequency f₀ (Hz)', 1.0, 50.0, 5.0, 0.5)
    duration = st.sidebar.slider('Duration (seconds)', 0.5, 4.0, 2.0, 0.5)
    fs = st.sidebar.select_slider('Sampling frequency (Hz)', [1000, 2000, 4000, 8000, 16000], value=4000)
    input_harmonics = st.sidebar.slider('Harmonics in generated signal', 1, 12, 5)
    noise = st.sidebar.slider('Noise level', 0.0, 0.15, 0.0, 0.01)
    t, signal = make_synthetic(f0, duration, fs, input_harmonics, noise)
    source_name = 'Synthetic periodic signal'
else:
    uploaded = st.sidebar.file_uploader('Upload a .wav file', type=['wav'])
    if uploaded is None:
        st.info('👈 Upload a WAV file from the sidebar, or choose the synthetic signal.')
        st.stop()
    try:
        t, signal, fs = load_wav(uploaded)
        f0 = estimate_f0(signal, fs)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f'Could not read the WAV file: {exc}')
        st.stop()

max_N = st.sidebar.slider('Maximum Fourier harmonics N', 1, 20, 20)
reconstruction_N = st.sidebar.slider('Reconstruction harmonics', 1, max_N, min(5, max_N))

# Limit plotting/calculation data for responsiveness.
if len(t) > 12000:
    ids = np.linspace(0, len(t) - 1, 12000).astype(int)
    ta, xa = t[ids], signal[ids]
else:
    ta, xa = t, signal

with st.spinner('Calculating Fourier coefficients...'):
    a0, an, bn = fourier_coefficients(ta, xa, f0, max_N)

n = np.arange(1, max_N + 1)
freq = n * f0
amp = np.sqrt(an**2 + bn**2)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric('Fundamental f₀', f'{f0:.2f} Hz')
c2.metric('Sampling rate', f'{fs:.0f} Hz')
c3.metric('Maximum N', str(max_N))
c4.metric('Current N', str(reconstruction_N))
st.caption(f'Source: **{source_name}**')

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(['📈 Signal', '📊 Coefficients', '🌈 Spectrum', '🔄 Reconstruction', '📚 Theory'])

with tab1:
    st.subheader('Input Signal')
    fig, ax = plt.subplots(figsize=(10, 4))
    step = max(1, len(t) // 5000)
    ax.plot(t[::step], signal[::step], linewidth=1.2)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude'); ax.set_title('Time-Domain Signal', fontweight='bold'); ax.grid(alpha=.2)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab2:
    st.subheader('Fourier Coefficients')
    rows = [{'Harmonic n': int(k), 'Frequency (Hz)': round(float(freq[i]), 3), 'aₙ': round(float(an[i]), 6), 'bₙ': round(float(bn[i]), 6), 'Amplitude': round(float(amp[i]), 6)} for i, k in enumerate(n)]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.info(f'a₀ = {a0:.6f}  |  Aₙ = √(aₙ² + bₙ²)')

with tab3:
    st.subheader('Harmonic Frequency Spectrum')
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.stem(freq, amp, basefmt=' ')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Amplitude'); ax.set_title('Fourier Harmonic Spectrum', fontweight='bold'); ax.grid(alpha=.2)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab4:
    st.subheader('Original vs Reconstructed Signal')
    rec = reconstruct(t, f0, a0, an, bn, reconstruction_N)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    step = max(1, len(t) // 5000)
    ax.plot(t[::step], signal[::step], label='Original', linewidth=1.2)
    ax.plot(t[::step], rec[::step], label=f'Reconstructed (N={reconstruction_N})', linewidth=1.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude'); ax.set_title('Fourier Reconstruction', fontweight='bold'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    audio = normalize(rec)
    audio_bytes = io.BytesIO()
    wavfile.write(audio_bytes, int(fs), np.int16(np.clip(audio, -1, 1) * 32767))
    st.markdown('### 🔊 Reconstructed Audio')
    st.audio(audio_bytes.getvalue(), format='audio/wav')

    st.markdown('### Effect of Number of Harmonics')
    vals = [v for v in [1, 5, 20] if v <= max_N]
    cols = st.columns(len(vals))
    for col, k in zip(cols, vals):
        with col:
            r = reconstruct(t, f0, a0, an, bn, k)
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            step = max(1, len(t) // 2500)
            ax.plot(t[::step], r[::step], linewidth=1.1)
            ax.set_title(f'N = {k}', fontweight='bold'); ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude'); ax.grid(alpha=.2)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab5:
    st.subheader('Mathematical Theory')
    st.markdown(r'''
### Full-Range Fourier Series

For a periodic signal:

\[
f(t)=\frac{a_0}{2}+\sum_{n=1}^{N}[a_n\cos(n\omega_0t)+b_n\sin(n\omega_0t)]
\]

where \(\omega_0=2\pi f_0\).

### Coefficients

\[
a_0=\frac{2}{T}\int f(t)dt
\]
\[
a_n=\frac{2}{T}\int f(t)\cos(n\omega_0t)dt
\]
\[
b_n=\frac{2}{T}\int f(t)\sin(n\omega_0t)dt
\]

### Harmonic amplitude

\[
A_n=\sqrt{a_n^2+b_n^2}
\]

### Project flow

**Signal → Fourier coefficients → Harmonic spectrum → Finite-harmonic reconstruction**

Increasing **N** includes more harmonic components and can give a more detailed approximation of the original periodic waveform.
''')

st.divider()
st.caption('Applied Mathematics Thinking-I • Voice Signal Frequency Spectrum & Harmonic Extractor')
