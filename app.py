import io
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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
    html_theory = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fourier Series Theory · Harmonic Analysis</title>
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
        displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
        processEscapes: true
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
  <style>
    :root {
      --primary: #0f4c75;
      --primary-dark: #0a3552;
      --primary-light: #1b6ca8;
      --primary-soft: #e8f1f8;
      --primary-muted: #a8c5d9;
      --text: #1a2a3a;
      --text-secondary: #4a5d6e;
      --border: #d0e0ec;
      --white: #ffffff;
      --radius: 10px;
      --shadow: 0 2px 12px rgba(15, 76, 117, 0.08);
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text);
      background: var(--white);
      line-height: 1.65;
      font-size: 16px;
    }

    /* Header */
    header {
      background: var(--primary);
      color: var(--white);
      padding: 1.25rem 0;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }

    .header-inner {
      max-width: 920px;
      margin: 0 auto;
      padding: 0 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .logo {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .logo span {
      opacity: 0.85;
      font-weight: 400;
      font-size: 0.9rem;
    }

    nav a {
      color: var(--white);
      text-decoration: none;
      font-size: 0.9rem;
      opacity: 0.9;
      margin-left: 1.5rem;
      transition: opacity 0.15s;
    }

    nav a:hover {
      opacity: 1;
      text-decoration: underline;
    }

    /* Main */
    main {
      max-width: 920px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem 4rem;
    }

    h1 {
      font-size: 1.9rem;
      font-weight: 800;
      color: var(--primary-dark);
      margin-bottom: 0.4rem;
      letter-spacing: -0.03em;
    }

    .subtitle {
      color: var(--text-secondary);
      font-size: 1.05rem;
      margin-bottom: 2.5rem;
    }

    h2 {
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--primary);
      margin: 2.4rem 0 1rem;
      padding-bottom: 0.45rem;
      border-bottom: 2px solid var(--primary-soft);
    }

    h3 {
      font-size: 1.1rem;
      font-weight: 650;
      color: var(--primary-dark);
      margin: 1.6rem 0 0.7rem;
    }

    p {
      margin-bottom: 1rem;
      color: var(--text);
    }

    /* Formula blocks */
    .formula-block {
      background: var(--primary-soft);
      border-left: 4px solid var(--primary);
      border-radius: 0 var(--radius) var(--radius) 0;
      padding: 1.1rem 1.4rem;
      margin: 1.1rem 0 1.4rem;
      overflow-x: auto;
    }

    .formula-block .MathJax {
      font-size: 1.05em !important;
    }

    .formula-label {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--primary);
      margin-bottom: 0.5rem;
    }

    /* Concept cards */
    .concept-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin: 1.2rem 0 1.6rem;
    }

    @media (max-width: 640px) {
      .concept-grid {
        grid-template-columns: 1fr;
      }
    }

    .concept-card {
      background: var(--white);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.1rem 1.25rem;
      box-shadow: var(--shadow);
    }

    .concept-card h4 {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 0.35rem;
    }

    .concept-card p {
      font-size: 0.9rem;
      color: var(--text-secondary);
      margin: 0;
    }

    /* Steps */
    .flow {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: center;
      margin: 1.2rem 0 1.8rem;
      font-size: 0.92rem;
    }

    .flow-step {
      background: var(--primary);
      color: white;
      padding: 0.4rem 0.9rem;
      border-radius: 6px;
      font-weight: 600;
    }

    .flow-arrow {
      color: var(--primary-muted);
      font-weight: 700;
    }

    /* Note box */
    .note {
      background: var(--white);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem 1.25rem;
      margin: 1.3rem 0;
      font-size: 0.93rem;
    }

    .note strong {
      color: var(--primary);
    }

    /* Table */
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 1.2rem 0 1.6rem;
      font-size: 0.93rem;
    }

    th, td {
      padding: 0.7rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }

    th {
      background: var(--primary-soft);
      color: var(--primary-dark);
      font-weight: 650;
    }

    td {
      color: var(--text);
    }

    /* Footer */
    footer {
      border-top: 1px solid var(--border);
      padding: 1.5rem;
      text-align: center;
      font-size: 0.85rem;
      color: var(--text-secondary);
      background: var(--primary-soft);
    }

    footer a {
      color: var(--primary);
      text-decoration: none;
    }

    footer a:hover {
      text-decoration: underline;
    }

    /* Utility */
    .mt-0 { margin-top: 0; }
    ul {
      margin: 0.6rem 0 1rem 1.4rem;
    }
    li {
      margin-bottom: 0.35rem;
      color: var(--text);
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div class="logo">
        Fourier Theory
        <span>· Harmonic Analysis</span>
      </div>
      <nav>
        <a href="#full-range">Full-Range</a>
        <a href="#half-range">Half-Range</a>
        <a href="#coefficients">Coefficients</a>
        <a href="#concepts">Concepts</a>
      </nav>
    </div>
  </header>

  <main>
    <h1>Mathematical Theory of Fourier Series</h1>
    <p class="subtitle">
      Complete reference for full-range and half-range Fourier series, coefficient formulas, 
      and the core concepts used in harmonic analysis of periodic signals.
    </p>

    <!-- Project Flow -->
    <h2 id="flow">Project Flow</h2>
    <div class="flow">
      <span class="flow-step">Signal</span>
      <span class="flow-arrow">→</span>
      <span class="flow-step">Fourier Coefficients</span>
      <span class="flow-arrow">→</span>
      <span class="flow-step">Harmonic Spectrum</span>
      <span class="flow-arrow">→</span>
      <span class="flow-step">Finite Reconstruction</span>
    </div>
    <p>
      Increasing the number of harmonics \(N\) includes higher-frequency components and yields 
      a progressively more accurate approximation of the original periodic waveform.
    </p>

    <!-- Full-Range -->
    <h2 id="full-range">1. Full-Range Fourier Series</h2>
    <p>
      Any periodic function \(f(t)\) with period \(T\) (fundamental frequency \(f_0 = 1/T\)) 
      can be expressed as an infinite sum of harmonically related sines and cosines.
    </p>

    <div class="formula-block">
      <div class="formula-label">Trigonometric Form</div>
      \[
        f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \Bigl[ a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t) \Bigr]
      \]
      where \(\omega_0 = 2\pi f_0 = \dfrac{2\pi}{T}\).
    </div>

    <p>
      In practical applications we truncate the sum at a finite order \(N\):
    </p>

    <div class="formula-block">
      <div class="formula-label">Finite Approximation</div>
      \[
        f_N(t) = \frac{a_0}{2} + \sum_{n=1}^{N} \Bigl[ a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t) \Bigr]
      \]
    </div>

    <h3>Full-Range Coefficients</h3>
    <p>
      The coefficients are obtained by orthogonality of the trigonometric basis over one period:
    </p>

    <div class="formula-block">
      <div class="formula-label">DC / Average Term</div>
      \[
        a_0 = \frac{2}{T} \int_{0}^{T} f(t)\, dt
      \]
      (sometimes written over any interval of length \(T\))
    </div>

    <div class="formula-block">
      <div class="formula-label">Cosine Coefficients</div>
      \[
        a_n = \frac{2}{T} \int_{0}^{T} f(t)\cos(n\omega_0 t)\, dt \qquad n = 1,2,3,\dots
      \]
    </div>

    <div class="formula-block">
      <div class="formula-label">Sine Coefficients</div>
      \[
        b_n = \frac{2}{T} \int_{0}^{T} f(t)\sin(n\omega_0 t)\, dt \qquad n = 1,2,3,\dots
      \]
    </div>

    <h3>Harmonic Amplitude & Phase</h3>
    <p>
      Each harmonic can be rewritten in amplitude-phase form:
    </p>

    <div class="formula-block">
      <div class="formula-label">Amplitude & Phase</div>
      \[
        A_n = \sqrt{a_n^2 + b_n^2}, \qquad
        \phi_n = \tan^{-1}\!\Bigl(\frac{b_n}{a_n}\Bigr)
      \]
      so that
      \[
        a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t) = A_n\cos(n\omega_0 t - \phi_n)
      \]
    </div>

    <!-- Half-Range -->
    <h2 id="half-range">2. Half-Range Fourier Series</h2>
    <p>
      When a function is defined only on the interval \([0, L]\), we can still expand it in a 
      Fourier series by inventing a suitable extension to a full period. Two canonical choices exist.
    </p>

    <h3>2.1 Half-Range Cosine Series (Even Extension)</h3>
    <p>
      Reflect the function evenly about \(t=0\) to obtain a period-\(2L\) even function. 
      Only cosine terms survive.
    </p>

    <div class="formula-block">
      <div class="formula-label">Half-Range Cosine Series</div>
      \[
        f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos\Bigl(\frac{n\pi t}{L}\Bigr)
      \]
    </div>

    <div class="formula-block">
      <div class="formula-label">Coefficients</div>
      \[
        a_0 = \frac{2}{L}\int_{0}^{L} f(t)\,dt, \qquad
        a_n = \frac{2}{L}\int_{0}^{L} f(t)\cos\Bigl(\frac{n\pi t}{L}\Bigr)\,dt
      \]
    </div>

    <h3>2.2 Half-Range Sine Series (Odd Extension)</h3>
    <p>
      Reflect the function oddly about \(t=0\) to obtain a period-\(2L\) odd function. 
      Only sine terms survive.
    </p>

    <div class="formula-block">
      <div class="formula-label">Half-Range Sine Series</div>
      \[
        f(t) = \sum_{n=1}^{\infty} b_n \sin\Bigl(\frac{n\pi t}{L}\Bigr)
      \]
    </div>

    <div class="formula-block">
      <div class="formula-label">Coefficients</div>
      \[
        b_n = \frac{2}{L}\int_{0}^{L} f(t)\sin\Bigl(\frac{n\pi t}{L}\Bigr)\,dt
      \]
    </div>

    <div class="note">
      <strong>Practical note:</strong> Half-range expansions are especially useful for solving 
      boundary-value problems (heat equation, wave equation) where the spatial domain is a finite 
      interval and boundary conditions dictate sine or cosine bases.
    </div>

    <!-- Comparison Table -->
    <h2 id="coefficients">3. Coefficient Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Series Type</th>
          <th>Interval</th>
          <th>Basis</th>
          <th>Key Formulas</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Full-Range</td>
          <td>\([0,T]\) or \([-T/2,T/2]\)</td>
          <td>\(1,\cos n\omega_0 t,\sin n\omega_0 t\)</td>
          <td>\(a_0,a_n,b_n\) as above</td>
        </tr>
        <tr>
          <td>Half-Range Cosine</td>
          <td>\([0,L]\)</td>
          <td>\(1,\cos(n\pi t/L)\)</td>
          <td>Even extension → only \(a_n\)</td>
        </tr>
        <tr>
          <td>Half-Range Sine</td>
          <td>\([0,L]\)</td>
          <td>\(\sin(n\pi t/L)\)</td>
          <td>Odd extension → only \(b_n\)</td>
        </tr>
      </tbody>
    </table>

    <!-- Concepts Used -->
    <h2 id="concepts">4. Core Concepts Used</h2>
    <div class="concept-grid">
      <div class="concept-card">
        <h4>Orthogonality</h4>
        <p>
          The set \(\{1,\cos n\omega_0 t,\sin n\omega_0 t\}\) is orthogonal on one period. 
          This property isolates each coefficient via a simple integral.
        </p>
      </div>
      <div class="concept-card">
        <h4>Periodicity & Fundamental Frequency</h4>
        <p>
          \(f(t+T)=f(t)\). The fundamental angular frequency \(\omega_0=2\pi/T\) determines 
          the spacing of all harmonics.
        </p>
      </div>
      <div class="concept-card">
        <h4>Harmonic Spectrum</h4>
        <p>
          Plotting \(A_n\) versus \(n f_0\) reveals the energy distribution across discrete 
          frequency components — the signature of a periodic signal.
        </p>
      </div>
      <div class="concept-card">
        <h4>Finite Reconstruction</h4>
        <p>
          Truncating at order \(N\) produces a band-limited approximation. Gibbs phenomenon 
          appears near discontinuities when \(N\) is finite.
        </p>
      </div>
      <div class="concept-card">
        <h4>Even / Odd Extensions</h4>
        <p>
          Choosing an even or odd periodic extension of a function defined on \([0,L]\) 
          automatically selects a pure cosine or pure sine series.
        </p>
      </div>
      <div class="concept-card">
        <h4>Parseval’s Relation</h4>
        <p>
          Average power of the signal equals the sum of the powers of its harmonics:
          \(\frac{1}{T}\int f^2 = \frac{a_0^2}{4} + \frac12\sum(a_n^2+b_n^2)\).
        </p>
      </div>
    </div>

    <!-- Complex Form (bonus, clean) -->
    <h2>5. Complex Exponential Form (Equivalent)</h2>
    <p>
      The same series can be written compactly using complex exponentials:
    </p>

    <div class="formula-block">
      <div class="formula-label">Complex Form</div>
      \[
        f(t) = \sum_{n=-\infty}^{\infty} c_n\, e^{j n \omega_0 t}
      \]
      with
      \[
        c_n = \frac{1}{T}\int_{0}^{T} f(t)\, e^{-j n \omega_0 t}\, dt
      \]
    </div>

    <p>
      The relationship between the two representations is:
    </p>
    <div class="formula-block">
      \[
        c_0 = \frac{a_0}{2},\qquad
        c_n = \frac{a_n - j b_n}{2},\qquad
        c_{-n} = \frac{a_n + j b_n}{2}
      \]
    </div>

    <div class="note">
      <strong>In this project:</strong> We compute the real coefficients \(a_n\) and \(b_n\) 
      numerically (via discrete sums approximating the integrals), form the harmonic amplitudes 
      \(A_n=\sqrt{a_n^2+b_n^2}\), display the spectrum, and reconstruct the signal with a 
      selectable number of harmonics.
    </div>
  </main>

  <footer>
    Applied Mathematics · Voice Signal Frequency Spectrum & Harmonic Extractor<br>
    Theory reference — Full-Range & Half-Range Fourier Series
  </footer>
</body>
</html>"""
    components.html(html_theory, height=1200, scrolling=True)

st.divider()
st.caption('Applied Mathematics Thinking-I • Voice Signal Frequency Spectrum & Harmonic Extractor')
