import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Fourier Series Theory & Signal Analysis", layout="wide")

# Modern, Professional UI Styling
st.markdown("""
<style>
    :root {
        --primary: #0F52BA;
        --bg-subtle: #F8FAFC;
        --border-color: #E2E8F0;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    .concept-card {
        background-color: var(--bg-subtle);
        border-left: 4px solid var(--primary);
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .concept-title {
        color: #0F172A;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.4rem;
    }
    .concept-desc {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.title("Fourier Series Mathematical Theory & Signal Analysis")

# Core Concept Highlight Card
st.markdown("""
<div class="concept-card">
    <div class="concept-title">💡 Core Mathematical Concept</div>
    <div class="concept-desc">
        <b>Fourier Series</b> decomposes any periodic signal into an infinite weighted sum of simple sinusoidal waves. 
        By calculating orthogonal projections over time intervals, complex speech and acoustic waveforms are reduced to discrete frequency components ($a_0, a_n, b_n$).
    </div>
</div>
""", unsafe_allow_html=True)

# App Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📚 Fourier Theory & Formulas", "📈 Signal Visualization", "⚙️ Concept Summary"])

with tab1:
    st.header("1. Full-Range vs. Half-Range Fourier Series")
    
    theory_full, theory_half = st.tabs(["Full-Range Fourier Series", "Half-Range Fourier Series"])
    
    with theory_full:
        st.subheader("Full-Range Fourier Series")
        st.write("Applicable to continuous signals defined over a complete periodic interval $[-L, L]$ or period $T = 2L$:")
        
        st.latex(r"f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{n \pi t}{L}\right) + b_n \sin\left(\frac{n \pi t}{L}\right) \right]")
        
        st.markdown("### Euler-Fourier Coefficients")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**DC Offset ($a_0$):**")
            st.latex(r"a_0 = \frac{1}{L} \int_{-L}^{L} f(t) \, dt")
        with c2:
            st.markdown("**Cosine Terms ($a_n$):**")
            st.latex(r"a_n = \frac{1}{L} \int_{-L}^{L} f(t) \cos\left(\frac{n \pi t}{L}\right) dt")
        with c3:
            st.markdown("**Sine Terms ($b_n$):**")
            st.latex(r"b_n = \frac{1}{L} \int_{-L}^{L} f(t) \sin\left(\frac{n \pi t}{L}\right) dt")

        st.markdown("---")
        st.markdown("### Harmonic Amplitude and Phase Spectrum")
        st.latex(r"A_n = \sqrt{a_n^2 + b_n^2}, \quad \phi_n = \arctan\left(\frac{-b_n}{a_n}\right)")

    with theory_half:
        st.subheader("Half-Range Fourier Expansion")
        st.write("Used when a signal is specified only over half a interval $[0, L]$. The function is mathematically extended to $[-L, L]$ using symmetry.")
        
        even_tab, odd_tab = st.tabs(["Half-Range Cosine (Even Symmetry)", "Half-Range Sine (Odd Symmetry)"])
        
        with even_tab:
            st.markdown("**Concept:** Assumes $f(-t) = f(t)$. All sine components drop out ($b_n = 0$).")
            st.latex(r"f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos\left(\frac{n \pi t}{L}\right)")
            st.latex(r"a_0 = \frac{2}{L} \int_{0}^{L} f(t) \, dt, \quad a_n = \frac{2}{L} \int_{0}^{L} f(t) \cos\left(\frac{n \pi t}{L}\right) dt")
            
        with odd_tab:
            st.markdown("**Concept:** Assumes $f(-t) = -f(t)$. All cosine components drop out ($a_0 = 0, a_n = 0$).")
            st.latex(r"f(t) = \sum_{n=1}^{\infty} b_n \sin\left(\frac{n \pi t}{L}\right)")
            st.latex(r"b_n = \frac{2}{L} \int_{0}^{L} f(t) \sin\left(\frac{n \pi t}{L}\right) dt")

with tab2:
    st.header("Interactive Signal Approximation")
    harmonics = st.slider("Select Number of Harmonics (N)", 1, 50, 5)
    
    t = np.linspace(-np.pi, np.pi, 1000)
    # Square wave approximation
    f_approx = np.zeros_like(t)
    for n in range(1, harmonics + 1, 2):
        f_approx += (4 / (n * np.pi)) * np.sin(n * t)
        
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, f_approx, color='#0F52BA', linewidth=2, label=f'N = {harmonics} Harmonics')
    ax.set_title("Square Wave Reconstruction via Fourier Sine Series")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper right")
    
    st.pyplot(fig)

with tab3:
    st.header("Pipeline Architecture")
    st.markdown("""
    1. **Input Signal Acquisition**: Raw acoustic / speech time-series signal ($f(t)$).
    2. **Orthogonal Projection**: Numerical integration across periodic limits to extract $a_0, a_n, b_n$.
    3. **Harmonic Synthesis**: Summing finite sinusoids to reconstruct audio signals without high-frequency noise.
    """)
