# Voice Fourier Analyzer

This is the updated Streamlit application for the maths mini project.

## Run in VS Code terminal

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the Local URL shown by Streamlit, usually `http://localhost:8501`.

## Fixed

The app no longer uses `np.trapz` or `np.trapezoid`. It contains its own compatibility-safe trapezoidal numerical integration, so the NumPy error from the previous version is removed.

## Features

- Synthetic periodic signal
- WAV upload
- Fourier coefficients a0, an and bn
- Harmonic frequency spectrum
- Adjustable reconstruction N
- N=1, N=5 and N=20 comparison
- Original vs reconstructed waveform
- Reconstructed audio playback
- Built-in Fourier theory section
- Improved Streamlit frontend
