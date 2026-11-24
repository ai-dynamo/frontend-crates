# Generic audio references

All inputs are synthetic; no checkpoint, inference runtime, network or GPU is
needed to replay the Rust tests.

- `log_mel.json`: complete f32 output tensors and frame-anchor masks for four
  distinct configurations, captured by `generate.py` from transformers 5.12.1
  `audio_utils.spectrogram` on NumPy 1.26.4 CPU. Inputs are determined by integer
  arithmetic in the generator and replay test. These captures use exact f32-bit
  assertions, including padding/framing, Slaney/HTK and window policies. They
  are not a model-family/PyTorch parity table. The f64 FFT-output mode has
  separate geometry/finite-output tests; these captures use complex64 rounding.
- `stereo.wav` / `stereo.flac`: 4096 frames of stereo 22.05 kHz int16 PCM.
  Both decode exactly to the integer-defined source. The pipeline test compares
  them with an independent f32le PCM importer through downmix, 22.05→16 kHz
  resampling, padding and feature extraction.
- `stereo.mp3` / `stereo.ogg`: lossy encodings of that source. Their
  `.decoded.f32le` files are **independent libsndfile decoder** references, not
  the source waveform (encoding itself is lossy). Tests compare every decoded
  scalar with an explicit numerical budget of 2e-6, and check the exact gapless
  length/rate/channel order. This does not assert bit-exact lossy decoding.
- `vbr-no-duration.mp3`: four seconds of synthetic mono audio at 22.05 kHz
  encoded with variable bitrates, with the leading Xing metadata frame removed.
  Its 156 audio frames decode to 89,856 samples (cross-checked with CoreAudio).
  The test verifies all frames are returned and the decoded frame cap is enforced.
- `codecs.json`: encoder/decoder package versions and source formula, generated
  by `generate_codecs.py` using NumPy 1.26.4, soundfile 0.13.1, bundled libsndfile
  1.2.2. Package versions are asserted by the scripts; Ogg stream identifiers
  may vary on regeneration without changing decoded samples.

Rubato 0.16.2 FFT resampling is tested with exact output lengths, latency/phase,
passband gain and stopband rejection. It is a named backend, not a substitute
claimed to be bit-exact to Python's librosa/soxr or torchaudio resamplers.
