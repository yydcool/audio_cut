"""audio_cut: template-based repetitive segment detection and lossless removal.

Pipeline overview:
    audio_io  -> decode input to mono float32 chunks (ffmpeg pipe, no temp files)
    features  -> streaming log-mel spectrogram extraction + global normalisation
    matcher   -> normalised cross-correlation of a template against the full file
    segments  -> peak selection, padding, merge, keep/drop computation, guards
    cutter    -> per-segment `ffmpeg -c copy` cuts + concat demuxer (no re-encode)
    report    -> matches.json + scores.csv for auditing and tuning

Extension points:
    * new feature type  -> subclass FrameFeatureExtractor, register with
      @register_extractor("name") in features.py
    * new matcher       -> subclass Matcher, register with
      @register_matcher("name") in matcher.py
    Both are selectable from the CLI via --feature / --matcher without
    touching any other module.
"""

__version__ = "0.1.0"
