from collections import defaultdict
import ipdb
import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    base_name = "livingspeech"
    metadata_file = Path(in_dir) / 'text.xml'
    metadata = ET.parse(metadata_file).getroot()

    speaker = "livingaudio"
    total = len(metadata.getchildren())
    for child in tqdm(metadata.getchildren(), total=total):
        base_name = child.get('id')
        text = child.text
        text = _clean_text(text, cleaners)
        
        wav_path = Path(in_dir) / 'en.rp.rbu.48000' / '48000_orig' / 'rbu_{}.wav'.format(base_name)
        if wav_path.exists():
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)

