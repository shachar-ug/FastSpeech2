from collections import defaultdict
import ipdb
import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    # base_name = "kids_speech"
    metadata = Path(in_dir) / 'trans' / 'reps.txt'
    data = metadata.read_text().split('\n')
    i_start = data.index('--- BEGIN DATA ---') + 1
    

    #for i in range(i_start, len(data[i_start:], 3):
    for i in tqdm(range(i_start, len(data[i_start:]), 4)):
        row = data[i].split('  ')
        uttr_id = row[0]
        n_instsances = row[1]
        text = ' '.join(row[:2])
        # uttr_id, n_instsances, text = data[i].split('  ')
        text = _clean_text(text, cleaners)
        base_name = data[i+1].split(' ')[0]
        
        gender = base_name[0]
        assert base_name[-1] == '2', "{} is not two, in {}".format(base_name[-1], base_name)
        speaker = base_name[1:-1].replace(uttr_id, '')
        # if config["preprocessing"]["speaker_id"] and config["preprocessing"]["speaker_id"] != speaker_id:
        #     continue

        wav_path = Path(in_dir) / 'wav' / '{}.wav'.format(base_name)
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