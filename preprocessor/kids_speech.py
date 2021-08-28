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
    base_name = "kids_speech"
    metadata = Path(in_dir) / 'trans' / 'reps.txt'
    data = metadata.read_text().split('\n')
    i_start = data.index('--- BEGIN DATA ---') + 1
    
    speaker_counter = defaultdict(int)
    #for i in range(i_start, len(data[i_start:], 3):
    for i in tqdm(range(i_start, len(data[i_start:]), 4)):
        row = data[i].split('  ')
        uttr_id = row[0]
        n_instsances = row[1]
        text = ' '.join(row[:2])
        # uttr_id, n_instsances, text = data[i].split('  ')
        text = _clean_text(text, cleaners)
        speaker = data[i+1].split(' ')[0]
        
        gender = speaker[0]
        assert speaker[-1] == '2', "{} is not two, in {}".format(speaker[-1], speaker)
        speaker_id = speaker[1:-1].replace(uttr_id, '')
        if config["preprocessing"]["speaker_id"] and config["preprocessing"]["speaker_id"] != speaker_id:
            continue

        speaker_counter[speaker_id] += 1
        wav_path = Path(in_dir) / 'wav' / '{}.wav'.format(speaker)

        #wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
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

    # speaker_counter = pd.DataFrame.from_dict(speaker_counter, index=range(len(speaker_counter))
    speaker_counter = pd.DataFrame(speaker_counter, index=[0]).T.sort_values(by=0, ascending=False)
    speaker_counter.to_csv(Path(out_dir) / "speaker_counts.csv")
    print("Written to ", out_dir)