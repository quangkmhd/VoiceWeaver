import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import os
import sys
from omegaconf import OmegaConf


if F5_TTS_PROJECT_PATH is None:
    os.system("git clone https://github.com/SWivid/F5-TTS.git F5-TTS-Vietnamese")
    F5_TTS_PROJECT_PATH = "F5-TTS-Vietnamese"

F5_TTS_SRC_PATH = os.path.join(F5_TTS_PROJECT_PATH, 'src')
if F5_TTS_SRC_PATH not in sys.path:
    sys.path.append(F5_TTS_SRC_PATH)

from f5_tts.infer.utils_infer import (
    load_vocoder, load_model, infer_process, preprocess_ref_audio_text
)

def find_file(filename, search_dirs):
    for directory in search_dirs:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath
    return None

JSON_INPUT_FILE = "output_vi.json"
REF_AUDIO_FILE = "giong_noi.WAV"

REF_AUDIO_FILE = "giong_noi.wav"
OUTPUT_AUDIO_FILE = "translated_audio.wav"

MODEL_NAME = "F5TTS_Base"
VOCODER_NAME = "vocos"

CKPT_FILE = "model_last.pt"
VOCAB_FILE = "vocab.txt"

MODEL_CONFIG_FILE = os.path.join(F5_TTS_SRC_PATH, "f5_tts", "configs", f"{MODEL_NAME}.yaml")
REF_TEXT = "Xin chào đây là văn bản tham khảo giọng nói"

def initialize_tts_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=False)
    model_cfg = OmegaConf.load(MODEL_CONFIG_FILE).model
    model_cls = globals()[model_cfg.backbone]
    ema_model = load_model(
        model_cls, model_cfg.arch, CKPT_FILE, 
        mel_spec_type=VOCODER_NAME, vocab_file=VOCAB_FILE
    )
    ema_model.to(device)
    ema_model.eval()
    ref_audio_preprocessed, ref_text_preprocessed = preprocess_ref_audio_text(REF_AUDIO_FILE, REF_TEXT)

    try:
        sample_rate = vocoder.sample_rate
    except AttributeError:
        if VOCODER_NAME == "vocos":
            sample_rate = 24000
        elif VOCODER_NAME == "bigvgan":
            sample_rate = 22050
        else:
            sample_rate = 24000
    return ema_model, vocoder, ref_audio_preprocessed, ref_text_preprocessed, sample_rate, device

def process_and_synthesize(json_file, model, vocoder, ref_audio, ref_text, sample_rate, device, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    end_time_last_segment = data['chunks'][-1]['timestamp'][1]
    video_duration_ms = int(end_time_last_segment * 1000)
    final_audio_array = np.zeros(int(video_duration_ms / 1000 * sample_rate) + sample_rate, dtype=np.float32)

    sorted_chunks = sorted(data['chunks'], key=lambda x: x['timestamp'][0])

    for chunk in tqdm(sorted_chunks, desc="Synthesizing"):
        start_time, end_time = chunk['timestamp']
        text_to_generate = chunk.get('text_vi', '').strip()

        if not text_to_generate or text_to_generate in ["[âm nhạc]", "(âm nhạc)"]:
            continue

        segment_duration_s = end_time - start_time
        if segment_duration_s <= 0:
            continue

        target_samples = int(segment_duration_s * sample_rate)
        speed = 1.0
        audio_segment, _, _ = infer_process(
            ref_audio, ref_text, text_to_generate, model, vocoder, 
            mel_spec_type=VOCODER_NAME, speed=speed
        )

        current_duration_s = len(audio_segment) / sample_rate
        if current_duration_s > 0.05 and abs(current_duration_s - segment_duration_s) / segment_duration_s > 0.15:
            speed = max(0.5, min(2.0, current_duration_s / segment_duration_s))
            audio_segment, _, _ = infer_process(
                ref_audio, ref_text, text_to_generate, model, vocoder, 
                mel_spec_type=VOCODER_NAME, speed=speed
            )

        if len(audio_segment) > target_samples:
            audio_segment = audio_segment[:target_samples]
        else:
            padding = np.zeros(target_samples - len(audio_segment), dtype=np.float32)
            audio_segment = np.concatenate([audio_segment, padding])

        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(audio_segment)

        if end_sample > len(final_audio_array):
            end_sample = len(final_audio_array)
            audio_segment = audio_segment[:end_sample - start_sample]

        if start_sample < len(final_audio_array) and (end_sample - start_sample) > 0:
            final_audio_array[start_sample:end_sample] += audio_segment

    torchaudio.save(output_file, torch.from_numpy(final_audio_array).unsqueeze(0), sample_rate)

if __name__ == "__main__":
    model, vocoder, ref_audio, ref_text, sr, dev = initialize_tts_model()
    process_and_synthesize(
        JSON_INPUT_FILE, model, vocoder, ref_audio, ref_text, sr, dev, OUTPUT_AUDIO_FILE
    )
