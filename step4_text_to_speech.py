import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import os
import sys
from omegaconf import OmegaConf
from importlib.resources import files

# --- SỬA LỖI MODULE NOT FOUND CHO KAGGLE ---
# 1. Đường dẫn cho môi trường Kaggle
current_dir = "/kaggle/working"  # Thư mục làm việc của Kaggle
input_dir = "/kaggle/input"      # Thư mục input của Kaggle

# Tìm đường dẫn F5-TTS trong input hoặc working
possible_f5_paths = [
    "/kaggle/input/f5-tts-vietnamese",           # Nếu upload như dataset
    "/kaggle/input/f5-tts",                     # Tên khác có thể
    "/kaggle/working/F5-TTS-Vietnamese",        # Nếu clone về working
    "/kaggle/working/F5-TTS",                   # Tên khác
    "/kaggle/input/f5-tts-vietnamese/F5-TTS-Vietnamese",  # Nested folder
]

F5_TTS_PROJECT_PATH = None
for path in possible_f5_paths:
    if os.path.exists(path):
        F5_TTS_PROJECT_PATH = path
        print(f"Đã tìm thấy F5-TTS tại: {path}")
        break

if F5_TTS_PROJECT_PATH is None:
    print("CẢNH BÁO: Không tìm thấy thư mục F5-TTS!")
    print("Các đường dẫn đã kiểm tra:")
    for path in possible_f5_paths:
        print(f"  - {path}")
    print("\nHãy đảm bảo bạn đã:")
    print("1. Upload F5-TTS như một dataset, hoặc")
    print("2. Clone F5-TTS vào /kaggle/working/")
    
    # Thử tự động clone nếu có git
    print("\nĐang thử clone F5-TTS...")
    try:
        os.system("cd /kaggle/working && git clone https://github.com/SWivid/F5-TTS.git F5-TTS-Vietnamese")
        F5_TTS_PROJECT_PATH = "/kaggle/working/F5-TTS-Vietnamese"
        print(f"Clone thành công! Sử dụng: {F5_TTS_PROJECT_PATH}")
    except:
        print("Không thể clone tự động. Vui lòng upload dataset hoặc clone thủ công.")
        sys.exit(1)

F5_TTS_SRC_PATH = os.path.join(F5_TTS_PROJECT_PATH, 'src')

if F5_TTS_SRC_PATH not in sys.path:
    sys.path.append(F5_TTS_SRC_PATH)

# 2. Import các hàm cần thiết từ thư viện f5_tts
try:
    from f5_tts.infer.utils_infer import (
        load_vocoder, load_model, infer_process, preprocess_ref_audio_text
    )
    from f5_tts.model import DiT, UNetT  # Cần thiết để load model
    print("Import F5-TTS thành công!")
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Không thể import các thành phần từ 'f5_tts'.")
    print(f"Đường dẫn src: {F5_TTS_SRC_PATH}")
    print("Hãy kiểm tra cấu trúc thư mục:")
    if os.path.exists(F5_TTS_PROJECT_PATH):
        print(f"Nội dung {F5_TTS_PROJECT_PATH}:")
        print(os.listdir(F5_TTS_PROJECT_PATH))
    sys.exit(1)

# --- CẤU HÌNH CHO KAGGLE ---
# Tìm file input
def find_file(filename, search_dirs):
    """Tìm file trong các thư mục"""
    for directory in search_dirs:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath
    return None

# Các thư mục để tìm file
search_directories = [
    "/kaggle/input",
    "/kaggle/working",
    current_dir,
]

# Đường dẫn cụ thể cho file của bạn
JSON_INPUT_FILE = "/kaggle/input/data-j/output_vi.json"
REF_AUDIO_FILE = "/kaggle/input/data-j/giong_noi.WAV"  # Thử cả hai định dạng
if not os.path.exists(REF_AUDIO_FILE):
    REF_AUDIO_FILE = "/kaggle/input/data-j/giong_noi.wav"  # Nếu file là .wav thường
OUTPUT_AUDIO_FILE = os.path.join(current_dir, "translated_audio.wav")

# Cấu hình model - tìm model checkpoint
MODEL_NAME = "F5TTS_Base"
VOCODER_NAME = "vocos"  # 'vocos' hoặc 'bigvgan'

# Đường dẫn cụ thể cho model và vocab
CKPT_FILE = "/kaggle/working/F5-TTS-Vietnamese/model_last.pt"
VOCAB_FILE = "/kaggle/input/vocab-txt/vocab.txt"

# Kiểm tra file tồn tại
if not os.path.exists(CKPT_FILE):
    print(f"CẢNH BÁO: Không tìm thấy model checkpoint tại {CKPT_FILE}")
    print("Model sẽ được tải từ Hugging Face (cần internet).")
    CKPT_FILE = None

if not os.path.exists(VOCAB_FILE):
    print(f"CẢNH BÁO: Không tìm thấy vocab file tại {VOCAB_FILE}")
    print("Sẽ sử dụng vocab mặc định.")
    VOCAB_FILE = None

# Config file
MODEL_CONFIG_FILE = os.path.join(F5_TTS_SRC_PATH, "f5_tts", "configs", f"{MODEL_NAME}.yaml")

# !!! QUAN TRỌNG: THAY THẾ BẰNG PHIÊN ÂM CHÍNH XÁC CỦA FILE giong_noi.WAV !!!
REF_TEXT = "Xin chào đây là văn bản tham khảo giọng nói"

# --- KIỂM TRA FILE ---
print("\n=== KIỂM TRA FILE ===")
files_to_check = [
    ("JSON Input", JSON_INPUT_FILE),
    ("Reference Audio", REF_AUDIO_FILE),
    ("Model Config", MODEL_CONFIG_FILE),
    ("Model Checkpoint", CKPT_FILE),
    ("Vocab File", VOCAB_FILE),
]

missing_files = []
for name, filepath in files_to_check:
    if filepath and os.path.exists(filepath):
        print(f"✓ {name}: {filepath}")
    else:
        print(f"✗ {name}: {filepath} (KHÔNG TỒN TẠI)")
        missing_files.append(name)

if missing_files:
    print(f"\nCÁC FILE SAU KHÔNG TỒN TẠI: {', '.join(missing_files)}")
    print("Hãy đảm bảo upload đúng file vào Kaggle dataset hoặc working directory.")
    
    # Liệt kê file có sẵn để debug
    print("\nFile trong /kaggle/input:")
    try:
        for item in os.listdir("/kaggle/input"):
            print(f"  - {item}")
            subpath = os.path.join("/kaggle/input", item)
            if os.path.isdir(subpath):
                print(f"    Nội dung {item}:")
                for subitem in os.listdir(subpath)[:10]:  # Chỉ hiện 10 file đầu
                    print(f"      • {subitem}")
    except:
        pass
    
    print("\nFile trong /kaggle/working:")
    try:
        for item in os.listdir("/kaggle/working"):
            print(f"  - {item}")
    except:
        pass

if REF_TEXT == "Xin chào đây là văn bản tham khảo giọng nói":
    print("\n⚠️  CẢNH BÁO: Bạn đang sử dụng văn bản tham chiếu mặc định.")
    print("   Vui lòng thay thế REF_TEXT bằng phiên âm chính xác của file giong_noi.WAV.")

# --- KHỞI TẠO MODEL ---
def initialize_tts_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nĐang sử dụng thiết bị: {device}")

    print("Đang tải Vocoder...")
    vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=False)

    print("Đang tải Model TTS...")
    model_cfg = OmegaConf.load(MODEL_CONFIG_FILE).model
    model_cls = globals()[model_cfg.backbone]
    
    ema_model = load_model(
        model_cls, model_cfg.arch, CKPT_FILE, 
        mel_spec_type=VOCODER_NAME, vocab_file=VOCAB_FILE  # Sử dụng vocab file cụ thể
    )
    ema_model.to(device)
    ema_model.eval()

    print("Xử lý audio tham chiếu...")
    ref_audio_preprocessed, ref_text_preprocessed = preprocess_ref_audio_text(REF_AUDIO_FILE, REF_TEXT)

    # Lấy sample_rate từ cấu hình vocoder hoặc sử dụng giá trị mặc định
    try:
        sample_rate = vocoder.sample_rate
    except AttributeError:
        # Nếu vocoder không có sample_rate, sử dụng giá trị mặc định dựa trên loại vocoder
        if VOCODER_NAME == "vocos":
            sample_rate = 24000  # Vocos thường dùng 24kHz
        elif VOCODER_NAME == "bigvgan":
            sample_rate = 22050  # BigVGAN thường dùng 22.05kHz
        else:
            sample_rate = 24000  # Mặc định
        print(f"Sử dụng sample rate mặc định: {sample_rate}Hz cho {VOCODER_NAME}")

    return ema_model, vocoder, ref_audio_preprocessed, ref_text_preprocessed, sample_rate, device

# --- HÀM TỔNG HỢP VÀ XỬ LÝ ---
def process_and_synthesize(json_file, model, vocoder, ref_audio, ref_text, sample_rate, device, output_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file JSON {json_file}: {e}")
        return

    # Sửa lỗi: Sử dụng key 'chunks' thay vì 'segments'
    if 'chunks' not in data or not data['chunks']:
        print("File JSON không có 'chunks' hoặc 'chunks' rỗng. Kết thúc.")
        return

    # Sửa lỗi: Lấy thời gian kết thúc từ chunk cuối cùng
    end_time_last_segment = data['chunks'][-1]['timestamp'][1]
    video_duration_ms = int(end_time_last_segment * 1000)
    final_audio_array = np.zeros(int(video_duration_ms / 1000 * sample_rate) + sample_rate, dtype=np.float32) # Thêm 1s buffer

    # Sửa lỗi: Sắp xếp các chunks dựa trên timestamp[0]
    sorted_chunks = sorted(data['chunks'], key=lambda x: x['timestamp'][0])

    for chunk in tqdm(sorted_chunks, desc="Đang tổng hợp giọng nói"):
        # Sửa lỗi: Lấy start_time và end_time từ 'timestamp'
        start_time, end_time = chunk['timestamp']
        # Sửa lỗi: Lấy text từ 'text_vi'
        text_to_generate = chunk.get('text_vi', '').strip()

        if not text_to_generate or text_to_generate in ["[âm nhạc]", "(âm nhạc)"]:
            continue

        segment_duration_s = end_time - start_time
        if segment_duration_s <= 0:
            continue
        target_samples = int(segment_duration_s * sample_rate)

        # Thử tổng hợp với tốc độ bình thường
        speed = 1.0
        try:
            audio_segment, _, _ = infer_process(
                ref_audio, ref_text, text_to_generate, model, vocoder, 
                mel_spec_type=VOCODER_NAME, speed=speed
            )
        except Exception as e:
            print(f"\nLỗi khi tổng hợp: '{text_to_generate[:50]}...'. Lỗi: {e}")
            continue

        # Điều chỉnh tốc độ nếu cần
        current_duration_s = len(audio_segment) / sample_rate
        # Chỉ điều chỉnh nếu chênh lệch đáng kể và thời lượng lớn hơn 0
        if current_duration_s > 0.05 and abs(current_duration_s - segment_duration_s) / segment_duration_s > 0.15:
            speed = max(0.5, min(2.0, current_duration_s / segment_duration_s))
            print(f"  [INFO] Điều chỉnh tốc độ thành: {speed:.2f} cho: '{text_to_generate[:30]}...' ")
            try:
                audio_segment, _, _ = infer_process(
                    ref_audio, ref_text, text_to_generate, model, vocoder, 
                    mel_spec_type=VOCODER_NAME, speed=speed
                )
            except Exception as e:
                print(f"\nLỗi khi tổng hợp lại với tốc độ mới: {e}")
                continue

        # Cắt hoặc đệm audio
        if len(audio_segment) > target_samples:
            audio_segment = audio_segment[:target_samples]
        else:
            padding = np.zeros(target_samples - len(audio_segment), dtype=np.float32)
            audio_segment = np.concatenate([audio_segment, padding])

        # Ghi vào mảng audio cuối cùng
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(audio_segment)
        
        # Đảm bảo không ghi vượt quá mảng
        if end_sample > len(final_audio_array):
            end_sample = len(final_audio_array)
            audio_segment = audio_segment[:end_sample - start_sample]
        
        if start_sample < len(final_audio_array) and (end_sample - start_sample) > 0:
            final_audio_array[start_sample:end_sample] += audio_segment

    print(f"Đang lưu file âm thanh vào: {output_file}")
    torchaudio.save(output_file, torch.from_numpy(final_audio_array).unsqueeze(0), sample_rate)
    print("Hoàn tất!")

if __name__ == "__main__":
    try:
        print("=== BẮT ĐẦU QUÁ TRÌNH TỔNG HỢP GIỌNG NÓI TRÊN KAGGLE ===")
        model, vocoder, ref_audio, ref_text, sr, dev = initialize_tts_model()
        process_and_synthesize(
            JSON_INPUT_FILE, model, vocoder, ref_audio, ref_text, sr, dev, OUTPUT_AUDIO_FILE
        )
    except Exception as e:
        print(f"Đã xảy ra lỗi nghiêm trọng trong quá trình thực thi: {e}")
        import traceback
        traceback.print_exc()