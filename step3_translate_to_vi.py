import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# --- CẤU HÌNH ---
# Đường dẫn đến tệp JSON tiếng Trung từ Bước 2
INPUT_JSON_PATH = "/content/drive/MyDrive/AI_voice_translate/output_zh.json"

# Đường dẫn để lưu tệp JSON tiếng Việt kết quả
OUTPUT_JSON_PATH = "/content/drive/MyDrive/AI_voice_translate/output_vi.json"

# Tên model dịch thuật
# Đã cập nhật sang mô hình dịch Trung-Việt: Helsinki-NLP/opus-mt-zh-vi
MODEL_NAME = "Helsinki-NLP/opus-mt-zh-vi"
# -----------------

def translate_zh_to_vi(input_path, output_path):
    """
    Dịch văn bản trong tệp JSON từ tiếng Trung sang tiếng Việt bằng mô hình
    Helsinki-NLP/opus-mt-zh-vi, giữ lại dấu thời gian.
    """
    # 1. Kiểm tra và thiết lập thiết bị (GPU/CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"Phát hiện GPU, sử dụng: {torch.cuda.get_device_name(0)}")
    else:
        print("Không tìm thấy GPU, sử dụng CPU (có thể chậm hơn).")

    # 2. Tải Tokenizer và Model
    print(f"Đang tải mô hình '{MODEL_NAME}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        print("Tải mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Vui lòng kiểm tra lại kết nối mạng hoặc tên model.")
        return

    # 3. Đọc tệp JSON đầu vào
    print(f"Đang đọc tệp đầu vào: '{input_path}'")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            whisper_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp '{input_path}'.")
        print("Vui lòng đảm bảo bạn đã chạy thành công Bước 2 và tệp output_zh.json đã được tạo.")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: Tệp '{input_path}' không phải là một tệp JSON hợp lệ.")
        return

    # 4. Dịch từng đoạn (chunk)
    chunks = whisper_data.get("chunks")
    if not chunks:
        print(f"Cảnh báo: Không tìm thấy 'chunks' trong tệp '{input_path}'. Không có gì để dịch.")
        return

    translated_chunks = []
    total_chunks = len(chunks)
    print(f"Bắt đầu dịch {total_chunks} đoạn văn bản...")

    for i, chunk in enumerate(tqdm(chunks, desc="Đang dịch", unit="đoạn")):
        chinese_text = chunk["text"]

        # Đối với mô hình opus-mt-zh-vi, không cần tiền tố ngôn ngữ
        input_text = chinese_text.strip()

        if not input_text:
             translated_chunks.append({
                "timestamp": chunk["timestamp"],
                "text_zh": chinese_text.strip(),
                "text_vi": "" # Để trống nếu không có nội dung gốc
            })
             # print(f"  -> Bỏ qua đoạn {i + 1}/{total_chunks} vì không có nội dung.")
             continue


        # Tokenize và chuyển lên GPU (nếu có)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)


        # Dịch
        try:
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask, # Thêm attention mask
                max_length=512,
                num_beams=5, # Sử dụng beam search để kết quả tốt hơn
                early_stopping=True
            )
            vietnamese_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"\nLỗi khi dịch đoạn {i + 1}: {e}")
            vietnamese_text = "[LỖI DỊCH]" # Đánh dấu lỗi dịch


        # Lưu lại kết quả cùng timestamp
        translated_chunks.append({
            "timestamp": chunk["timestamp"],
            "text_zh": chinese_text.strip(),
            "text_vi": vietnamese_text
        })
        # print(f"  -> Đã dịch xong đoạn {i + 1}/{total_chunks}")

    # 5. Tạo và lưu tệp JSON kết quả
    full_translated_text = " ".join([c["text_vi"] for c in translated_chunks])
    final_output = {
        "text_full_vi": full_translated_text,
        "chunks": translated_chunks
    }

    print(f"Đang lưu kết quả vào tệp '{output_path}'...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        print("Hoàn tất! Đã lưu tệp dịch tiếng Việt thành công.")
    except Exception as e:
        print(f"Lỗi khi lưu tệp JSON: {e}")

if __name__ == "__main__":
    translate_zh_to_vi(INPUT_JSON_PATH, OUTPUT_JSON_PATH)