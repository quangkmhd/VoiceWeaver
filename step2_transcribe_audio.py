import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset
import json


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

print("Đang tải model, vui lòng chờ...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# `load_dataset` không dùng để tải một file audio đơn lẻ như thế này.
# Thay vào đó, bạn có thể truyền thẳng đường dẫn file vào pipeline.
# dataset = load_dataset("0726.MP3", "clean", split="validation")
# sample = dataset[0]["audio"]

print("\nBắt đầu dịch file âm thanh. Quá trình này có thể mất vài phút...")
result = pipe("0726.MP3", return_timestamps=True, generate_kwargs={"task": "transcribe"})
print("Xử lý hoàn tất!")


print("\nNội dung dịch đầy đủ:")
print(result["text"])
print("\nChi tiết từng đoạn (chunks):")
print(result["chunks"])

output_filename = "output.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"\nĐã lưu kết quả vào file {output_filename}")
