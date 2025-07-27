import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

INPUT_JSON_PATH = "output.json"
OUTPUT_JSON_PATH = "output_vi.json"
MODEL_NAME = "vinai/vinai-translate-en2vi-v2"

def translate_en_to_vi(input_path, output_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    with open(input_path, "r", encoding="utf-8") as f:
        whisper_data = json.load(f)

    chunks = whisper_data.get("chunks")
    if not chunks:
        print(f"Warning: No 'chunks' found in '{input_path}'. Nothing to translate.")
        return

    translated_chunks = []
    for chunk in tqdm(chunks, desc="Translating", unit="chunk"):
        english_text = chunk["text"].strip()

        if not english_text:
            translated_chunks.append({
                "timestamp": chunk["timestamp"],
                "text_en": "",
                "text_vi": ""
            })
            continue

        input_ids = tokenizer(english_text, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            top_k=100,
            top_p=0.95,
            decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
        )
        vietnamese_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        translated_chunks.append({
            "timestamp": chunk["timestamp"],
            "text_en": english_text,
            "text_vi": vietnamese_text
        })

    full_translated_text = " ".join([c["text_vi"] for c in translated_chunks])
    final_output = {
        "text_full_vi": full_translated_text,
        "chunks": translated_chunks
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(f"Translation complete! Results saved to '{output_path}'.")

if __name__ == "__main__":
    translate_en_to_vi(INPUT_JSON_PATH, OUTPUT_JSON_PATH)