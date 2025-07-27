import os
from moviepy.editor import VideoFileClip

def extract_audio(video_path, audio_path):
    """
    Trích xuất âm thanh từ tệp video và lưu dưới dạng tệp âm thanh.

    Args:
        video_path (str): Đường dẫn đến tệp video đầu vào.
        audio_path (str): Đường dẫn để lưu tệp âm thanh được trích xuất.
    """
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy tệp video tại '{video_path}'")
        return

    try:
        print(f"Bắt đầu trích xuất âm thanh từ '{video_path}'...")
        video = VideoFileClip(video_path)
        # Ghi tệp âm thanh với codec WAV lossless (pcm_s16le) để đảm bảo chất lượng tốt nhất
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        print(f"Đã trích xuất âm thanh thành công và lưu tại: {audio_path}")
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình trích xuất âm thanh: {e}")

if __name__ == "__main__":
    # --- THAY ĐỔI DƯỚI ĐÂY ---
    # Vui lòng thay thế "path/to/your/video.mp4" bằng đường dẫn thực tế đến tệp video của bạn.
    # Ví dụ trên Windows: "C:\\Users\\Admin\\Videos\\my_video.mp4"
    # Ví dụ trên Linux/Mac: "/home/user/videos/my_video.mp4"
    input_video_path = "path/to/your/video.mp4" 
    # --- KẾT THÚC THAY ĐỔI ---
    
    output_audio_path = "extracted_audio.wav"

    if input_video_path == "path/to/your/video.mp4":
        print("Vui lòng mở tệp 'step1_extract_audio.py' và cập nhật biến 'input_video_path' với đường dẫn đến tệp video của bạn.")
    else:
        extract_audio(input_video_path, output_audio_path) 