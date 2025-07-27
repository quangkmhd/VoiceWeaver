import os
from moviepy.editor import VideoFileClip, AudioFileClip

# --- CẤU HÌNH ---
# Vui lòng cập nhật đường dẫn đến tệp VIDEO GỐC của bạn ở đây.
# Đây phải là cùng một video bạn đã sử dụng ở Bước 1.
# Ví dụ trên Windows: "C:\\Users\\Admin\\Videos\\my_video.mp4"
# Ví dụ trên Linux/Mac: "/home/user/videos/my_video.mp4"
INPUT_VIDEO_PATH = "SaveDouyin.com_Douyin_Media_001_be1af1a488ff2b0a982e4cac6cff9fcc.mp4"

# Đường dẫn đến tệp âm thanh lồng tiếng đã tạo ở Bước 4
TRANSLATED_AUDIO_PATH = "translated_audio.wav"

# Đường dẫn để lưu video cuối cùng đã được lồng tiếng
FINAL_VIDEO_PATH = "final_video_dubbed.mp4"
# -----------------

def merge_audio_to_video(video_path, audio_path, output_path, original_volume=0.1):
    """
    Thay thế âm thanh của một tệp video bằng một tệp âm thanh mới.
    """
    # 1. Kiểm tra sự tồn tại của các tệp đầu vào
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy tệp video tại '{video_path}'.")
        print("Vui lòng cập nhật biến 'INPUT_VIDEO_PATH' trong script.")
        return
        
    if not os.path.exists(audio_path):
        print(f"Lỗi: Không tìm thấy tệp âm thanh tại '{audio_path}'.")
        print("Vui lòng đảm bảo bạn đã chạy thành công Bước 4.")
        return

    print("Bắt đầu quá trình ghép âm thanh vào video...")
    
    # 2. Tải video và audio clips
    video_clip = None
    audio_clip = None
    try:
        print(f"Đang tải video từ: {video_path}")
        video_clip = VideoFileClip(video_path).volumex(original_volume)
        print(f"Đã giảm âm lượng video gốc xuống {original_volume * 100}%")
        
        print(f"Đang tải âm thanh từ: {audio_path}")
        audio_clip = AudioFileClip(audio_path)
    except Exception as e:
        print(f"Lỗi khi tải tệp: {e}")
        if video_clip: video_clip.close()
        if audio_clip: audio_clip.close()
        return

    # 3. Gán âm thanh mới cho video
    print("Đang thay thế track âm thanh của video...")
    video_with_new_audio = video_clip.set_audio(audio_clip)

    # 4. Ghi tệp video cuối cùng
    # codec="libx264" và audio_codec="aac" là các lựa chọn phổ biến, tương thích tốt.
    print(f"Đang ghi tệp video cuối cùng ra '{output_path}'...")
    print("Quá trình này có thể mất một lúc tùy thuộc vào độ dài và độ phân giải của video...")
    try:
        video_with_new_audio.write_videofile(
            output_path, 
            codec='libx264', 
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a', 
            remove_temp=True
        )
        print("✨ Hoàn tất! Video đã được lồng tiếng thành công. ✨")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi ghi tệp video: {e}")
    finally:
        # Giải phóng tài nguyên
        if video_clip: video_clip.close()
        if audio_clip: audio_clip.close()
        if 'video_with_new_audio' in locals() and video_with_new_audio:
            video_with_new_audio.close()


if __name__ == "__main__":
    
        # Bạn có thể thay đổi giá trị 0.1 thành một giá trị khác (từ 0.0 đến 1.0)
        # để điều chỉnh âm lượng của video gốc trong video cuối cùng.
        # Ví dụ: 0.2 là 20%, 0.5 là 50%, 0.0 là tắt tiếng hoàn toàn.
        merge_audio_to_video(INPUT_VIDEO_PATH, TRANSLATED_AUDIO_PATH, FINAL_VIDEO_PATH, original_volume=0.3)