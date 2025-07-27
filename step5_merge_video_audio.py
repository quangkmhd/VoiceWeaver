import os
from moviepy.editor import VideoFileClip, AudioFileClip

INPUT_VIDEO_PATH = "path/to/your/video.mp4"
TRANSLATED_AUDIO_PATH = "translated_audio.wav"
FINAL_VIDEO_PATH = "final_video_dubbed.mp4"

def merge_audio_to_video(video_path, audio_path, output_path, original_volume=0.1):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'. Please update 'INPUT_VIDEO_PATH'.")
        return
        
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'. Please ensure Step 4 ran successfully.")
        return

    print("Starting to merge audio and video...")
    
    video_clip = VideoFileClip(video_path)
    if video_clip.audio:
        video_clip = video_clip.volumex(original_volume)

    audio_clip = AudioFileClip(audio_path)

    video_with_new_audio = video_clip.set_audio(audio_clip)

    video_with_new_audio.write_videofile(
        output_path, 
        codec='libx264', 
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a', 
        remove_temp=True
    )

    video_clip.close()
    audio_clip.close()
    video_with_new_audio.close()

    print(f"Successfully created dubbed video at: {output_path}")

if __name__ == "__main__":
    if INPUT_VIDEO_PATH == "path/to/your/video.mp4":
        print("Please open 'step5_merge_video_audio.py' and update the 'INPUT_VIDEO_PATH' variable.")
    else:
        merge_audio_to_video(INPUT_VIDEO_PATH, TRANSLATED_AUDIO_PATH, FINAL_VIDEO_PATH, original_volume=0.15)