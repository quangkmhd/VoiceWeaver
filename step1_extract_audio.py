import os
from moviepy.editor import VideoFileClip

def extract_audio(video_path, audio_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    print(f"Audio successfully extracted and saved at: {audio_path}")

if __name__ == "__main__":
    input_video_path = "path/to/your/video.mp4"
    output_audio_path = "extracted_audio.wav"
    extract_audio(input_video_path, output_audio_path)