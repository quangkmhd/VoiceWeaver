import json
import datetime

def format_srt_time(seconds):
    """Converts seconds to SRT time format HH:MM:SS,ms."""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def json_to_srt(json_path, srt_path):
    """Converts a JSON file with text chunks and timestamps to an SRT file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return

    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(data['chunks'], 1):
            start_time, end_time = chunk['timestamp']
            text = chunk.get('text_vi', '') # Use text_vi, default to empty string if not found

            f.write(str(i) + '\n')
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(text + '\n\n')

    print(f"Successfully converted {json_path} to {srt_path}")

if __name__ == '__main__':
    json_file = 'output_vi.json'
    srt_file = 'output_vi.srt'
    json_to_srt(json_file, srt_file)
