# Voice Translate - Vietnamese Video Translation Pipeline

**Voice Translate** is a complete automated pipeline for translating videos from foreign languages to Vietnamese. This project includes 5 main steps: audio extraction, speech recognition, text translation, Vietnamese speech synthesis, and final video merging.

> **Note**: This project was developed based on reference from the [F5-TTS-Vietnamese](https://github.com/nguyenthienhy/F5-TTS-Vietnamese.git) repository for Vietnamese speech synthesis.

## 🏗️ Pipeline Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Extract Audio  │───▶│ Audio Transcribe│
│   (.mp4/.avi)   │    │  (step1_...)    │    │  (step2_...)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Final Video    │◀───│  Merge Video    │◀───│  Translate Text │
│   with Voice    │    │   & Audio       │    │   to Vietnamese │
│   (.mp4)        │    │  (step5_...)    │    │  (step3_...)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲                       │
                                │              ┌─────────────────┐
                                │              │ Text-to-Speech  │
                                └──────────────│ Vietnamese TTS  │
                                               │  (step4_...)    │
                                               └─────────────────┘
                              
```

## ✨ Features

- 🎬 **Audio Extraction**: Extract audio from input video
- 🎤 **Speech Recognition**: Use OpenAI Whisper for speech-to-text conversion
- 🌐 **Automatic Translation**: Translate text to Vietnamese
- 🗣️ **Speech Synthesis**: Generate natural Vietnamese voice using F5-TTS
- 🎞️ **Video Merging**: Combine original video with new Vietnamese audio
- 📝 **Subtitle Export**: Create SRT files from translation results

## 📦 Installation

### Step 1: Clone this repository

```bash
git clone https://github.com/your-username/voice_tranlate.git
cd voice_tranlate
```

### Step 2: Clone F5-TTS Vietnamese (dependency)

```bash
git clone https://github.com/nguyenthienhy/F5-TTS-Vietnamese.git
```

### Step 3: Create virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Using individual steps:

#### Step 1: Extract audio from video

```bash
python step1_extract_audio.py
```

*Edit the video path in the file before running*

#### Step 2: Speech recognition

```bash
python step2_transcribe_audio.py
```

#### Step 3: Translate to Vietnamese

```bash
python step3_translate_to_vi.py
```

#### Step 4: Vietnamese speech synthesis

```bash
python step4_text_to_speech.py
```

#### Step 5: Merge video and audio

```bash
python step5_merge_video_audio.py
```

## 🛠️ Built With

- **[Python](https://www.python.org/)** - Main programming language
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Speech-to-text
- **[F5-TTS](https://github.com/SWivid/F5-TTS)** - Text-to-speech
- **[MoviePy](https://zulko.github.io/moviepy/)** - Video processing
- **[Transformers](https://huggingface.co/transformers/)** - NLP models

## 🙏 References and Acknowledgments

- **[F5-TTS-Vietnamese](https://github.com/nguyenthienhy/F5-TTS-Vietnamese.git)** - High-quality Vietnamese TTS model
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Powerful speech recognition tool
- **[Hugging Face](https://huggingface.co/)** - AI and NLP models platform

---

⭐ **If this project is helpful, please give it a star!** ⭐
