# Thirdy Party Credits
- Radio sounds are from http://www.w2sjw.com/.

# Requirements
- ollama and llama3.2
- ffmpeg
- ffplay
- openai whisper
- pyaudio
- gtts-cli (if not using piper)
- piper-tts (if not using gtts)
  - If using pip to install piper, follow these steps:
    - install piper-phonemize-cross
    - install piper-tts
    - install onnxruntime
  - voice data is required for piper. The program looks for voice data in `./voices/`
