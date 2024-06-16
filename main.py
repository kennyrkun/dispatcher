import pyaudio
import wave
import audioop
import whisper
import struct
import datetime
import time
import os

from transformers import pipeline

# only so that whisper can download different models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

whisperModel = whisper.load_model("base")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
textGenerator = pipeline("text-generation", model="gpt2")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Sound level threshold for starting/stopping recording
THRESHOLD = 500  # Adjust as needed

# Duration limits in seconds
MIN_DURATION = 1
PAD_DURATION = 2.5 # if we are already recording, wait this long before we stop
MAX_DURATION = 30

stream = None

# Start the audio stream
def startStream():
    global stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

def speak(text, filename):
    return os.system(f"gtts-cli \"1, 2, ${text}\" --lang en --tld co.uk --output \"tx-{filename}\"")

frames = []
recording = False
start_time = None
last_detection_time = 0

# Save file path
save_path = os.path.join(os.path.expanduser('~'), 'Documents/GitHub/dispatcher')

startStream()

lastUnit = ""

availablePhrases = [
    "show me 10-8",
    "show me 108",
    "show me available",
    "show me tonight",
    "show me available",
    "show me in service",

    "show us 10-8",
    "show us 108",
    "show us available",
    "show us tonight",
    "show us available",
    "show us in service",

    "we're 10-8",
    "we're 108",
    "we're available",
    "we're tonight",
    "we're available",
    "we're in service",
]

unavailablePhrases = [
    "show me 10-7",
    "show me 107",
    "show me unavailable",
    "show me out of service",

    "show us 10-7",
    "show us 107",
    "show us unavailable",
    "show us out of service",

    "we're 10-7",
    "we're 107",
    "we're unavailable",
    "we're out of service",
]

while True:
    # we have to restart the stream after transcribing because whisper fucks it for some reason
    try:
        data = stream.read(1024)
    except:
        print("An exception occurred, restarting stream.")

        startStream()

        continue

    audio_data = struct.unpack(str(2 * 1024) + 'B', data)
    
    current_time = time.time()
    
    # TODO: find a way to replace rms because audioop is going away next python version
    level = audioop.rms(data, 2)

    print(level, end="\r")

    # sound is loud enough to record, or the pad duration has not elapsed
    if level > THRESHOLD or last_detection_time + PAD_DURATION > current_time:
        if not recording:
            print("Sound detected at level ", level, " starting recording...")
            start_time = current_time  # Initialize the start time

        recording = True
        frames.append(data)

        if level > THRESHOLD:
            last_detection_time = current_time
        
        # Check for max duration
        if start_time and current_time - start_time >= MAX_DURATION:
            print("Max duration reached, terminating.")
            recording = False

    # sound is too quiet
    else:
        if recording:
            duration = current_time - start_time

            if last_detection_time + PAD_DURATION < current_time:
                if duration >= MIN_DURATION:
                    print(f"Sound stopped, saving audio... Duration: {duration: .2f} seconds")
                    # Save the audio
                    filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
                    wf = wave.open(f"{save_path}/rx-{filename}", 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(44100)
                    wf.writeframes(b''.join(frames))
                    wf.close()
            
                    # fp16 is false because it generates a warning (at least on macos)
                    transcription = whisperModel.transcribe(f"rx-{filename}", fp16=False)
                    transcription = transcription['text'].strip(" .,\n").lower()

                    print(f'transcription: \"{transcription}\"')

                    if len(transcription) > 0:
                        #if transcription.endswith("control"):
                        #    lastUnit = transcription[0:transcription.find("control")]
                        #    speak("control, go ahead", filename)
                        #elif (len(lastUnit) > 0):
                        #    if transcription in availablePhrases:
                        #        speak("control is clear {$lastUnit}, you're in service", filename)
                        #    elif transcription in availablePhrases:
                        #        speak("control is clear {$lastUnit}, you're out of service", filename)
                        #    else: # repeat back what they said
                        #        speak(lastUnit + " unable to copy", filename)
                        #else: # repeat back what they said
                        #    speak(transcription, filename)

                        generatedResponse = textGenerator(transcription)

                        speak(generatedResponse[0]['generated_text'], filename)

                        # play the generated speech file
                        os.system(f"afplay -r 1.3 \"tx-{filename}\"")

                        # delete the generated speech file
                        os.remove(f"tx-{filename}")
                    else:
                        # rename the received audio to failed so we know
                        os.rename("rx-" + filename, "failed-rx-" + filename)
                else:
                    print(f"Sound stopped, discarding audio... Duration: {duration: .2f} seconds")
                
            # Reset recording states
            recording = False
            frames = []
            start_time = None