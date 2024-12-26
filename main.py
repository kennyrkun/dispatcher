import pyaudio
import wave
import audioop
import whisper
import struct
import datetime
import time
import os
import requests

# only so that whisper can download different models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

whisperModel = whisper.load_model("base")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Sound level threshold for starting/stopping recording
THRESHOLD = 500  # Adjust as needed

# Duration limits in seconds
MIN_DURATION = 1
PAD_DURATION = 2.5 # if we are already recording, wait this long before we stop
MAX_DURATION = 30

# Save file path
save_path = os.path.join(os.path.expanduser('~'), 'Documents/GitHub/dispatcher')

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
    "show us ten eight",

    "we're 10-8",
    "we're 108",
    "we're available",
    "we're tonight",
    "we're available",
    "we're in service",
    "we're ten eight",
]

unavailablePhrases = [
    "show me 10-7",
    "show me 107",
    "show me unavailable",
    "show me out of service",
    "show me ten seven",

    "show us 10-7",
    "show us 107",
    "show us unavailable",
    "show us out of service",
    "show us ten seven",

    "we're 10-7",
    "we're 107",
    "we're unavailable",
    "we're out of service",
    "we're ten seven",
]

def promptResponse(string):
    request = requests.post("http://localhost:11434/api/generate", json = {
        "model": "llama3.2",
        "stream": False,
        "prompt": string
    })

    print("Raw response content: ", request.content)

    return request.json()

def speak(text, filename):
    global save_path

    return os.system(f"gtts-cli \"1, 2, ${text}\" --lang en --output \"{save_path}/tx-{filename}\"")

def processLoop():
    frames = []
    recording = False
    start_time = None
    last_detection_time = 0
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=1024
    )

    print("Started stream, beginning processing...")

    while True:
        data = stream.read(1024)

        current_time = time.time()
        
        # TODO: find a way to replace rms because audioop is going away next python version
        level = audioop.rms(data, 2)
        
        print(f"Audio level: {level}", end="\r")

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
                        print(f"Sound stopped with duration of {duration: .2f} seconds")
                        # Save the audio
                        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
                        wf = wave.open(f"{save_path}/rx-{filename}", 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(44100)
                        wf.writeframes(b''.join(frames))
                        wf.close()

                        if not os.path.isfile(f"{save_path}/rx-{filename}"):
                            raise Exception("File for received audio did not exist to transcribe.")
                
                        # fp16 is false because it generates a warning (at least on macos)
                        transcription = whisperModel.transcribe(f"{save_path}/rx-{filename}", fp16=False)
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

                            response = promptResponse(transcription)

                            if response.get("error") is not None:
                                raise Exception("Error response from model: " + response.get("error"))
                            elif response.get("response") is None:
                                raise Exception("Reponse from model was None.")
                            
                            response = response.get("response")

                            print(f"Response: {response}")

                            speak(response, filename)

                            # play the generated speech file
                            os.system(f"afplay -r 1.3 \"{save_path}/tx-{filename}\"")

                            # delete the generated speech file
                            os.remove(f"{save_path}/tx-{filename}")
                        else:
                            # rename the received audio to failed so we know
                            os.rename(f"{save_path}/rx-{filename}", f"failed-rx-{filename}")
                    else:
                        # TODO: last unit was unreadable
                        print(f"Sound stopped, discarding audio... Duration: {duration: .2f} seconds")
                    
                # Reset recording states
                recording = False
                frames = []
                start_time = None

while True:
    try:
        processLoop()
    except Exception as e:
        print("An exception occurred:", e)
        print("The stream will be restarted.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt quit")
        exit()