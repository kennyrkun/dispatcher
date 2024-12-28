import argparse

parser = argparse.ArgumentParser(
    prog = "Dispatcher",
    description = "Uses AI to transcribe messages from an audio input, then generate a response based on that message and play it to an output."
)

parser.add_argument("-mdc",
    action = "store_true",
    help = "play an mdc squak after finished transmitting"
)

parser.add_argument("-saveSpokenAudio",
    action = "store_true",
    help = "do not delete audio files containing spoken responses"
)

parser.add_argument("-saveReceivedAudio",
    action = "store_false",
    help = "do not delete files containing received transmission audio"
)

parser.add_argument("-delayTone",
    type = float,
    default = 1.5,
    help = "how long to wait before generateSpokenResponseing while playing a tone, in seconds"
)

parser.add_argument("-delay",
    type = float,
    help = "how long to wait before generateSpokenResponseing, in seconds"
)

flags = parser.parse_args()

print(flags.__dict__)

import audioop
import datetime
import os
import pyaudio
import requests
import sys
import time
import wave
import whisper

# only so that whisper can download different models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

whisperModel = whisper.load_model("base")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Sound level threshold for starting/stopping recording
THRESHOLD = 800  # Adjust as needed

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
    print("Prompting response...")

    request = requests.post("http://localhost:11434/api/generate", json = {
        "model": "llama3.2",
        "stream": False,
        "prompt": string
    })

    print("Raw response content: ", request.content)

    return request.json()

def generateSpokenResponse(text, filename):
    global save_path

    print("Generating response audio...")

    return subprocess.check_call(
        [
            "gtts-cli",
            #"--debug",
            #"--lang", "en",
            "--output", f"{save_path}/tx-{filename}",
            text
        ],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT
    )

def ffplay(filename, args = ""):
    return os.system(f"ffplay {args} \"{filename}\" -autoexit -nodisp")

def playTone(freq = 1000, length = 5):
    os.system(f"ffmpeg -f lavfi -i 'sine=frequency={freq}:duration={length}' tone.wav")
    # TODO: use a generic audio player
    ffplay("tone.wav")
    os.remove("tone.wav")

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
    print("") # new line to prevent audio level from overwriting things

    while True:
        data = stream.read(1024)

        current_time = time.time()
        
        # TODO: find a way to replace rms because audioop is going away next python version
        level = audioop.rms(data, 2)
        
        sys.stdout.write("\033[F") # up one line
        print(f"Audio level: {level}")
        sys.stdout.write("\033[K") # clear to end of line

        # sound is loud enough to record, or the pad duration has not elapsed
        if level > THRESHOLD or last_detection_time + PAD_DURATION > current_time:
            if not recording:
                print("Sound detected at level ", level, " starting recording...")
                print("") # new line to prevent audio level from overwriting things
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

                        if not flags.saveReceivedAudio:
                            os.remove(f"{save_path}/rx-{filename}")

                        print(f'transcription: \"{transcription}\"')

                        if len(transcription) > 0:
                            #if transcription.endswith("control"):
                            #    lastUnit = transcription[0:transcription.find("control")]
                            #    generateSpokenResponse("control, go ahead", filename)
                            #elif (len(lastUnit) > 0):
                            #    if transcription in availablePhrases:
                            #        generateSpokenResponse("control is clear {$lastUnit}, you're in service", filename)
                            #    elif transcription in availablePhrases:
                            #        generateSpokenResponse("control is clear {$lastUnit}, you're out of service", filename)
                            #    else: # repeat back what they said
                            #        generateSpokenResponse(lastUnit + " unable to copy", filename)
                            #else: # repeat back what they said
                            #    generateSpokenResponse(transcription, filename)

                            response = promptResponse(transcription)

                            if response.get("error") is not None:
                                raise Exception("Error response from model: " + response.get("error"))
                            elif response.get("response") is None:
                                raise Exception("Reponse from model was None.")
                            
                            response = response.get("response")

                            print(f"Response: {response}")

                            if flags.delayTone is not None:
                                playTone(1000, flags.delayTone)
                            elif flags.delay is not None:
                                time.sleep(flags.delay)

                            generateSpokenResponse(response, filename)

                            # play the generated speech file
                            ffplay(f"{save_path}/tx-{filename}", "-af 'atempo=1.3'")

                            if flags.mdc:
                                ffplay(f"{save_path}/MDC1200.wav")

                            # delete the generated speech file
                            if not flags.saveSpokenAudio:
                                os.remove(f"{save_path}/tx-{filename}")

                        # TODO: last unit was unreadable
                        elif os.path.isfile(f"{save_path}/rx-{filename}"):
                            # rename the received audio to failed so we know
                            os.rename(f"{save_path}/rx-{filename}", f"failed-rx-{filename}")
                    else:
                        print(f"Sound stopped, discarding audio... Duration: {duration: .2f} seconds")
                    
                # Reset recording states
                recording = False
                frames = []
                start_time = None
                print("") # new line to prevent audio level from overwriting things

while True:
    try:
        processLoop()
    except Exception as e:
        print("An exception occurred:", e)
        print("The stream will be restarted.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt quit")
        exit()