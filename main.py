import argparse

parser = argparse.ArgumentParser(
    prog = "Dispatcher",
    description = "Uses AI to transcribe messages from an audio input, then generate a response based on that message and play it to an output."
)

parser.add_argument("-mdc",
    choices = [
        "1200",
        "1200-dos",
        "1200-fdny",
        "random"
    ],
    nargs = "?",
    default = "1200",
    const = "1200",
    help = "simulate mdc tones"
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

parser.add_argument("-minDuration",
    type = float,
    default = 1,
    help = "received transmission must be at least this long in seconds to be considered"
)

parser.add_argument("-padDuration",
    type = float,
    default = 2.5,
    help = "how long in seconds to keep listening before a received transmission is considered to be over"
)

parser.add_argument("-maxDuration",
    type = float,
    default = 30,
    help = "max duration of a transmission in seconds"
)

parser.add_argument("-threshold",
    type = int,
    default = 800,
    help = "start recording once audio levels are above this value. when audio levels are below, the recording will be considered to be over."
)

flags = parser.parse_args()

print(flags.__dict__)

import audioop
import datetime
import os
import pyaudio
import requests
import random
import subprocess
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

# Save file path
workingDirectory = os.path.join(os.path.expanduser('~'), 'Documents/GitHub/dispatcher')
recordingDirectory = os.path.join(workingDirectory, "recordings")
soundsDirectory = os.path.join(workingDirectory, "sounds")

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
    global workingDirectory

    print("Generating response audio...")

    subprocess.check_call(
        [
            "gtts-cli",
            #"--debug",
            #"--lang", "en",
            "--output", filename,
            text
        ],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT
    )

    # how much of the clip to cut off
    cutTime = 0.4

    # cuts off the last little bit of audio, because google leaves some hang time and i want the mdc tones to be right after speech is finished
    os.system(f"ffmpeg -i {filename} -ss 0 -to $(echo $(ffprobe -i {filename} -show_entries format=duration -v quiet -of csv='p=0') - {cutTime} | bc) -c copy -f wav {filename}_new")

    os.remove(filename)
    # give the new file the correct name
    os.rename(f"{filename}_new", filename)

def ffplay(filename, args = ""):
    print(f"Playing file {filename}...")
    return os.system(f"ffplay {args} \"{filename}\" -autoexit -nodisp -hide_banner -loglevel error")

def playTone(freq = 300, lengthSeconds = 5):
    print(f"Playing {freq}hz tone for {lengthSeconds} seconds...")
    # does not use ffplay function bc that calls a file
    os.system(f"ffplay -f lavfi 'anoisesrc=a=0.1:c=white:d={lengthSeconds}' -autoexit -nodisp -hide_banner -loglevel error")

def clearPreviousLine():
    print("\033[A", end="\r")
    print("\033[K", end="\r")

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
        
        clearPreviousLine()
        print(f"Audio level: {level}")

        # sound is loud enough to record, or the pad duration has not elapsed
        if level > flags.threshold or last_detection_time + flags.padDuration > current_time:
            if not recording:
                clearPreviousLine()
                print("Sound detected at level ", level, " starting recording...")
                print("") # new line to prevent audio level from overwriting things
                start_time = current_time  # Initialize the start time

            recording = True
            frames.append(data)

            if level > flags.threshold:
                last_detection_time = current_time
            
            # Check for max duration
            if start_time and current_time - start_time >= flags.maxDuration:
                clearPreviousLine()
                print("Max duration reached, terminating.")
                recording = False
        
        # sound is too quiet
        else:
            if recording:
                duration = current_time - start_time

                if last_detection_time + flags.padDuration < current_time:
                    if duration >= flags.minDuration:
                        clearPreviousLine()
                        print(f"Sound stopped at level {level} with duration of {duration: .2f} seconds")
                        # Save the audio
                        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
                        wf = wave.open(f"{recordingDirectory}/rx-{filename}", 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(44100)
                        wf.writeframes(b''.join(frames))
                        wf.close()

                        if not os.path.isfile(f"{recordingDirectory}/rx-{filename}"):
                            raise Exception("File for received audio did not exist to transcribe.")
                
                        # fp16 is false because it generates a warning (at least on macos)
                        transcription = whisperModel.transcribe(f"{recordingDirectory}/rx-{filename}", fp16=False)
                        transcription = transcription['text'].strip(" .,\n").lower()

                        if not flags.saveReceivedAudio:
                            os.remove(f"{recordingDirectory}/rx-{filename}")

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

                            generateSpokenResponse(response, f"{recordingDirectory}/tx-{filename}")

                            # if delayTone is 0, the program will hang.
                            if flags.delayTone is not None and flags.delayTone > 0:
                                playTone(lengthSeconds = flags.delayTone)
                            elif flags.delay is not None:
                                time.sleep(flags.delay)

                            # play the generated speech file
                            ffplay(f"{recordingDirectory}/tx-{filename}", "-af 'atempo=1.3'")

                            if flags.mdc == "1200":
                                ffplay(f"{soundsDirectory}/mdc_eot/MDC1200.wav")
                            elif flags.mdc == "1200-fdny":
                                ffplay(f"{soundsDirectory}/mdc_eot/Saber1200.wav")
                            elif flags.mdc == "1200-dos":
                                ffplay(f"{soundsDirectory}/mdc_eot/MDC1200-DOS.wav")
                            elif flags.mdc == "random":
                                ffplay(f"{soundsDirectory}/mdc_eot/" + random.choice(os.listdir(f"{soundsDirectory}/mdc_eot")))
                            else:
                                print(f"MDC EOT mode: {flags.mdc}")

                            # delete the generated speech file
                            if not flags.saveSpokenAudio:
                                os.remove(f"{recordingDirectory}/tx-{filename}")

                        # TODO: last unit was unreadable
                        elif os.path.isfile(f"{recordingDirectory}/rx-{filename}"):
                            # rename the received audio to failed so we know. failed can't come after filename because filename contains .wav
                            os.rename(f"{recordingDirectory}/rx-{filename}", f"{recordingDirectory}/rx-failed-{filename}")
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