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
    default = "1200-fdny",
    const = "1200-fdny",
    help = "simulate mdc tones"
)

parser.add_argument("-dontSaveTranscript",
    action = "store_true",
    default = False,
    help = "don't save received/transmitted transcript in logfile"
)

parser.add_argument("-saveReceivedAudio",
    action = "store_false",
    default = True,
    help = "save files containing received transmission audio"
)

parser.add_argument("-delayNoise",
    type = float,
    default = 0.3,
    help = "plays noise for x seconds to help a radio open vox"
)

parser.add_argument("-delayNoiseVolume",
    type = float,
    default = 1,
    help = "volume to play delayNoise at"
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

parser.add_argument("-voiceVolume",
    type = float,
    default = 1,
    help = "volume the voice is played back at"
)

parser.add_argument("-soundsVolume",
    type = float,
    default = 0.08,
    help = "volume sounds other than the voice are played back at"
)

parser.add_argument("-voiceEngine",
    choices = [
        "piper",
        "gtts-remote"
    ],
    nargs = "?",
    default = "piper",
    const = "piper",
    help = "which speech engine to use. piper is local and gtts is remote."
)

parser.add_argument("-voiceSpeed",
    type = float,
    default = 1.1,
    help = "speed at which the spoken response should be played back"
)

parser.add_argument("-piperVoice",
    type = str,
    default = "en_US-libritts_r-medium",
    help = "which voice model piper should use"
)

parser.add_argument("-ollamaModel",
    type = str,
    default = "gemma2:2b",
    help = "model that ollama should use"
)

parser.add_argument("-debug",
    action = "store_true",
    help = "print more debug information"
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

# Save file path
workingDirectory = os.path.join(os.path.expanduser('~'), 'Documents/GitHub/dispatcher')
recordingDirectory = os.path.join(workingDirectory, "recordings")
soundsDirectory = os.path.join(workingDirectory, "sounds")
voicesDirectory = os.path.join(workingDirectory, "voices")

# only so that whisper can download different models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

whisperModel = whisper.load_model("base")

# Initialize PyAudio
p = pyaudio.PyAudio()
MIC_STREAM_CHUNK_SIZE = 1024

if flags.voiceEngine == "piper":
    from piper.voice import PiperVoice

    model = f"{voicesDirectory}/{flags.piperVoice}.onnx"
    voice = PiperVoice.load(model)

    # in piper, 0.8 is faster and 1.3 is slower.
    # this formula reverses the number so that
    # speed is consistent between gtts and piper
    # 1.3 -> 0.8 and 0.8 -> 1.3 etc
    voice.config.length_scale = 1 - (flags.voiceSpeed - 1.0)

messageHistory = [
    {
        "role": "system",
        "content": "You are a researcher at a remote arctic research facility. The facility is named Station Alpha. Answer in the first person, using phrases like \"me\" or \"I\". Do not narrate. Do not make sound effects. Limit replies to 2 or 3 short sentences. "
    }
]

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

def beginTransmit():
    # if delayNoise is 0, the program will hang.
    if flags.delayNoise is not None and flags.delayNoise > 0:
        playNoise(lengthSeconds = flags.delayNoise)
    elif flags.delay is not None:
        time.sleep(flags.delay)

def endTransmit():
    if flags.mdc == "1200":
        playSound("mdc_eot/MDC1200")
    elif flags.mdc == "1200-fdny":
        playSound("mdc_eot/Saber1200")
    elif flags.mdc == "1200-dos":
        playSound("mdc_eot/MDC1200-DOS")
    elif flags.mdc == "random":
        # finds a random ifle in the mdc_eot directory and removes the last 4 chars from it (.wav)
        playRandomSoundInDirectory("mdc_eot")
    else:
        print(f"MDC EOT mode: {flags.mdc}")

def promptResponse(string):
    startTime = time.time()

    print("Prompting response...")

    messageHistory.append({
        "role": "user",
        "content": string
    })

    request = requests.post("http://localhost:11434/api/chat", json = {
        "model": flags.ollamaModel,
        "messages": messageHistory,
        "stream": False,
    })

    if flags.debug:
        print("Raw response content: ", request.content)

    print(f"Took {round(time.time() - startTime, 2)}s")

    response = request.json()

    if response.get("error") is not None:
        raise Exception("Error response from model: " + response.get("error"))
    elif response.get("message") is None:
        raise Exception("Message from model was None.")

    response = response.get("message")["content"]

    print(f"Response: {response}")
    return response

def speakResponse(text):
    global recordingDirectory

    startTime = time.time()

    print("Generating response audio...")

    text = text.replace("*", "")

    if flags.voiceEngine == "gtts-remote":
        print("Requesting generated speech...")

        filename = f"{recordingDirectory}/tx-{getNewRecordingFilename()}"

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

        beginTransmit()

        ffplay(filename, f"-af 'volume={flags.voiceVolume}' -af 'atempo={flags.voiceSpeed}'")

        os.remove(filename)

        endTransmit()
    else: # use piper
        print("Generating speech...")

        program = subprocess.Popen(
            [
                "ffplay",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "s16le",
                "-ar", str(voice.config.sample_rate),
                "-nodisp",
                "-infbuf",
                "-autoexit",
                "-af", f"volume={flags.voiceVolume}",
                # can't use atempo because it exits too early and cuts off the last phonem.
                # instead, we use piper's length_scale to change voice speed.
                "-i", "pipe:"
            ],
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE
        )

        beginTransmit()

        for byteData in voice.synthesize_stream_raw(text):
            program.stdin.write(byteData)

        program.stdin.close()
        program.wait()

        endTransmit()

    print(f"Took {round(time.time() - startTime, 2)}s")

def ffplay(filename, args = ""):
    print(f"Playing file {filename}...")
    return os.system(f"ffplay {args} \"{filename}\" -autoexit -nodisp -hide_banner -loglevel error")

def playSound(soundName):
    print(f"Playing file {soundName}...")
    return ffplay(f"{soundsDirectory}/{soundName}.wav", f"-af 'volume={flags.soundsVolume}'")

def playRandomSoundInDirectory(directory):
    # TODO: might also use os.path.isdir() ?
    playSound(f"{directory}/{random.choice([f for f in os.listdir(f"{soundsDirectory}/{directory}") if not f.startswith('.')])[:-4]}")

def playError():
    beginTransmit()
    playRandomSoundInDirectory("error")
    endTransmit()

def playNoise(lengthSeconds = 5):
    print(f"Playing noise for {lengthSeconds} seconds...")
    # does not use ffplay function bc that calls a file
    os.system(f"ffplay -f lavfi 'anoisesrc=a=0.1:c=white:d={lengthSeconds}' -af 'volume={flags.delayNoiseVolume}' -autoexit -nodisp -hide_banner -loglevel error")

def clearPreviousLine():
    print("\033[A", end="\r")
    print("\033[K", end="\r")

def getNewRecordingFilename():
    return f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"

def openMicrophoneStream():
    stream = p.open(
        format = pyaudio.paInt16,
        channels = 1,
        rate = 44100,
        input = True,
        frames_per_buffer = MIC_STREAM_CHUNK_SIZE
    )

    # throw away the first chunk of data to avoid
    # recording being started immediately
    stream.read(MIC_STREAM_CHUNK_SIZE, exception_on_overflow=False)

    return stream

def processLoop():
    frames = []
    recording = False
    start_time = None
    last_detection_time = 0
    
    micStream = openMicrophoneStream()

    print("Started stream, beginning processing...")
    print("") # new line to prevent audio level from overwriting things

    while True:
        data = micStream.read(MIC_STREAM_CHUNK_SIZE, exception_on_overflow=False)

        current_time = time.time()
        
        level = audioop.rms(data, 2)
        
        clearPreviousLine()
        print(f"Audio level: {level}")

        # sound is loud enough to record,
        # or the last detected sound was less than padDuration (eg 2 seconds) ago
        if level > flags.threshold or last_detection_time + flags.padDuration > current_time:
            if not recording:
                clearPreviousLine()
                print(f"Sound detected at level {level} starting recording...")
                print("") # new line to prevent audio level from overwriting things
                start_time = current_time  # Initialize the start time

            recording = True
            frames.append(data)

            if level > flags.threshold:
                last_detection_time = current_time
            
            # Check for max duration
            if start_time and current_time - start_time >= flags.maxDuration:
                clearPreviousLine()
                print("Max duration reached, stopping recording.")
                recording = False
        
        # sound is too quiet, or last detection + padDuration was earlier than the current time
        else:
            if recording:
                duration = current_time - start_time

                if last_detection_time + flags.padDuration < current_time:
                    if duration >= flags.minDuration:
                        clearPreviousLine()
                        print(f"Sound stopped at level {level} with duration of {duration: .2f} seconds")

                        micStream.close()

                        # Save the audio
                        filename = getNewRecordingFilename()
                        wf = wave.open(f"{recordingDirectory}/rx-{filename}", 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(44100)
                        wf.writeframes(b''.join(frames))
                        wf.close()

                        if not os.path.isfile(f"{recordingDirectory}/rx-{filename}"):
                            raise Exception("File for received audio did not exist to transcribe.")
                        
                        print("Transcribing audio...")
                
                        transcribeStartTime = time.time()

                        # fp16 is false because it generates a warning (at least on macos)
                        transcription = whisperModel.transcribe(f"{recordingDirectory}/rx-{filename}", fp16=False)
                        transcription = transcription['text'].strip(" .,\n").lower()

                        print(f'Transcription: \"{transcription}\"')
                        print(f"Took {round(time.time() - transcribeStartTime, 2)}s")

                        if not flags.saveReceivedAudio:
                            os.remove(f"{recordingDirectory}/rx-{filename}")

                        if flags.dontSaveTranscript is not True:
                            with open(f"{recordingDirectory}/transcript.log", "a") as transcript:
                                transcript.write(f"RX: {transcription}\n")

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

                            if flags.dontSaveTranscript is not True:
                                with open(f"{recordingDirectory}/transcript.log", "a") as transcript:
                                    transcript.write(f"TX: {response}")

                            speakResponse(response)

                        # if the length of the transcription is zero,
                        # check to see if we got an audio file at all.
                        elif os.path.isfile(f"{recordingDirectory}/rx-{filename}"):
                            # rename the received audio to failed so we know. failed can't come after filename because filename contains .wav
                            os.rename(f"{recordingDirectory}/rx-{filename}", f"{recordingDirectory}/rx-failed-{filename}")

                            playError()
                    else:
                        print(f"Sound stopped, discarding audio... Duration: {duration: .2f} seconds")
                
                # Reset recording states
                frames = []
                recording = False
                start_time = None
                last_detection_time = 0
                micStream = openMicrophoneStream()
                print("Resetting recording state") # new line to prevent audio level from overwriting things

while True:
    try:
        processLoop()
    except Exception as e:
        print(f"A {type(e).__name__} exception occurred:", e)
        print("The stream will be restarted.")
        playError()
    except KeyboardInterrupt:
        print("KeyboardInterrupt quit")
        exit()