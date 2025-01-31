import argparse

parser = argparse.ArgumentParser(
    prog = "Dispatcher",
    description = "Uses AI to transcribe messages from an audio input, then generate a response based on that message and play it to an output."
)

parser.add_argument("-mdcStart",
    choices = [
        "MDC1200",
        "MDC1200-DOS",
        "MDC1200-Saber",
        "random"
    ],
    nargs = "?",
    default = "MDC1200-DOS",
    const = "MDC1200-DOS",
    help = "simulate mdc tones at the beginning of the transmission"
)

parser.add_argument("-mdcEnd",
    choices = [
        "MDC1200",
        "MDC1200-DOS",
        "MDC1200-Saber",
        "random"
    ],
    nargs = "?",
    default = "MDC1200-Saber",
    const = "MDC1200-Saber",
    help = "simulate mdc tones at the end of the transmission"
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
    default = 0.4,
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
    default = 2,
    help = "how long in seconds to keep listening before a received transmission is considered to be over"
)

parser.add_argument("-maxDuration",
    type = float,
    default = 30,
    help = "max duration of a transmission in seconds"
)

parser.add_argument("-threshold",
    type = int,
    default = 500,
    help = "start recording once audio levels are above this value. when audio levels are below, the recording will be considered to be over."
)

parser.add_argument("-voiceVolume",
    type = float,
    default = 0.5,
    help = "volume the voice is played back at"
)

parser.add_argument("-soundsVolume",
    type = float,
    default = 1,
    help = "volume sounds other than the voice are played back at"
)

parser.add_argument("-voiceSpeed",
    type = float,
    default = 1.2,
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
    help = "name of the llm that ollama should use"
)

parser.add_argument("-ollamaUri",
    type = str,
    default = "http://localhost:11434",
    help = "uri to ollama server. do not include trailing slash."
)

parser.add_argument("-prompt",
    type = str,
    default = "ResearchStation",
    help = "url to prompt used by llm"
)

parser.add_argument("-idleDelay",
    type = int,
    default = 120,
    help = "how long to wait before beginning idle chat using the llm. -1 disables idle messages"
)

parser.add_argument("-idleIntervalMin",
    type = int,
    default = 5,
    help = "minimum time between idle messages"
)

parser.add_argument("-idleIntervalMax",
    type = int,
    default = 60,
    help = "max time before next idle message"
)

parser.add_argument("-initialStreamChunkDiscardCount",
    type = int,
    default = 2,
    help = "number of chunks to discard from the microphone stream after it has been initialised. helps to prevent recording from being triggered immediately after the stream is opened."
)

parser.add_argument("-debug",
    action = "store_true",
    help = "print more debug information"
)

flags = parser.parse_args()

print(flags.__dict__)

import audioop
import datetime
import json
import os
from piper.voice import PiperVoice
import pyaudio
import requests
import random
import subprocess
import sys
import time
import wave
import whisper

workingDirectory = os.path.join(os.path.expanduser("~"), "Documents/GitHub/dispatcher")
recordingDirectory = os.path.join(workingDirectory, "recordings")
soundsDirectory = os.path.join(workingDirectory, "sounds")
voicesDirectory = os.path.join(workingDirectory, "voices")
promptsDirectory = os.path.join(workingDirectory, "prompts")

# only so that whisper can download different models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

whisperModel = whisper.load_model("base")

p = pyaudio.PyAudio()
MIC_STREAM_CHUNK_SIZE = 1024

def loadRandomPiperVoice():
    return loadPiperVoice(random.choice([f for f in os.listdir(f"{voicesDirectory}/") if not f.startswith(".") and not f.endswith(".json")]))

def loadPiperVoice(voice):
    print(f"Loading voice: {voice}")

    voice = PiperVoice.load(f"{voicesDirectory}/{voice}")

    # in piper, 0.8 is faster and 1.3 is slower.
    # this formula reverses the number so that
    # speed is consistent between gtts and piper
    # 1.3 -> 0.8 and 0.8 -> 1.3 etc
    voice.config.length_scale = 1 - (flags.voiceSpeed - 1.0)

    return voice

if flags.piperVoice == "random":
    dispatcherVoice = loadRandomPiperVoice()
else:
    dispatcherVoice = loadPiperVoice(f"{flags.piperVoice}.onnx")

voice2 = loadPiperVoice(f"en_GB-alba-medium.onnx")

with open(f"{promptsDirectory}/{flags.prompt}.txt", "r") as promptFile:
    prompt = promptFile.read()
    prompt = json.loads(prompt)

userMessageHistory = [
    {
        "role": "system",
        "content": prompt["primary"]
    }
]

lastIdleMessageTime = 0
nextIdleMessageDelay = flags.idleIntervalMin
lastIdleMessage = "Karnaka Station, this is research command, radio check."
lastIdleSpeaker = None

micStream = None

lastUnit = None

availablePhrases = [
    "i'm 10-8",
    "i'm 10.8",
    "i'm 108",
    "i'm ten eight"
    "i'm available",
    "i'm tonight",
    "i'm available",
    "i'm in service",

    "show me 10-8",
    "show me 10.8",
    "show me 108",
    "show me ten eight"
    "show me available",
    "show me tonight",
    "show me available",
    "show me in service",

    "show us 10-8",
    "show us 10.8",
    "show us 108",
    "show us ten eight",
    "show us available",
    "show us tonight",
    "show us available",
    "show us in service",

    "we're 10-8",
    "we're 10.8",
    "we're 108",
    "we're ten eight",
    "we're available",
    "we're tonight",
    "we're available",
    "we're in service",
]

unavailablePhrases = [
    "i'm 10-7",
    "i'm 10.7",
    "i'm 107",
    "i'm ten seven",
    "i'm unavailable",
    "i'm out of service",
    "i'm ten seven",

    "show me 10-7",
    "show me 10.7",
    "show me 107",
    "show me ten seven",
    "show me unavailable",
    "show me out of service",
    "show me ten seven",

    "show us 10-7",
    "show us 10.7",
    "show us 107",
    "show us ten seven",
    "show us unavailable",
    "show us out of service",
    "show us ten seven",

    "we're 10-7",
    "we're 10.7",
    "we're 107",
    "we're ten seven",
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
    
    if flags.mdcStart is not None:
        if flags.mdcStart == "random":
            playRandomSoundInDirectory("mdc")
        else:
            playSound(f"mdc/{flags.mdcStart}")

def endTransmit():
    if flags.mdcEnd is not None:
        if flags.mdcEnd == "random":
            playRandomSoundInDirectory("mdc")
        else:
            playSound(f"mdc/{flags.mdcEnd}")

def promptResponse(messageHistory):
    startTime = time.time()

    print("Prompting response...")

    request = requests.post(f"{flags.ollamaUri}/api/chat", json = {
        "model": flags.ollamaModel,
        "messages": messageHistory,
        "stream": False,
    })

    if flags.debug:
        print("Raw response content: ", request.content)

    print(f"Reponse took {round(time.time() - startTime, 2)}s to generate.")

    response = request.json()

    if response.get("error") is not None:
        raise Exception("Error response from model: " + response.get("error"))
    elif response.get("message") is None:
        raise Exception("Message from model was None.")

    response = response.get("message")["content"]

    print(f"Response: {response}")
    return response

def resetMessageHistory():
    userMessageHistory[1:]
    print("Message history reset.")

def speakResponse(text, voice):
    global recordingDirectory

    startTime = time.time()

    print("Generating response audio...")

    text = text.replace("*", "")

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
    playSound(f"{directory}/{random.choice([f for f in os.listdir(f"{soundsDirectory}/{directory}") if not f.startswith(".")])[:-4]}")

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
    return f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.wav"

def appendToTranscript(text):
    if flags.dontSaveTranscript is not True:
        with open(f"{recordingDirectory}/transcript.log", "a") as transcriptFile:
            transcriptFile.write(f"RX: {text}\n")

def openMicrophoneStream():
    global micStream

    print("Opening micStream.")

    micStream = p.open(
        format = pyaudio.paInt16,
        channels = 1,
        rate = 44100,
        input = True,
        frames_per_buffer = MIC_STREAM_CHUNK_SIZE
    )

    # throw away the first chunk of data to avoid
    # recording being started immediately
    for x in range(0, flags.initialStreamChunkDiscardCount):
        micStream.read(MIC_STREAM_CHUNK_SIZE, exception_on_overflow=False)

def closeMicStream():
    global micStream

    print("Closing micStream.")

    micStream.close()

def processLoop():
    global lastUnit
    global lastIdleMessage
    global lastIdleSpeaker
    global lastIdleMessageTime
    global nextIdleMessageDelay
    global micStream

    print("Beginning process loop")

    frames = []
    recording = False
    recordingStartTime = None
    lastDetectionTime = time.time()
    
    openMicrophoneStream()

    print("Started stream, beginning processing...")

    while True:
        data = micStream.read(MIC_STREAM_CHUNK_SIZE, exception_on_overflow=False)

        currentTime = time.time()
        timeSinceLastDetection = currentTime - lastDetectionTime
        
        level = audioop.rms(data, 2)
        
        clearPreviousLine()
        print(f"Audio level ({timeSinceLastDetection: .1f}s): {level}.")

        # sound is loud enough to record,
        # or the last detected sound was less than padDuration (eg 2 seconds) ago
        if level > flags.threshold or (recording and lastDetectionTime + flags.padDuration > currentTime):
            if not recording:
                clearPreviousLine()
                print(f"Sound detected at level {level} (threshold is {flags.threshold}) starting recording...")
                print("") # new line to prevent audio level from overwriting things
                recordingStartTime = currentTime  # Initialize the start time

            recording = True
            frames.append(data)

            if level > flags.threshold:
                lastDetectionTime = currentTime
            
            # Check for max duration
            if recordingStartTime and currentTime - recordingStartTime >= flags.maxDuration:
                clearPreviousLine()
                print("Max duration reached, stopping recording.")
                recording = False
        
        # sound below threshold, or last detection + padDuration was earlier than the current time
        else:
            if recording:
                duration = currentTime - recordingStartTime

                if lastDetectionTime + flags.padDuration < currentTime:
                    if duration >= flags.minDuration:
                        totalProcessStart = time.time()

                        clearPreviousLine()
                        print(f"Sound stopped at level {level} with duration of {duration: .2f} seconds")

                        closeMicStream()

                        # Save the audio
                        filename = getNewRecordingFilename()
                        wf = wave.open(f"{recordingDirectory}/rx-{filename}", "wb")
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
                        transcription = transcription["text"].strip(" .,\n").lower()

                        print(f"Transcription: \"{transcription}\"")
                        print(f"Transcription took {(time.time() - transcribeStartTime): .1f}s")

                        if not flags.saveReceivedAudio:
                            os.remove(f"{recordingDirectory}/rx-{filename}")

                        appendToTranscript(f"RX: {transcription}")

                        userMessageHistory.append({
                            "role": "user",
                            "content": transcription
                        })

                        if len(transcription) > 0:
                            if transcription == "innoculate shield pacify":
                                resetMessageHistory()
                                response = "Innoculation complete."
                            # acquires the unit who is speaking, for use later
                            elif transcription.endswith("control"):
                                lastUnit = transcription[0:transcription.find("control")]
                                resetMessageHistory()
                                response = "control, goahead"
                            # if we're in a conversation with a specific unit
                            elif lastUnit is not None:
                                if availablePhrases.count(transcription):
                                    response = f"control is clear {lastUnit}, you're in service"
                                elif unavailablePhrases.count(transcription):
                                    response = f"control is clear {lastUnit}, you're out of service"
                                elif transcription.endswith("clear"):
                                    lastUnit = None
                                    continue # TODO: this causes an exception to be thrown, but that's okay because it sounds cool.
                                else: # repeat back what they said
                                    response = lastUnit + " unable to copy, please say again."
                            else: # repeat back what they said
                                response = promptResponse(userMessageHistory)

                            appendToTranscript(f"TX: {response}")

                            userMessageHistory.append({
                                "role": "assistant",
                                "content": response
                            })

                            speakResponse(response, dispatcherVoice)

                        # if the length of the transcription is zero,
                        # check to see if we got an audio file at all.
                        #elif os.path.isfile(f"{recordingDirectory}/rx-{filename}"):
                        #    # rename the received audio to failed so we know. failed can't come after filename because filename contains .wav
                        #    os.rename(f"{recordingDirectory}/rx-{filename}", f"{recordingDirectory}/rx-failed-{filename}")

                        #    playError()

                        print(f"Total processing time: {round(time.time() - totalProcessStart, 2)}s")
                    else:
                        print(f"Sound stopped, discarding audio... Duration: {duration: .2f} seconds")
                
                # Reset recording states
                frames = []
                recording = False
                recordingStartTime = None
                openMicrophoneStream()
                print("Resetting recording state") # new line to prevent audio level from overwriting things
            else: # if not recording, do idle messages
                if flags.idleDelay > -1 and currentTime > lastDetectionTime + flags.idleDelay:
                    if currentTime > lastIdleMessageTime + nextIdleMessageDelay:
                        print(f"Responding to last idle message.")

                        if lastIdleSpeaker is voice2:
                            lastIdleSpeaker = dispatcherVoice
                        else:
                            lastIdleSpeaker = voice2

                        messages = [
                            {
                                "role": "system",
                                "content": prompt["primary"]
                            },
                            {
                                "role": "system",
                                "content": "Reply to the next message with this context: " + prompt["idle1"] if lastIdleSpeaker is dispatcherVoice else prompt["idle2"]
                            },
                            {
                                "role": "user",
                                "content": lastIdleMessage
                            }
                        ]

                        lastIdleMessage = promptResponse(messages)

                        speakResponse(lastIdleMessage, lastIdleSpeaker)

                        lastIdleMessageTime  = time.time()
                        nextIdleMessageDelay = random.randint(flags.idleIntervalMin, flags.idleIntervalMax)

                        print(f"Next idle message delay: {nextIdleMessageDelay}s")
                        print("") # new line to prevent audio level from overwriting things

while True:
    try:
        processLoop()
    except Exception as e:
        print(f"A {type(e).__name__} exception occurred on line {e.__traceback__.__dict__}:", e)
        print("The stream will be restarted.")
        playError()
    except KeyboardInterrupt:
       print("KeyboardInterrupt quit")
       closeMicStream()
       exit()