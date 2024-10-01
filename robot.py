# for key
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device, Button
from signal import pause

# for audio
import sys
import time
import getopt
import wave
import alsaaudio

# for SenseVoice
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# for llama
import llama_cpp

import threading
import queue

start_record_event = threading.Event()
stop_record_event = threading.Event()
sensevoice_event = threading.Event()
llama_event = threading.Event()
ask_text_q = queue.Queue()


Device.pin_factory = LGPIOFactory()

# key init
key2 = Button(17)

def key2_pressed():
    start_record_event.set()
def key2_released():
    stop_record_event.set()

# Bind key press event
key2.when_pressed = key2_pressed
key2.when_released = key2_released



#mic.setperiodsize(160)

def recording_thread():
    while True:
        start_record_event.wait()
        start_record_event.clear()
        print("Start record...")
        # record audio init
        device = 'default'
        wavfile = wave.open("test.wav", 'wb')

        # Open the device in nonblocking capture mode. The last argument could
        # just as well have been zero for blocking mode. Then we could have
        # left out the sleep call in the bottom of the loop
        mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, channels=1, rate=44100, format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=160, device=device)

        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)    #PCM_FORMAT_S16_LE
        wavfile.setframerate(44100)

        while True:
            if stop_record_event.is_set():
                stop_record_event.clear()
                print("Stop record...")
                wavfile.close()
                sensevoice_event.set()
                break
                # Read data from device
            l, data = mic.read()
            
            if l:
                wavfile.writeframes(data)
                time.sleep(.001)

def sensevoice_thread():
    model_dir = "iic/SenseVoiceSmall"
    m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()
    print("Load sensevoid model done")

    while True:
        sensevoice_event.wait()
        sensevoice_event.clear()
        res = m.inference(
            data_in=f"./test.wav",
            language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
            ban_emo_unk=False,
            **kwargs,
        )

        text = rich_transcription_postprocess(res[0][0]["text"])
        ask_text_q.put(text)
        llama_event.set()


def llama_thread():
    model = llama_cpp.Llama(
    model_path="./models/llama/qwen1_5-0_5b-chat-q4_0.gguf",
    )
    print("Load llama model done")
    while True:
        llama_event.wait()
        llama_event.clear()
        ask_text = ask_text_q.get()
        print(ask_text)
        ans_text = model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": f"{ask_text}, 回答在50个字以内"
            }],
            logprobs=False
        )
        ans_text = ans_text['choices'][0]['message']['content']
        print(ans_text)

thread_record = threading.Thread(target=recording_thread)
thread_sensevoice = threading.Thread(target=sensevoice_thread)
thread_llama = threading.Thread(target=llama_thread)

thread_record.start()
thread_sensevoice.start()
thread_llama.start()

thread_record.join()
thread_sensevoice.join()
thread_llama.join()

# Keep the program running
pause()