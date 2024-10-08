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
from datetime import datetime

# for SenseVoice
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# for llama
import llama_cpp
import re

# for tts
import pyttsx4
import subprocess
import psutil

import threading
import queue

# for oled
from PIL import Image, ImageSequence
from PIL import ImageFont, ImageDraw
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

serial_64 = i2c(port=1, address=0x3D)
oled_right = ssd1306(serial_64, width=128, height=64)
serial_32 = i2c(port=1, address=0x3C)
oled_left = ssd1306(serial_32, width=128, height=64)

start_record_event = threading.Event()
stop_record_event = threading.Event()
sensevoice_event = threading.Event()
llama_event = threading.Event()
tts_event = threading.Event()
stop_tts_event = threading.Event()

llama_load_done = threading.Event()
senvc_load_done = threading.Event()

ask_text_q = queue.Queue()
ans_text_q = queue.Queue()


Device.pin_factory = LGPIOFactory()

# key init
key2 = Button(17)

def key2_pressed():
    start_record_event.set()
    stop_tts_event.set()
def key2_released():
    stop_record_event.set()
    stop_tts_event.clear()

# Bind key press event
key2.when_pressed = key2_pressed
key2.when_released = key2_released



#mic.setperiodsize(160)

def recording_thread():
    while True:
        start_record_event.wait()
        start_record_event.clear()
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
        print("Start speaking...")
        time_start = datetime.now()
        while True:
            if stop_record_event.is_set():
                stop_record_event.clear()
                time_stop = datetime.now()
                print("Stop speaking...")
                wavfile.close()
                if (time_stop.timestamp() - time_start.timestamp() >= 1):
                    sensevoice_event.set()
                else:
                    print('The speaking time is too short')
                break
                # Read data from device
            l, data = mic.read()
            
            if l:
                wavfile.writeframes(data)
                time.sleep(.001)

def sensevoice_thread():
    model_dir = "./models/SenseVoiceSmall"
    m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()
    senvc_load_done.set()
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
    verbose = False,
    )
    ch_punctuations_re = "[，。？；]"
    llama_load_done.set()
    print("Load llama model done")
    while True:
        llama_event.wait()
        llama_event.clear()
        ask_text = ask_text_q.get()
        print(ask_text)
        ans_text = model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": f"{ask_text}"
            }],
            logprobs=False,
            #stream=True,
            max_tokens = 80,
        )
        ans_text = ans_text['choices'][0]['message']['content']
        print(ans_text)
        ans_text_tts = ans_text.replace("，", "。")
        ans_text_q.put(ans_text_tts)
#        ans_text = ""
#        for chunk in ans_stream:
#            delta = chunk['choices'][0]['delta']
#            if 'role' in delta:
#                print(delta['role'], end=': ')
#            elif 'content' in delta:
#                #print(delta['content'], end='')
#                ans_text += delta['content']
#                if re.search(ch_punctuations_re, ans_text):
#                    print(ans_text, end='')
#                    ans_text_q.put(ans_text)
#                    ans_text = ""

def terminate_process(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        time.sleep(0.5)
        if parent.is_running():
            parent.kill()
    except Exception as e:
        print(f"Error terminating process: {e}")

def tts_thread():
    #engine = pyttsx4.init()
    #engine.setProperty('voice', 'zh')
    piper_cmd = './piper/piper --model ./models/piper/zh_CN-huayan-medium.onnx --output-raw | aplay -r 22050 -f S16_LE -t raw -'
    while True:
        tts_text = ans_text_q.get()
        command = f"echo '{tts_text}' | {piper_cmd}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while True:
            if stop_tts_event.is_set():
                terminate_process(process.pid)
                break
            if process.poll() is not None:
                break
            time.sleep(0.01)
        process.wait()
        #engine.say(tts_text)
        #engine.runAndWait()

def oled_thread(oled_device, dir):
    with Image.open(f"./img/{dir}_logo.bmp") as img:
        img_resized = img.convert("1").resize((128, 64))
        oled_device.display(img_resized)
        llama_load_done.wait()
        senvc_load_done.wait()
    frames = []
    durations = [] 
    with Image.open(f"./img/{dir}_eye.gif") as img:
        for frame in ImageSequence.Iterator(img):
            frames.append(frame.convert("1").resize((128,64)))
            durations.append(frame.info.get('duration', 100) / 1000.0)
    while True:
    # Display each image with an idea
        for frame, duration in zip(frames, durations):
            oled_device.display(frame)
            time.sleep(duration*2)


thread_record = threading.Thread(target=recording_thread)
thread_sensevoice = threading.Thread(target=sensevoice_thread)
thread_llama = threading.Thread(target=llama_thread)
thread_tts = threading.Thread(target=tts_thread)
thread_oled_left = threading.Thread(target=oled_thread, args=(oled_left,"left"))
thread_oled_right = threading.Thread(target=oled_thread, args=(oled_right,"right"))

thread_record.start()
thread_sensevoice.start()
thread_llama.start()
thread_tts.start()
thread_oled_left.start()
thread_oled_right.start()

thread_record.join()
thread_sensevoice.join()
thread_llama.join()
thread_tts.join()
thread_oled_left.join()
thread_oled_right.join()

# Keep the program running
pause()