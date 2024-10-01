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

import threading

start_record_event = threading.Event()
stop_record_event = threading.Event()

Device.pin_factory = LGPIOFactory()

# key init
key2 = Button(17)

def key2_pressed():
    start_record_event.set()
    print("Press KEY2(BCM17)")
def key2_released():
    stop_record_event.set()
    print("Release KEY2(BCM17)")

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
                break
                # Read data from device
            l, data = mic.read()
            
            if l:
                wavfile.writeframes(data)
                time.sleep(.001)

thread_record = threading.Thread(target=recording_thread) 

thread_record.start()

thread_record.join()

# Keep the program running
pause()