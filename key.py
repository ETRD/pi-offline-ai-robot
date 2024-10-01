from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device, Button
from signal import pause

Device.pin_factory = LGPIOFactory()

# Define the button object and specify the BCM pin number
key1 = Button(4)
key2 = Button(17)
key3 = Button(23)
key4 = Button(24)
def key1_pressed():
    print("Press KEY1(BCM4)")

def key2_pressed():
    print("Press KEY2(BCM17)")

def key2_released():
    print("Release KEY2(BCM17)")

def key3_pressed():
    print("Press KEY3(BCM23)")

def key4_pressed():
    print("Press KEY4(BCM24)")

# Bind key press event
key1.when_pressed = key1_pressed
key2.when_pressed = key2_pressed
key3.when_pressed = key3_pressed
key4.when_pressed = key4_pressed

key2.when_released = key2_released

# Keep the program running
pause()