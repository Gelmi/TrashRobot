import speech_recognition as sr
import pyttsx3
from ctypes import *

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


# This redirects the C-level stderr to /dev/null
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so.2')
asound.snd_lib_error_set_handler(c_error_handler)


def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    command = ""
    try:
        with sr.Microphone() as source:
            print('Listening...')
            voice = listener.listen(source)
            command = listener.recognize_whisper(voice)
            command = command.lower()
    except sr.UnknownValueError:
        pass
    except sr.RequestError:
        pass
    except Exception as e:
        print(f"Error: {e}")
    return command


def run_voice_assistant():
    command = take_command()
    print(command)
    if 'robot' in command or 'robert' in command or 'george' in command:
        if 'help me' in command or 'helped me' in command or 'helping' in command:
            talk('How can I assist you?')
        else:
            talk('Sorry, I did not understand that.')


while True:
    run_voice_assistant()

