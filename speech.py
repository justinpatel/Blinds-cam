import speech_recognition as sr
import pyttsx3

class speech_to_text():
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine= pyttsx3.init()

    def speak(self,message):
        self.engine.say('{}'.format(message))
        self.engine.runAndWait()    

    def recognize_speech_from_mic(self):
        #print("Listening...")
        #speak("Listening")
        text = None
        with self.microphone as source:
            #speak('speak')
            #print('Speak :')
            #print('listening')
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                print('You said: {}'.format(text))
            except:
                #print("Sorry could not recognize your voice")
                self.speak('Sorry could not recognize your voice')

        return text
                

    
        
