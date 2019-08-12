import pyttsx3
# from run_bicep_curl import bicepCount
import sys

print(sys.argv)

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("rate", 165)
engine.setProperty("voice", "english-us")

engine.say("Number {}".format(sys.argv[1]))

engine.runAndWait()
