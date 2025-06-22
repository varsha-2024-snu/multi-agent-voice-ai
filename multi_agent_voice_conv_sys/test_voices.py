from stt_tts import speak
from time import sleep

print("Testing Optimist Voice (luna)...")
speak("Hello! I am the optimistic voice!", agent="optimist")
sleep(2)  

print("Testing Realist Voice (storm)...")
speak("This is the realist giving practical advice.", agent="realist")
sleep(2)

print("Testing Planner Voice (echo)...")
speak("Hi, I'm the planner voice in your system.", agent="planner")


# NOT FOR IMPLEMENTATION, JUST FOR DEEPGRAM TESTING AND DEBUGGING