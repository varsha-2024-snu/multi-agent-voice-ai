from voice import speak
from time import sleep
from voice import text_to_speech

print("Testing Optimist Voice (luna)...")
text_to_speech("Hello! I am the optimistic voice!", agent="optimist")
sleep(2)  

print("Testing Realist Voice (storm)...")
text_to_speech("This is the realist giving practical advice.", agent="realist")
sleep(2)

print("Testing Planner Voice (echo)...")
text_to_speech("Hi, I'm the planner voice in your system.", agent="planner")


# NOT FOR IMPLEMENTATION, JUST FOR DEEPGRAM TESTING AND DEBUGGING