import pyttsx3
import time

def speak_text(text, voice_index=0, rate=90, volume=2.0):
    engine = pyttsx3.init()
    
    # Set speed (words per minute)
    engine.setProperty('rate', rate)
    
    # Set volume (0.0 to 1.0)
    engine.setProperty('volume', volume)
    
    # Select voice (default index 0)
    voices = engine.getProperty('voices')
    if voice_index >= len(voices):
        voice_index = 0  # fallback
    engine.setProperty('voice', voices[voice_index].id)
    
    # Split text into sentences to add short pauses for more natural delivery
    sentences = [s.strip() for s in text.replace('।', '.').split('.')]
    for sentence in sentences:
        if sentence:
            engine.say(sentence)
            engine.runAndWait()
            time.sleep(0.2)  # small pause between sentences

# -----------------------------
# Main program
# -----------------------------
if __name__ == "__main__":
    print("============================== start typing ur text =======================")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input(" type text here : ")
        if user_input.lower() == "exit":
            break
        
        # Speak the typed text
        speak_text(user_input, rate=10, volume=1.0)



#sk_9125ec1584b0cd2d414f71ee45df58c1fff220ae167e9187
