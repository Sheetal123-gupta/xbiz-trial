from num2words import num2words
import re

def human_like_gtts(text, use_audio_enhancement=True):
    """
    Complete pipeline for human-like gTTS output
    """
    # Step 1: Text preprocessing
    enhanced_text = enhance_text_for_speech(text)
    
    # Step 2: Smart voice selection
    voice_params = get_best_voice(enhanced_text)
    
    # Step 3: Generate speech
    tts = gTTS(text=enhanced_text, **voice_params)
    
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    
    # Step 4: Audio enhancement
    if use_audio_enhancement:
        final_audio = enhance_audio(audio_bytes)
    else:
        final_audio = audio_bytes
    
    return final_audio

# Usage example
text = """
Hello! This is an important demonstration. 
We have 5 main points to cover today. 
First, remember that practice makes perfect. 
Second, always test your code. 
Third, use version control like Git. 
Fourth, document your work. 
And fifth, never stop learning!
"""

# Generate human-like speech
audio = human_like_gtts(text)
play_audio(audio)