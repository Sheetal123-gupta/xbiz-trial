from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import soundfile as sf

# Load the processor and model
processor = AutoProcessor.from_pretrained("ai4bharat/indic-parler-tts")
model = AutoModelForSpeechSeq2Seq.from_pretrained("ai4bharat/indic-parler-tts")

# Your input text
text = "नमस्ते, यह एआई4भारत की आवाज़ है।"

# Preprocess and generate speech
inputs = processor(text, return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], voice="hi_informal")

# Save audio
sf.write("out.mp3", speech.numpy(), samplerate=24000)
