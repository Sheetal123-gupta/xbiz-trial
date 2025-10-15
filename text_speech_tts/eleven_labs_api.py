from elevenlabs import ElevenLabs
import os

# Initialize client with your API key
client = ElevenLabs(api_key="sk_9125ec1584b0cd2d414f71ee45df58c1fff220ae167e9187")

# Choose a voice ID (you can list all available voices below)
voice_id = "C8wZRioDZqA6fkwDW6Df"  

# Generate speech
result = client.text_to_speech.convert(
    voice_id=voice_id,
    model_id="eleven_multilingual_v2",
    text="ज़रूर! अगर आप चाहते हैं कि कोई जानकारी Google पर दिखने वाले तरीके से हिंदी में लिखी जाए — जैसे संक्षिप्त, स्पष्ट और तथ्यात्मक — तो मैं उसी शैली में प्रस्तुत कर सकता हूँ उदाहरण के लिए, अगर विषय है Python में Text to Speech कैसे करें, तो Google-शैली में हिंदी जानकारी कुछ इस तरह होगी"
)

# Save the output audio to file
output_file = "output.mp3"
with open(output_file, "wb") as f:
    for chunk in result:
        f.write(chunk)

# Play the file (on Windows)
os.system(f"start {output_file}")
