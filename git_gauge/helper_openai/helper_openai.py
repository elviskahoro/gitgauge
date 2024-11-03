from __future__ import annotations
from pathlib import Path
from openai import OpenAI
import os


def openai_text_to_speech(
    prompt: str,
    api_key: str = os.getenv("OPENAI_API_KEY"),
):
    """Generates a brief response using OpenAI's API and converts the text to speech."""
    client = OpenAI()
    try:
        speech_file_path = Path(
            "/Users/elvis/Documents/elviskahoro/git_gauge/assets/speech.mp3"
        )
        # Generate text using OpenAI
        response = client.audio.speech.create(
            model="tts-1", voice="alloy", input=prompt
        )
        response.stream_to_file(speech_file_path)

        return response

    except Exception as e:
        print(f"Error: {e}")
        return "Failed to generate text or convert to speech due to an error."


def openai_text_to_speech_for_swot(
    website: str,
    api_key: str = os.getenv("OPENAI_API_KEY"),
):
    client = OpenAI()
    try:
        # Generate SWOT analysis using OpenAI
        prompt = f"Create 4 lines SWOT analysis for the company based on the following website: {website}"

        completion = client.chat.completions.create(
            model="o1-mini",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        generated_text = completion.choices[0].message.content

        response_audio = client.audio.speech.create(
            model="tts-1", voice="alloy", input=generated_text
        )
        # response_audio.stream_to_file(speech_file_path)
        return response_audio

    except Exception as e:
        print(f"Error: {e}")
        return "Failed to generate SWOT analysis or convert to speech due to an error."
