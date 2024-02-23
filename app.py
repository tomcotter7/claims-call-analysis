import json
import logging
import os
import pandas as pd
import streamlit as st

from openai import OpenAI
from pydub import AudioSegment

from models import build_response_model

upload_path = "./uploads/"
download_path = "./downloads/"

logging.basicConfig(level=logging.INFO)

@st.cache_data
def process_mp3_file(audio_file, upload_path, download_path, output_audio_file):
    audio_data = AudioSegment.from_mp3(os.path.join(upload_path,audio_file.name))
    audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags={"comments": "Converted with pydub"})
    logging.info(f"File {audio_file.name} converted to {output_audio_file}")
    return output_audio_file

@st.cache_data
def process_audio(audio_file: str) -> str:
    client = OpenAI()
    audio = open(audio_file, 'rb')
    transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
    )
    logging.info(f"Transcript: {transcript.text}")
    segments = transcript.segments # type: ignore
    data = ""
    for segment in segments:
        data += f"Timestamp: {round(float(segment['start']), 1)}s - {round(float(segment['end']), 2)}\n{segment['text']}\n"

    return data


@st.cache_data
def process_csv_file(csv_file):
    questions = pd.read_csv(csv_file)
    logging.info(f"Questions: {questions}")
    return questions

@st.cache_data
def process_results(transcript, questions):
    response_model = build_response_model(questions)
    client = OpenAI()
    resp = client.chat.completions.create(
            model='gpt-4',
            messages = [
                {"role": "system", "content": "You are a claims adjuster. You have a list of questions that need to be filled out. Use the provided transcript to answer the questions. Return the answers as a JSON object. For each answer, include the timestamp(s) from which the answer was obtained"},
                {"role": "user", "content": transcript}
            ],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "Response",
                        "description": "Requested answers to the questions",
                        "parameters": response_model.model_json_schema()
                    }
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": "Response"}
            }
    )
    logging.info(f"Results: {resp}")
    args = resp.choices[0].message.tool_calls[0].function.arguments # type: ignore
    obj = json.loads(args)
    cleaned_obj = {k.replace('_', ' '): v for k, v in obj.items()}
    return cleaned_obj
 
mp3_file = st.file_uploader('Upload a recording of the claims call', type='mp3', help='Upload a recording of the claims call')
if mp3_file is not None:
    with open(os.path.join(upload_path, mp3_file.name), "wb") as f:
        f.write((mp3_file).getbuffer())
    output_audio_file = process_mp3_file(mp3_file, upload_path, download_path, 'output.mp3')
    audio_file = open(os.path.join(download_path, output_audio_file), 'rb')
    audio_bytes = audio_file.read()

    st.markdown("---")
    st.markdown("**Original Audio**")
    st.audio(audio_bytes)
    
    transcript = process_audio(str(os.path.abspath(os.path.join(download_path,output_audio_file))))

    csv_file = st.file_uploader('Upload the questions CSV file', type='csv', help='Upload the questions CSV file')
    if csv_file is not None:
        questions = process_csv_file(csv_file)
        if st.toggle('Show Questions'):
            st.markdown("**Your Questions**")
            st.write(questions)

        st.markdown("---")
        results = process_results(transcript, questions)
        if st.toggle('Show Results'):
            st.markdown("**Results**")
            for k, v in results.items():
                st.markdown(f"**{k}:**")
                st.write(f"{v['answer']}")
                st.markdown(f"**Timestamp:** {v['timestamp']}s")
