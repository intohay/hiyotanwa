import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import io
load_dotenv()

client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


input_folder: str = 'data/audio'
input_csv: str = 'data/hiyoritalk.csv'
output_csv: str = 'data/hiyoritalk_transcribed.csv'




def transcribe_audio(file_path: str) -> str:
    
    audio_file: io.BufferedReader = open(file_path, "rb")

    transcription: str = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )

    return transcription.text



df: pd.DataFrame = pd.read_csv(input_csv)

audio_files: list[str] = os.listdir(input_folder)

for index, row in df.iterrows():
    if pd.isna(row['filename']):
        continue
    
    if not pd.isna(row['text']):
        continue


    filename: str = row['filename']
    basename: str = os.path.splitext(filename)[0]

    if basename + '.m4a' in audio_files:

        input_path: str = os.path.join(input_folder, basename + '.m4a')

        transcription: str = transcribe_audio(input_path)

        df.at[index, 'text'] = transcription

        print(f'Transcribed {input_path}')
    
    # save everytime
    df.to_csv(output_csv, index=False)



