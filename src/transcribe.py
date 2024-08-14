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

# 入力CSVの読み込み
df_input: pd.DataFrame = pd.read_csv(input_csv)

# 出力CSVが存在する場合は読み込む
if os.path.exists(output_csv):
    df_output: pd.DataFrame = pd.read_csv(output_csv)
else:
    df_output = pd.DataFrame(columns=df_input.columns)

# input_csvにあってoutput_csvにないレコードを抽出
new_records = df_input[~df_input['id'].isin(df_output['id'])]


# 新しいレコードを出力CSVに追加
df_output = pd.concat([df_output, new_records], ignore_index=True)
# 追加したレコードだけを再度取得
new_records_added = df_output[df_output['id'].isin(new_records['id'])]

audio_files: list[str] = os.listdir(input_folder)

# 新たに追加されたレコードだけを処理
for index, row in new_records_added.iterrows():
    if pd.isna(row['filename']):
        continue
    
    filename: str = row['filename']
    basename: str = os.path.splitext(filename)[0]

    if basename + '.m4a' in audio_files:
        input_path: str = os.path.join(input_folder, basename + '.m4a')
        transcription: str = transcribe_audio(input_path)
        df_output.at[index, 'text'] = transcription
        
        print(f'Transcribed {input_path}')

    df_output.to_csv(output_csv, index=False)