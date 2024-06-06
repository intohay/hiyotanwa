from pydub import AudioSegment
import pygame
import pandas as pd
import os
from prompt_toolkit import prompt
from prompt_toolkit.document import Document
from prompt_toolkit.shortcuts import PromptSession
import textwrap
pygame.init()


input_csv: str = "data/hiyoritalk_transcribed.csv"
output_csv: str = "data/hiyoritalk_transcribed_checked.csv"

if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
else:
    df = pd.read_csv(input_csv)

def wrap_text(text: str, width: int = 80) -> str:
    return "\n".join(textwrap.wrap(text, width))


# add checked column
if "checked" not in df.columns:
    df["checked"] = False


session = PromptSession()


for index, row in df.iterrows():
    if pd.isna(row["text"]):
        continue
    
    if pd.isna(row["filename"]):
        continue

    if row["filename"].endswith(".jpg"):
        continue

    if row["checked"]:
        continue

    
    basename: str = os.path.splitext(row["filename"])[0]
    filename: str = basename + ".m4a"

    print(f"Checking {filename}...")
    
    text: str = row["text"]
    print(text)

    

    audio = AudioSegment.from_file(f"data/audio/{filename}", format="m4a")
    temp_filename = "temp.wav"
    
    audio.export(temp_filename, format="wav")
    pygame.mixer.music.load(temp_filename)
    
    while True:
        
        pygame.mixer.music.play()
        print("Playing audio...")

        print("Is the transcription correct? (y/n/retry)")
        answer: str = input()
        
        if answer == "y":
            df.at[index, "checked"] = True
            break
        elif answer == "n":
            print("Please input correct transcription (press Ctrl-D when done):")
            wrapped_text = wrap_text(text, width=80)

            # 複数行入力を可能にするプロンプト
            corrected_text: str = session.prompt(
                message="> ",
                default=wrapped_text,
            )

            
            
            df.at[index, "text"] = corrected_text.replace('\n', ' ')

            df.at[index, "checked"] = True
            break
        elif answer == "retry":
            continue

    df.to_csv(output_csv, index=False)
    print("Saved.")

    print("Continue? (y/n)")
    answer:str  = input()
    if answer == "n":
        break

    print()
    print("Next...")
