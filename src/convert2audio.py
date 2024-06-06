import os
from pydub import AudioSegment
import shutil
import subprocess
# ffmpegがインストールされているディレクトリのパスを設定します
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"



# MP4ファイルが含まれるフォルダのパス
input_folder: str = 'data/media'
# 変換後のM4Aファイルを保存するフォルダのパス
output_folder: str = 'data/audio'

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def has_audio_stream(file_path: str) -> bool:

    command = [
        "ffmpeg", "-i", file_path, "-map", "0:a:0", "-c", "copy", "-f", "null", "-"
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    return result.returncode == 0


input_files: set[str] = {os.path.splitext(f)[0] for f in os.listdir(input_folder) if f.endswith('.mp4') or f.endswith('.m4a')} 
output_files: set[str] = {os.path.splitext(f)[0] for f in os.listdir(output_folder)}

new_files: set[str] = input_files - output_files

# フォルダ内のファイルをループ処理
for filename in os.listdir(input_folder):
    print(filename)
    base_name: str
    ext: str
    base_name, ext = os.path.splitext(filename)
    
    if base_name in new_files:
        input_path: str = os.path.join(input_folder, filename)

        if ext == ".mp4":
            if has_audio_stream(input_path):
                try: 
                    output_filename: str = os.path.splitext(filename)[0] + '.m4a'
                    output_path: str = os.path.join(output_folder, output_filename)
                    
                    # 音声ファイルの読み込み
                    audio: AudioSegment = AudioSegment.from_file(input_path, format='mp4')
                    # 音声ファイルの保存
                    audio.export(output_path, format='ipod')

                    print(f'Converted {input_path} to {output_path}')
                except Exception as e:
                    print(f'Error: {e}')
            else:
                print(f'No audio stream found in {input_path}')
        
    
        elif ext == ".m4a":
            # output_folderへコピー
            input_path: str = os.path.join(input_folder, filename)
            output_path: str = os.path.join(output_folder, filename)
            shutil.copy(input_path, output_path)
            print(f'Copied {input_path} to {output_path}')

