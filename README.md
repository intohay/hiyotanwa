# hiyotanwa
## Data preparation
You need to prepare the following data to run the programs.
- `data/`: Directory containing the data used in the programs.
    - `data/hiyoritalk.csv`: Data file containing the data used in the programs.
    - `data/media`: Directory containing the media files used in the programs.


The format of the hiyoritalk.csv file is as follows.
- `id`: ID of the data
- `name`: Name of the member
- `date`: Date of the data
- `text`: Text data
- `filename`: File name of the data


## Analysis
### Installation

You need to install the required libraries before running the programs.

```bash
pip install -r requirements.txt
```


### Illeism analysis
Illeism analysis is performed using the `illeism_analysis.py` script.

Before running the script, you need to convert the video data to audio data. You can convert the video data to audio data using the following command.

```bash
python convert2audio.py
```

And you need to transcribe the audio data. 
After setting the OpenAI API key in the form of `OPENAI_API_KEY` in the `.env` file, you can transcribe the audio data using the following command


```bash
python transcribe.py
```

Next, you need to check the transcription results using with the following command.

```bash
python check_transcription.py
```

After checking the transcription results, you can analyze the illeism using the following command.

```bash
python illeism_analysis.py
```

The results of the analysis are shown on the display like this.

![hiyotanwa_2004-2405](https://github.com/root2116/hiyotanwa/assets/63008759/bf35d874-30d9-49a4-9c2b-e56769c93753)



### Yaho analysis
Yaho analysis is performed using the `yaho_analysis.py` script.

Before running the script, you are required to perform sentiment analysis on the whole data using the following command.

```bash
python sentiment_analysis.py
```

After performing sentiment analysis, you can analyze the yaho using the following command.

```bash
python yaho_analysis.py
```

The results of the analysis are saved in the working directory as `scatterplot.png` and `boxplot.png` like below.


![scatterplot](https://github.com/root2116/hiyotanwa/assets/63008759/e781ddba-6c60-4663-9e89-4c4880816345)
![boxplot](https://github.com/root2116/hiyotanwa/assets/63008759/5b68da56-e681-4d62-876c-5a357bbf6702)

