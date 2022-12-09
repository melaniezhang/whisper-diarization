# CS 229 Final Project
Fall 2022

This repository is a fork of `openai/whisper` with some added tools and scripts related to speaker diarization.

Files authored by us: `clustering/clustering_diarizer.py`, `data_processing/process_ami_annotations.py`

The code we wrote to perform all dataset pre-processing and run our clustering experiments is in [this Colab notebook](https://colab.research.google.com/drive/1SVXOpi4TWWk20AYfKyhT1BGNEV6cRjXx?usp=sharing).

The code we used to fine-tune the Whisper model (adapted from [this fine-tuning notebook](https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz) and edited heavily) is in [this Colab notebook](https://colab.research.google.com/drive/1ubsscw7O0c0qnZzXgQZLFjKm87vgRbWB?usp=sharing).

###Setup Instructions
1. `sudo apt update && sudo apt install ffmpeg` (dependency for the Whisper package)
2. Clone this repo: `git clone https://github.com/melaniezhang/whisper-diarization.git`
4. `pip install -r requirements.txt`
5. `pip install librosa`
6. `python clustering/clustering_diarizer.py`