import librosa
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import whisper
import matplotlib.pyplot as plt
import librosa.display

HOP_LENGTH = 512
SAMPLE_RATE = 22050

class AudioSegment():
    def __init__(self, mel: torch.Tensor, start_time: float, end_time: float, text: str):
        self.mel = mel
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

def time_to_idx(time: float):
    """
    Converts from timestamp to an index in a mel spectrogram array
    """
    return int(time * whisper.audio.SAMPLE_RATE / whisper.audio.HOP_LENGTH)

def plot_mfccs(audio_file: str):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    print(mfcc.shape)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')
    plt.show()

def get_mfccs(audio_file: str):
    """
    Returns tuple (mfccs, intervals_s)
    where mfccs is a matrix of shape (n_mfcc, n_frames)
    and intervals is a list of length n_frames (mapping mfccs[i] to its start second)
    """
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, n_mfcc=13, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    audio_length = len(y) / sr  # in seconds
    step = HOP_LENGTH / sr  # in seconds
    intervals_s = np.arange(start=0, stop=audio_length, step=step)

    print(f'audio length: {audio_length}')
    print(f'MFCC shape: {mfcc.shape}')
    print(f'intervals_s shape: {intervals_s.shape}')
    print(f'First 5 intervals: {intervals_s[:5]} second')
    print(f'Last 5 intervals: {intervals_s[len(intervals_s) - 5:]}')
    return (mfcc, intervals_s)

class KMeansDiarizer():
    def __init__(self):
        self.whisper_model = whisper.load_model("medium")

    def get_audio_segments(self, audio_file: str):
        """
        Returns list of AudioSegments obtained by transcribing the given audio using a Whisper model.
        Each audio segment consists of the log-Mel spectrogram segment corresponding to the start & end times
        """
        segment_list = []
        transcribe_result = whisper.transcribe(self.whisper_model, audio_file)
        mel = transcribe_result['mel']
        for segment in transcribe_result['segments']:
            start_idx = time_to_idx(segment['start'])
            end_idx = time_to_idx(segment['end'])
            audio_segment = AudioSegment(mel[:, start_idx:end_idx], segment['start'], segment['end'], segment['text'])
            segment_list.append(audio_segment)
        return segment_list

    def get_mfccs(self, audio_file: str):
        """
        Returns tuple (mfccs, intervals_s)
        where mfccs is a tensor of shape (n_mfcc, n_frames)
        and intervals is a list of length n_frames (mapping mfccs[i] to its start second)
        """
        y, sr = librosa.load(audio_file)

        hop_length = 512  # number of samples between successive frames
        mfcc = librosa.feature.mfcc(y=y, n_mfcc=13, sr=sr, hop_length=hop_length)

        audio_length = len(y) / sr  # in seconds
        step = hop_length / sr  # in seconds
        intervals_s = np.arange(start=0, stop=audio_length, step=step)

        print(f'audio length: {audio_length}')
        print(f'MFCC shape: {mfcc.shape}')
        print(f'intervals_s shape: {intervals_s.shape}')
        print(f'First 5 intervals: {intervals_s[:5]}')
        print(f'Last 5 intervals: {intervals_s[len(intervals_s) - 5:]}')
        return (mfcc, intervals_s)


audio_file = "../tests/melzh/hailey-bieber-interview.mp3"
diarizer = KMeansDiarizer()
# use whisper model to produced segments (start, end, text) from the audio file
segments = diarizer.get_audio_segments(audio_file)
# segment_indices = [segment.start_time for segment in segments]
# print("sample indices: ", librosa.time_to_samples(segment_indices, sr=SAMPLE_RATE))

# TRY #1: get mfccs for the entire audio file, and perform k-means clustering on each individual mfcc
# get mfcc features
mfccs, buckets_s = get_mfccs(audio_file)
kmeans = KMeans(n_clusters=3, random_state=42)
# perform k means clustering on these features
kmeans.fit(mfccs.T)

# loop through the segments and record all labels that were predicted for each mfcc in the segment
mfcc_idx = 0
segment_labels = []
for segment in segments:
    segment_labels.append([])
    end_time = segment.end_time
    while mfcc_idx < len(buckets_s) and buckets_s[mfcc_idx] < end_time:
        segment_labels[-1].append(kmeans.labels_[mfcc_idx])
        mfcc_idx += 1
print(segment_labels)

# TRY #2:
# get mfccs for each audio segment. average them, and perform k-means clustering on the averages
segment_mfcc_averages = []
for segment in segments:
    duration = segment.end_time - segment.start_time
    y, sr = librosa.load(audio_file, offset=segment.start_time, duration=duration, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=13, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    mfcc_avg = np.mean(mfcc, axis=1)
    segment_mfcc_averages.append(mfcc_avg)
kmeans = KMeans(n_clusters=2, random_state=42)
# perform k means clustering on these features
kmeans.fit(segment_mfcc_averages)
segments_with_labels = zip(segments, kmeans.labels_)
for segment_label in segments_with_labels:
    segment, label = segment_label
    print(f"[{segment.start_time}-{segment.end_time}] Speaker {label}: {segment.text}")
