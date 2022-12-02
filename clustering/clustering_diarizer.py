import sys

import librosa
import torch
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from typing import List

import whisper
import matplotlib.pyplot as plt
import librosa.display
from pyannote.core import Annotation, Segment

HOP_LENGTH = 512
SAMPLE_RATE = 22050
IS_COLAB = "google.colab" in sys.modules

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


def get_annotation(segments: List[AudioSegment], speaker_labels: List[int], file_id: str):
    hypothesis = Annotation()
    for segment, label in zip(segments, speaker_labels):
        hypothesis[Segment(segment.start_time, segment.end_time)] = str(label)
    return hypothesis

def process_rttm(rttm_file):
    speaker_num = 0
    speakers = {}
    hypothesis = Annotation()
    rttm_lines = []

    with open(rttm_file, 'r') as rttm:
        for line in rttm:
            rttm_lines.append(line)
            parts = line.split()
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            speaker = parts[7]
            if speaker not in speakers:
                speakers[speaker] = speaker_num
                speaker_num += 1
            speaker_id = speakers[speaker]
            hypothesis[Segment(start_time, end_time)] = speaker_id
    print("hypothesis.to_rttm:\n" + hypothesis.to_rttm())
    return hypothesis, speaker_num

class ClusteringDiarizer():
    def __init__(self, verbose=False):
        self.whisper_model = whisper.load_model("small")
        self.verbose = verbose

    def get_encoder_blocks(self, audio_file: str):
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
        mel = mel[None, :, :]
        self.whisper_model.embed_audio(mel)  # self.encoder.forward(mel)
        print("encoder states:", self.whisper_model.encoder.encoder_states)

    def get_audio_segments(self, audio_file: str, prompt: str = ""):
        """
        Returns list of AudioSegments obtained by transcribing the given audio using a Whisper model.
        Each audio segment consists of the log-Mel spectrogram segment corresponding to the start & end times
        """
        segment_list = []
        if prompt:
            transcribe_result = whisper.transcribe(self.whisper_model, audio_file, prompt=prompt)
        else:
            transcribe_result = whisper.transcribe(self.whisper_model, audio_file)
        print("finished transcribing using Whisper model!")
        mel = transcribe_result['mel']
        for segment in transcribe_result['segments']:
            start_idx = time_to_idx(segment['start'])
            end_idx = time_to_idx(segment['end'])
            audio_segment = AudioSegment(mel[:, start_idx:end_idx], segment['start'], segment['end'], segment['text'])
            segment_list.append(audio_segment)
            if self.verbose:
                print(f"[{audio_segment.start_time}-{audio_segment.end_time}] {audio_segment.text}")
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
        if (self.verbose):
          print(f'audio length: {audio_length}')
          print(f'MFCC shape: {mfcc.shape}')
          print(f'intervals_s shape: {intervals_s.shape}')
          print(f'First 5 intervals: {intervals_s[:5]}')
          print(f'Last 5 intervals: {intervals_s[len(intervals_s) - 5:]}')
        return (mfcc, intervals_s)

    def predict_kmeans_meeting(self, meeting_name:str):
        metric = DiarizationErrorRate()
        # replace w/ folder u save the files to
        path_prefix = '/Users/melaniezhang/Desktop/for_andrew'
        rttm_file = f'{path_prefix}/{meeting_name}.rttm'
        audio_path = f'{path_prefix}/{meeting_name}.wav'
        transcription_path = f'{path_prefix}/{meeting_name}.txt'
        with open(transcription_path, 'r') as transcription:
          print("ground truth transcription:\n", transcription.read())
        reference, n_speakers = process_rttm(rttm_file)
        print(f"numspeakers : {n_speakers}")
        hypothesis = self.predict_kmeans(audio_path, num_speakers=n_speakers)
        print("DER = {0:.3f}".format(metric(reference, hypothesis)))

    def predict_kmeans(self, audio_file: str, num_speakers=2, method="AVERAGE_POOL"):
      """
      Given an input audio file and number of speakers, predict a diarization
      using the requested `method`, using K-means clustering.
      Returns a pyannote.core.Annotation

      Implemented `method`s:
      AVERAGE_POOL: use Whisper model to segment the audio into speaker chunks.
      For each segment, extract MFCCs and average them. Perform k-means clustering
      on the segment MFCC averages.
      """
      allowed_methods = ["AVERAGE_POOL"]
      if method not in allowed_methods:
        raise NotImplementedError(f"No implementation for method {method}")
      segments = self.get_audio_segments(audio_file)
      segment_features = []
      for segment in segments:
          duration = segment.end_time - segment.start_time
          y, sr = librosa.load(audio_file, offset=segment.start_time, duration=duration, sr=SAMPLE_RATE)
          mfcc = librosa.feature.mfcc(y=y, n_mfcc=13, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
          if method == "AVERAGE_POOL":
            segment_feature = np.mean(mfcc, axis=1)
          # TODO(Melanie): try out variance / min / max pooling
          segment_features.append(segment_feature)
      kmeans = KMeans(n_clusters=num_speakers, random_state=42)
      kmeans.fit(segment_features)
      segments_with_labels = zip(segments, kmeans.labels_)
      if (self.verbose):
        for segment_label in segments_with_labels:
            segment, label = segment_label
            print(f"[{segment.start_time}-{segment.end_time}] Speaker {label}: {segment.text}")
      output_annotation = get_annotation(segments, kmeans.labels_, audio_file)
      if (self.verbose):
        print(output_annotation.to_rttm())
      return output_annotation

    def predict_spectral(self, audio_file: str, num_speakers=2):
        """
        TODO(Andrew): fill out. Ouput should be a pyannote.core.Annotation object, like in the kmeans predictor.
        """
        pass

if __name__ == "__main__":
  audio_file = "../tests/melzh/hailey-bieber-interview.mp3"
  if IS_COLAB:
      audio_file = '/content/drive/MyDrive/cs229-final-project/whisper-diarization/tests/melzh/hailey-bieber-interview.mp3'
  diarizer = ClusteringDiarizer(verbose=True)
  # diarizer.predict_kmeans(audio_file)
  print(diarizer.predict_kmeans_meeting("IS1009a-0"))
