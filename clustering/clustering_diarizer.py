import sys
from typing import List, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyannote.audio import Model, Inference
from pyannote.audio.core.io import Audio
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.cluster import KMeans

import whisper

HOP_LENGTH = 512
SAMPLE_RATE = 22050
IS_COLAB = "google.colab" in sys.modules


class AudioSegment():
    def __init__(self, mel: Union[torch.Tensor, None], start_time: float, end_time: float, text: str,
                 speaker: str = None):
        self.mel = mel
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.speaker = speaker


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


def get_annotation(segments: List[AudioSegment], speaker_labels: List[int]):
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
    return hypothesis, speaker_num


class ClusteringDiarizer():
    def __init__(self, verbose=True):
        self.whisper_model = whisper.load_model("small")
        self.verbose = verbose
        self.pyannote_model = Model.from_pretrained("pyannote/embedding",
                                                    use_auth_token="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")

    def get_encoder_blocks(self, audio_file: str):
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
        mel = mel[None, :, :]
        self.whisper_model.embed_audio(mel)  # self.encoder.forward(mel)
        print("encoder states:", self.whisper_model.encoder.encoder_states)

    def get_audio_segments_whisper(self, audio_file: str, prompt: str = ""):
        """
        Returns list of AudioSegments obtained by transcribing the given audio using a Whisper model.
        Each audio segment consists of the log-Mel spectrogram segment corresponding to the start & end times
        """
        segment_list = []
        if prompt:
            transcribe_result = whisper.transcribe(self.whisper_model, audio_file, prompt=prompt)
        else:
            transcribe_result = whisper.transcribe(self.whisper_model, audio_file)
        if self.verbose:
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

    def get_audio_segments_uniform(self, audio_file: str):
        """
        Divides audio file into uniform audio segments.
        """
        segment_list = []

        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

        # # remove silence
        # y = librosa.effects.split(y, top_db=30)
        # l = []
        # for i in y:
        #     l.append(y[i[0]:i[1]] )
        # y = np.concatenate(l,axis=0)

        # split into 1 second intervals
        segment_length_time = 1  # in seconds
        segment_length = sr * segment_length_time  # number of samples in segment

        num_segments = int(np.ceil(len(y) / segment_length))

        for i in range(num_segments):
            audio_segment = AudioSegment(mel=None, start_time=i, end_time=i + segment_length_time,
                                         text=None)  # need segment start and end time
            segment_list.append(audio_segment)
        return segment_list

    def get_audio_segments_uniform_remove_silence(self, audio_file: str):
        """
        Divides audio file into uniform audio segments, removing silence.
        """
        segment_list = []

        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

        # remove silence
        time_stamps = librosa.effects.split(y, top_db=23)

        segment_length_time_max = 1.5  # in seconds
        n_samples = sr * segment_length_time_max

        final_time_stamps = []
        count = 0
        for i in range(len(time_stamps)):
            if i == 0:
                final_time_stamps.append(time_stamps[i].tolist())
            else:
                if time_stamps[i, 0] - final_time_stamps[count][1] < n_samples:
                    final_time_stamps[count] = [final_time_stamps[count][0], time_stamps[i, 1]]
                else:
                    final_time_stamps.append(time_stamps[i].tolist())
                    count += 1
        time_stamps = final_time_stamps

        # split into uniform intervals
        segment_length_time = 1  # in seconds
        segment_length = sr * segment_length_time  # number of samples in segment

        for k in time_stamps:
            num_segments = int(np.ceil(len(y[k[0]:k[1]]) / segment_length))
            for i in range(num_segments):
                audio_segment = AudioSegment(mel=None, start_time=k[0] / sr + i * segment_length_time,
                                             end_time=k[0] / sr + i * segment_length_time + segment_length_time,
                                             text=None)  # need segment start and end time
                if self.verbose:
                    print(f"segment start: {audio_segment.start_time} segment end: {audio_segment.end_time}")
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
        if (self.verbose):
            print(f'audio length: {audio_length}')
            print(f'MFCC shape: {mfcc.shape}')
            print(f'intervals_s shape: {intervals_s.shape}')
            print(f'First 5 intervals: {intervals_s[:5]}')
            print(f'Last 5 intervals: {intervals_s[len(intervals_s) - 5:]}')
        return (mfcc, intervals_s)

    def predict_kmeans_meeting(self, meeting_name: str, dataset_path_prefix: str, methods: List[str], segmenting_strategy: str,
                               clustering_method: str):
        method_results = {}
        metric = DiarizationErrorRate()
        rttm_file = f'{dataset_path_prefix}/rttm/{meeting_name}.rttm'
        audio_path = f'{dataset_path_prefix}/audio/{meeting_name}.wav'
        transcription_path = f'{dataset_path_prefix}/transcriptions/{meeting_name}.txt'
        with open(transcription_path, 'r') as transcription:
            if self.verbose:
                print("ground truth transcription:\n", transcription.read())
        reference, n_speakers = process_rttm(rttm_file)
        if self.verbose:
            print(f"numspeakers : {n_speakers}")
        if segmenting_strategy == 'UNIFORM':
            segments = self.get_audio_segments_uniform(audio_path)
        elif segmenting_strategy == 'UNIFORM_REMOVE_SILENCE':
            segments = self.get_audio_segments_uniform_remove_silence(audio_path)
        elif segmenting_strategy == 'WHISPER':
            segments = self.get_audio_segments_whisper(audio_path)
        else:
            raise RuntimeError("invalid segmenting strategy")
        for method in methods:
            hypothesis = self.predict_kmeans(audio_path, segments, num_speakers=n_speakers, method=method,
                                             clustering_method=clustering_method)
            der = metric(reference, hypothesis)
            if self.verbose:
                print(f"metric(reference, hypothesis): {metric(reference, hypothesis)}")
                print(f"der: {der}")
                print(f"reference:\n{reference.to_rttm()}")
                print(f"hypothesis:\n{hypothesis.to_rttm()}")
            method_results[method] = (hypothesis, reference, der)
            if self.verbose:
                print(f"method {method} DER = {der}")
        return method_results

    def predict_kmeans(self, audio_file: str, segments: List[AudioSegment], num_speakers=2,
                       method="AVERAGE_POOL", clustering_method: str = 'KMEANS'):
        """
        Given an input audio file and number of speakers, predict a diarization
        using the requested `method`, using K-means clustering.
        Returns a pyannote.core.Annotation

        Implemented `method`s:
        AVERAGE_POOL: use Whisper model to segment the audio into speaker chunks.
        For each segment, extract MFCCs and average them. Perform k-means clustering
        on the segment MFCC averages.
        """
        #   allowed_methods = ["AVERAGE_POOL", "MIN_POOL", "MAX_POOL", "STD_POOL", "COMBO", "MAX_AVG"]
        #   if method not in allowed_methods:
        #     raise NotImplementedError(f"No implementation for method {method}")
        segment_features = []
        if method == "PYANNOTE":
            inference = Inference(self.pyannote_model, window="whole")
            pyannote_audio = Audio().validate_file(audio_file)
            audio_length = Audio().get_duration(pyannote_audio)
        for i, segment in enumerate(segments):
            duration = segment.end_time - segment.start_time
            y, sr = librosa.load(audio_file, offset=segment.start_time, duration=duration, sr=SAMPLE_RATE)
            mfcc = librosa.feature.mfcc(y=y, n_mfcc=13, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

            avg = np.mean(mfcc, axis=1)
            min_pool = np.min(mfcc, axis=1)
            max_pool = np.max(mfcc, axis=1)
            std = np.std(mfcc, axis=1)

            if method == "AVERAGE_POOL":
                segment_feature = avg
            elif method == "MIN_POOL":
                segment_feature = min_pool
            elif method == "MAX_POOL":
                segment_feature = max_pool
            elif method == "STD_POOL":
                segment_feature = std
            elif method == "AVG_MAX":
                segment_feature = np.concatenate((max_pool, avg), axis=0)
            elif method == "AVG_MIN":
                segment_feature = np.concatenate((min_pool, avg), axis=0)
            elif method == "AVG_STD":
                segment_feature = np.concatenate((std, avg), axis=0)
            elif method == "MIN_MAX":
                segment_feature = np.concatenate((min_pool, max_pool), axis=0)
            elif method == "MIN_STD":
                segment_feature = np.concatenate((min_pool, std), axis=0)
            elif method == "MAX_STD":
                segment_feature = np.concatenate((max_pool, std), axis=0)
            elif method == "AVG_MIN_MAX":
                segment_feature = np.concatenate((avg, min_pool, max_pool), axis=0)
            elif method == "AVG_MIN_STD":
                segment_feature = np.concatenate((avg, min_pool, std), axis=0)
            elif method == "AVG_MAX_STD":
                segment_feature = np.concatenate((avg, max_pool, std), axis=0)
            elif method == "MIN_MAX_STD":
                segment_feature = np.concatenate((min_pool, max_pool, std), axis=0)
            elif method == "COMBO":
                segment_feature = np.concatenate((avg, min_pool, max_pool, std), axis=0)
            elif method == "RANDOM":
                segment_feature = None
            elif method == "PYANNOTE":
                if segment.end_time > audio_length:
                    segment.end_time = audio_length
                segment_feature = inference.crop(pyannote_audio, Segment(segment.start_time, segment.end_time))
            else:
                raise RuntimeError(f"Unknown method: {method}")
            segment_features.append(segment_feature)
        if method == "RANDOM":
            labels = np.random.randint(0, num_speakers, len(segments))
        elif clustering_method == "KMEANS":
            kmeans = KMeans(n_clusters=num_speakers, random_state=14)
            kmeans.fit(segment_features)
            labels = kmeans.labels_
        elif clustering_method == 'SPECTRAL':
            from sklearn.cluster import SpectralClustering
            spectral = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0)
            spectral.fit(segment_features)
            labels = spectral.labels_
        else:
            raise RuntimeError("invalid clustering method")
        segments_with_labels = zip(segments, labels)
        #   if (self.verbose):
        #     for segment_label in segments_with_labels:
        #         segment, label = segment_label
        #         print(f"[{segment.start_time}-{segment.end_time}] Speaker {label}: {segment.text}")
        output_annotation = get_annotation(segments, labels)
        if (self.verbose):
            print(output_annotation.to_rttm())
        return output_annotation

def process_transcription(transcription_lines: List[str], verbose=False):
    # returns AudioSegment list, speaker_num, and pyannote.Annotation from the transcription lines
    annotation = Annotation()
    segments = []
    speakers = {}
    speaker_num = 0
    for line in transcription_lines:
        parts = line.split('|')
        start_time = float(parts[0])
        end_time = float(parts[1])
        # if (end_time - start_time) < 0.5:
        #     if verbose:
        #         print(f"skipping segment of length {start_time - end_time}")
        #     continue
        speaker_and_text = parts[2].split(':')
        speaker = speaker_and_text[0].split()[-1]
        if speaker not in speakers:
            speakers[speaker] = speaker_num
            speaker_num += 1
        speaker_id = speakers[speaker]
        text = speaker_and_text[1]
        annotation[Segment(start_time, end_time)] = str(speaker_id)
        segments.append(AudioSegment(None, start_time, end_time, text, str(speaker_id)))
    if len(segments) != len(annotation.get_timeline().segments_list_):
        raise RuntimeError("Number of segments are not equal :(")
    return segments, annotation, speaker_num

if __name__ == "__main__":
    # audio_file = "../tests/melzh/hailey-bieber-interview.mp3"
    # if IS_COLAB:
    #     audio_file = '/content/drive/MyDrive/cs229-final-project/whisper-diarization/tests/melzh/hailey-bieber-interview.mp3'
    # pyannote_diarizer = PyannoteEmbeddingDiarizer(verbose=True)
    # pyannote_diarizer.k_means_diarize_meeting("IS1009a-0", segmenting_strategy="UNIFORM",
    #                                           audio_path_prefix='/Users/melaniezhang/Desktop/for_andrew/audio/',
    #                                           transcription_path_prefix='/Users/melaniezhang/Desktop/for_andrew/transcriptions/',)
    # pyannote_diarizer.k_means_diarize_meeting("IS1008a-2", segmenting_strategy="UNIFORM",
    #                                           audio_path_prefix='/Users/melaniezhang/Desktop/for_andrew/audio/',
    #                                           transcription_path_prefix='/Users/melaniezhang/Desktop/for_andrew/transcriptions/')
    diarizer = ClusteringDiarizer(verbose=False)
    # from andrew
    allowed_methods = ["AVERAGE_POOL", "MIN_POOL", "MAX_POOL", "STD_POOL", "AVG_MAX", "AVG_MIN", "AVG_STD", "MIN_MAX",
                       "MIN_STD", "MAX_STD",
                       "AVG_MIN_MAX", "AVG_MIN_STD", "AVG_MAX_STD", "MIN_MAX_STD", "COMBO", "RANDOM", "PYANNOTE"]
    print("Begin Meeting Prediction")
    result = diarizer.predict_kmeans_meeting("IS1009a-0", methods=allowed_methods, dataset_path_prefix='/Users/melaniezhang/Desktop/for_andrew',
                                            segmenting_strategy='UNIFORM_REMOVE_SILENCE', clustering_method='KMEANS')
    for method in allowed_methods:
        print(f'{method}: {result[method][2]}')
    # model = whisper.load_model('base')
    # woptions = whisper.DecodingOptions(language="en", without_timestamps=False)  # change here
    # # wmodel = whisper.load_model("base")
    # wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task)
    # dec_input = [*wtokenizer.sot_sequence]
    #
    # def get_timestamp_token(time):
    #     return int(time / 0.02 + wtokenizer.timestamp_begin)
    #
    # dec_input.append(get_timestamp_token(0.0))
    # dec_input.extend(wtokenizer.encode("Hello!"))
    # dec_input.append(get_timestamp_token(1.0))
    # dec_input.append(get_timestamp_token(1.5))
    # dec_input.extend(wtokenizer.encode("Hey there."))
    # dec_input.append(get_timestamp_token(2.5))
    # print(dec_input)
    # print(wtokenizer.decode_with_timestamps(dec_input))

