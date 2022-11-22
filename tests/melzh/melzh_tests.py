import whisper

model = whisper.load_model("base")

audio = whisper.load_audio("hailey-bieber-interview.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
print(mel.shape)
#
# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")
#
# # decode the audio
# # options = whisper.DecodingOptions(fp16 = False)
# # result = whisper.decode(model, mel, options)
#
# # try transcribe?
# transcribe_result = whisper.transcribe(model, "hailey-bieber-interview.mp3")
# for segment in transcribe_result["segments"]:
#     print(segment)
#
#
# # print the recognized text
# print(transcribe_result)