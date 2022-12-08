import collections
import re
import os

WORDS_DIRECTORY = "/Users/melaniezhang/Downloads/ami_public_manual_1.6.2/words/"
MEETINGS_LIST_FILE = "/Users/melaniezhang/whisper-diarization/data_processing/files.txt"
TRANSCRIPTIONS_OUT_DIR = "/Users/melaniezhang/whisper-diarization/data/ami/transcriptions/"

def produce_transcription(file_list, speakers, out_file):
	# given a list of words files (AMI meeting annotations) and a list of speaker names, parse the words files
	# for each speaker and combine the output into one text file.
	class SpeechSegment:
		def __init__(self, speaker="", starttime=-float('inf'), endtime=-float('inf'), text=''):
			self.speaker = speaker
			self.starttime = starttime
			self.endtime = endtime
			self.text = text

		def __str__(self):
			return f"Speaker: {self.speaker} starttime: {self.starttime} endtime: {self.endtime} text: {self.text}"

	word_times = {}
	for n in range(len(file_list)):
		file = open(file_list[n], 'r')
		lines = list(file.readlines())
		i = 0
		prev_timestamp = -float('inf')
		while i < len(lines):
			line = lines[i].strip()
			line_parts = line.split()
			if len(line_parts) > 0 and line.startswith("<w"):
				pattern = r'"([A-Za-z0-9_\./\\-]*)"'
				elements = list(re.findall(pattern, line))
				text_pattern = r'>(.*)<'
				text = list(re.findall(text_pattern, line))
				if len(text) != 1:
					raise RuntimeError("there should only be one word per line.")
				text = text[0]
				# For some reason the apostrophes all show up as hex symbols. I couldn't figure out what encoding to
				# use for them to render normally.
				if '&' in text:
					if ('&#39;' not in text):
						raise RuntimeError("Weird symbol found")
					# replace hex stuff with '
					text = text.replace('&#39;', '\'')
				if len(elements) >= 3:
					if 'punc' not in line:
						endtime = float(elements[1])
						starttime = float(elements[2])
						if starttime > endtime:
							starttime, endtime = endtime, starttime
						segment = SpeechSegment(speakers[n], starttime, endtime, text)
						word_times[starttime] = segment
						prev_timestamp = starttime
					elif 'punc' in line:
						if prev_timestamp < 0:
							print(f"Saw punctuation but never any text :( line: {line}")
						else:
							# punctuation!
							word_times[prev_timestamp].text += text
			i+=1
		file.close()

	word_times_sorted = sorted(word_times.keys())
	segment_list = []

	current_segment = word_times[word_times_sorted[0]]
	print(f"last timestamp: {word_times_sorted[-1]}")

	for key in word_times_sorted[1:]:
		if word_times[key].speaker == current_segment.speaker:
			current_segment.text += " " + word_times[key].text
			current_segment.endtime = max(word_times[key].endtime, current_segment.endtime)
		else:
			segment_list.append(current_segment)
			current_segment = word_times[key]

	def stringify_for_transcription(segment: SpeechSegment):
		return f'{segment.starttime}|{segment.endtime}| Speaker {segment.speaker}: {segment.text}\n'

	file = open(out_file,"w")
	file.writelines([stringify_for_transcription(line) for line in segment_list])
	file.close()
	print(f"Transcription done! Written to file {out_file}")

# process all words annotations in the AMI zip folder
filenames_map = collections.defaultdict(list)
for filename in os.listdir(WORDS_DIRECTORY):
	parts = filename.split(".")
	full_path = WORDS_DIRECTORY + filename
	filenames_map[parts[0]].append((filename, parts[1], full_path))  # appending (filename, speaker)
meeting_list = list(open(MEETINGS_LIST_FILE, 'r').readlines())
for meeting in meeting_list:
	meeting = meeting.strip()
	if not filenames_map[meeting]:
		raise RuntimeError("no speaker files found for meeting" + meeting)
	file_list = []
	speaker_list = []
	for file, speaker, full_path in filenames_map[meeting.strip()]:
		file_list.append(full_path)
		speaker_list.append(speaker)
	produce_transcription(file_list, speaker_list, f"{TRANSCRIPTIONS_OUT_DIR}/{meeting}.txt")

