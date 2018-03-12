import os
import lmdb
import librosa

rootdir = '/home/yang/speech/train/train/'
wav_files = []
for (dirpath, dirnames, filenames) in os.walk(wav_path):
     for filename in filenames:
	  if filename.endswith('.wav') or filename.endswith('.WAV'):
		filename_path = os.sep.join([dirpath, filename])
                print(filename_path)
