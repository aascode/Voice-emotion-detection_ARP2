import os

#output file path
os.chdir("C:/Users/totti/Downloads/Audio_Speech_Actors_01-24")
print(os.getcwd())

#audio file path
audio_file_dir = "C:/Users/totti/Downloads/Audio_Speech_Actors_01-24"

emotion_dict = {
    "01" : "neutral",
    "02" : "calm",
    "03" : "happy",
    "04" : "sad",
    "05" : "angry",
    "06" : "fearful",
    "07" : "disgust",
    "08" : "surprised"
}

def get_emotion(number):
    return emotion_dict.get(number)

emotions = []
filenames = []
for subdir, dirs, files in os.walk(audio_file_dir):
    for file in files:
        if '.wav' in file:
            emotion_number = file[6:8]
            emotion = get_emotion(emotion_number)
            filenames.append(file)
            emotions.append(emotion)

import csv
import itertools

#prepare data to write to csv file
writestream = []
writestream.append("filenames;")
writestream.append("emotions\n")
for index, val in enumerate(filenames):
    writestream.append(filenames[index])
    writestream.append(';')
    writestream.append(emotions[index])
    writestream.append('\n')

#output file
with open("annotation.csv", "w") as f:
    for string in writestream:
        f.write(string)

