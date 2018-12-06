from os import system, listdir
from os.path import isfile, join

path = 'C:/Users/vince_000/Documents/openSMILE/openSMILE-2.1.0/bin/Win32/wav_files/'

files = [f for f in listdir(path) if isfile(join(path, f))]

for f in files:
  print(f)  
  cmd = 'SMILExtract_Release -C ../../config/IS09_emotion.conf -I ' + path + f + ' instName ' + f
  os.system(cmd)
  
