import os
from os import system, listdir
from os.path import isfile, join

path = 'C:/Users/vince_000/Documents/openSMILE/openSMILE-2.1.0/bin/Win32/'

# For no layers

files = [f for f in listdir(path) if isfile(join(path, f))]
#
#for f in files:
#  print(f)  
#  cmd = 'SMILExtract_Release -C ../../config/IS09_emotion.conf -I ' + path + f + ' -instname ' + f
#  os.system(cmd)
  
import os# For sub directories

for root, dirs, files in os.walk("wav_files"):  
    for filename in files:
          subfilename = root.replace('\\','/') + '/' + filename 
          #print(dirs)
          #print(root)
          cmd = 'SMILExtract_Release -C ../../config/IS09_emotion.conf -I ' + path + subfilename + ' -instname ' + filename
          os.system(cmd)
          print(cmd)


