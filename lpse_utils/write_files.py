#!/bin/python3

# Produce light beamlets file
def light_input(light_type,beams,specs):
      
  # Setup options
  keywords = ['intensity','phase','polarization',\
              'frequencyShift','group','direction',\
              'evolution.source','evolution.offset',\
              'evolution.width','evolution.sgOrder']

  # Write file
  f = open('lpse.parms', "a")
  f.write(f'{light_type}.nBeams = {str(beams)};\n')
  for i in range(beams):
    for j in range(len(specs)):
      f.write(f'{light_type}.{str(i+1)}.{keywords[j]} = {str(specs[j][i])};\n')
  f.close()
    
# Produces standard #include file
def std_input(kwords,specs):
  f = open('lpse.parms', "a")
  for i in range(len(kwords)):
    f.write(f'{kwords[i]} = {str(specs[i])};\n')
  f.close()
