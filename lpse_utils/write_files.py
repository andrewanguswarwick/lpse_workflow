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

# Extracts keywords and values from class object for writing to file
def Package(obj,kwords=None,specs=None,prefix=None):
    dic = obj.__dict__
    kys = list(dic.keys())
    if kwords is None:
        kwords = []
    if specs is None:
        specs = []
    for i in kys[:-1]:
        att = dic[i]
        if (att == None):
            continue
        elif (isinstance(att,obj.iclasses)):
            Package(att,kwords,specs,i+'.')
        else:
            if prefix is None:
                kwords.append(i)
            else:
                kwords.append(prefix+i)
            specs.append(att)
    return kwords, specs

# Job control class
class job_control:
    # Constructor
    def __init__(self):
        # Attributes with default values
        self.version = None # LPSE Version
        self.title = None # Title for run
        self.verbose = 2 # amount of output, value in range [-1,5]
        self.seed = 0 # 0 for random, any other integer for repeatable results
        
        # Inner classes
        self.resources = self.Resources()
        
        # Create list of inner class types for checking against
        # Ensure this goes last
        self.iclasses = (self.Resources)
        

    # Missing additional job control inner classes for threads, locks, error trapping, and debugging aids
    class Resources:
        def __init__(self):
            # Attributes with default values
            self.heartbeatInterval = 0 # minutes to output percentage completion, 0 disables
            
            # Inner class list
            self.iclasses = ()
    
    # Methods
    def write(self):        
        kwords, specs = Package(self)
        std_input(kwords,specs)
        print('File written.')
