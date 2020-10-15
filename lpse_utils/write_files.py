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
def package(obj,kwords=None,specs=None,prefix=None):
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
            prefix = str(prefix or '') + i + '.'
            package(att,kwords,specs,prefix)
        else:
            kwords.append(str(prefix or '')+i)
            specs.append(att)
    return kwords, specs

# Uses extracted keywords and values to update lpse.parms
def std_write(self):        
    kwords, specs = package(self)
    std_input(kwords,specs)
    print('Attributes written:')
    adict = {i:j for i,j in zip(kwords,specs)}
    print(adict)

# Job control class
class job_control:
    # Constructor
    def __init__(self):
        # Attributes with default values
        self.version = None # LPSE Version
        self.title = None # Title for run
        self.verbose = 2 # amount of output, value in range [-1,5]
        self.seed = 0 # 0 for random, any other integer for fixed seed
        
        # Inner classes
        self.resources = self.Resources()
        
        # Create list of inner class types for checking against
        # Ensure this goes last
        self.iclasses = (self.Resources)

    # Missing additional inner classes for threads, locks, error trapping, and debugging aids
    class Resources:
        def __init__(self):
            # Attributes with default values
            self.heartbeatInterval = 0 # minutes to output percentage completion, 0 disables
            
            # Inner class list
            self.iclasses = ()
    
    # Methods
    write = std_write

# Gridding class
class gridding:
    def __init__(self):
        # Inner classes
        self.grid = self.Grid()
        self.iclasses = (self.Grid)

    # Missing additional inner class for mpi nodes
    class Grid:
        def __init__(self):
            # Attributes
            self.sizes = None # Grid size in microns
            self.nodes = None # Number of grid nodes
            
            # Inner classes
            self.antiAliasing = self.AntiAliasing()
            self.iclasses = (self.AntiAliasing)

        class AntiAliasing:
            def __init__(self):
                # Attributes
                self.range = 0.0 # Fractional range of ignored wavenumbers for anti-aliasing
                self.isAutomatic = 'true' # Whether to automatically calculate
                
                # Inner classes
                self.iclasses = ()
    
    # Methods
    write = std_write
