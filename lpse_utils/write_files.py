#!/bin/python3

# Appends lpse.parms with keywords and specifications
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
    # If attribute an inner class recursively call package
    elif (isinstance(att,obj.iclasses)):
      prefixnew = str(prefix or '') + i + '.'
      package(att,kwords,specs,prefixnew)
    else:
      kwords.append(str(prefix or '')+i)
      specs.append(att)
  return kwords, specs

# Uses extracted keywords and values to update lpse.parms
def std_write(self,pout=True):        
  kwords, specs = package(self)
  std_input(kwords,specs)
  if pout:
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
    # Ensure this goes last for correct package() function
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

# Basic enable class
class Enable:
    def __init__(self,default):
      self.enable = default
      self.iclasses = ()

# Components class
class components:
  def __init__(self):
    self.laser = Enable('false')
    self.raman = Enable('false')
    self.lw = Enable('false')
    self.iaw = Enable('false')
    self.hpe = Enable('false')
    self.iclasses = (Enable)

  write = std_write

# Temporal control class
class temporal_control:
  def __init__(self):
    self.simulation = self.Simulation()
    self.iclasses = (self.Simulation)

  class Simulation:
    def __init__(self):
      self.samplePeriod = None # Output sampling period [ps]
      self.time = self.Time()
      self.sampleTime = self.SampleTime()
      self.iclasses = (self.Time,self.SampleTime)

    class Time:
      def __init__(self):
        self.end = None # Sim end [ps]
        self.iclasses = ()

    class SampleTime:
      def __init__(self):
        self.timeStepID = None # Alternative to sample period, specifying timestep IDs
        self.iclasses = ()
  
  write = std_write

# Io control class
class io_control:
  def __init__(self):
    self.grid = self.Grid()
    self.lw = self.Wave(1)
    self.iaw = self.Wave(2)
    self.laser = self.Wave(3)
    self.raman = self.Wave(3)
    self.iclasses = (self.Grid,self.Wave)

  # Missing inner class for checkpoints

  class Grid:
    def __init__(self):
      self.downSampleFactors = 1 # Spatial downsampling per dimension in output
      self.iclasses = ()

  class Wave:
    def __init__(self,sv):
      self.save = self.Save(sv)
      self.iclasses = (self.Save)

    class Save:
      def __init__(self,sv):
        if sv == 1:
          self.pots = None # Electrostatic potentials
          self.iclasses = ()
        elif sv == 2:
          self.Nelf = None # Ion electorn concentrations
          self.iclasses = ()
        else:
          self.E0 = self.Dim(True) # Electric field
          self.S0 = self.Dim(False) # Poynting Flux
          self.iclasses = (self.Dim)

      class Dim:
        def __init__(self,sqr):
          self.x = None
          self.y = None
          self.z = None
          if sqr:
            self.__dict__['2'] = None
          self.iclasses = ()

  write = std_write
  
  # Return list of filenames used
  def fnames(self):
    kwords, specs = package(self)
    return specs[1:]

# Physical parameters class
class physical_parameters:
  def __init__(self):
    self.physical = self.Physical()
    self.lw = self.Lw()
    self.densityProfile = self.Density()
    self.iclasses = (self.Physical,self.Lw,self.Density)

  class Physical:
    def __init__(self):
      self.Z = None # Average ion charge
      self.Te = None # Electron temperature
      self.Ti = None # Ion temperature
      self.MiOverMe = None # Ion-electron mass ratio
      self.iclasses = ()

  class Lw:
    def __init__(self):
      self.envelopeDensity = 0.25 # EPW envelope fractional density
      self.iclasses = ()

  class Density:
    def __init__(self):
      self.shape = 'linear' # Density profile shape
      self.geometry = 'cartesian' # Profile geometry
      self.sgOrder = 2 # Order/power used in gaussian/inversePower profiles
      self.NminOverNc = None # Minimum fractional density
      self.NmaxOverNc = None # Maximum fractional density
      self.NminLocation = None # Minimum fractional density location
      self.NmaxLocation = None # Maximum fractional density location
      self.temporalSlope = 0.0 # Fractional density change with time [ps^-1]
      self.iclasses = ()

  write = std_write

# Light control class
class light_control:
  def __init__(self):
    self.laser = self.Light(True)
    self.raman = self.Light(False)
    self.iclasses = (self.Light)

  class Light:
    def __init__(self,laze):
      self.solver = 'static' # Algorithm for light wave eqs.
      self.ionAcousticPerturbations = Enable('false')
      self.static = self.Static()
      self.evolution = self.Evolution()
      self.spectral = self.Spectral()
      self.pulseShape = self.PulseShape()
      if laze:
        self.wavelength = 0.351 # Laser wavelength [microns]
        self.maxRamanStepsPerStep = 2 # Max raman steps per laser step
        self.pumpDepletion = self.Pump()
        self.iclasses = (self.Pump,Enable,self.Static,self.Evolution,\
                         self.Spectral,self.PulseShape)
      else:
        self.maxLaserStepsPerStep = 10 # Max laser steps per raman step
        self.sourceTerm = self.Source()
        self.iclasses = (self.Source,Enable,self.Static,self.Evolution,\
                         self.Spectral,self.PulseShape)

    class Pump:
      def __init__(self):
        self.SRS = Enable('false') # SRS pump depletion
        self.TPD = Enable('false') # TPD pump depletion
        self.iclasses = (Enable)
    class Source:
      def __init__(self):
        self.lw = Enable('false') # Raman Lw source term
        self.iclasses = (Enable)
    class Static:
      def __init__(self):
        self.computeFieldInXSpace = Enable('false') # Static light field computed in x-space
        self.iclasses = (Enable)
    class Spectral:
      def __init__(self):
        self.dt = 2e-4 # Spectral solver timestep
        self.iclasses = ()
    class PulseShape:
      def __init__(self):
        self.enable = 'false' # Scales the lights power according to pulse shape file
        self.file = None # Pulse shape filename
        self.nPulsesToList = 25 # Max number of pulse shape entries to echo to output log
        self.iclasses = (Enable)
    class Evolution:
      # Missing additional evolution inner class for resonant absorption
      def __init__(self):
        self.solverOrder = 2 # Finite differences solver order
        self.dtFraction = 0.95 # Fraction of critical timestep
        self.absorption = 0.0 # Collisional absorption coefficient [ps^-1]
        self.Labc = 0 # Distance of absorption zone from boundary [microns]
        self.Loff = 0.0 # Gap between absorbing boundary region and injection [microns]
        self.riseTime = 30.0 # Beam rise time as 1/e point [fs]
        self.abc = self.Abc()
        self.iclasses = (self.Abc)
      class Abc:
        def __init__(self):
          self.type = 'pml' # Absorbing boundary type
          self.iclasses = ()

  write = std_write

# Langmuir wave control class
class lw_control:
  def __init__(self):
    self.lw = self.Lw()
    self.iclasses = (self.Lw)

  class Lw:
    def __init__(self):
      self.solver = 'spectral' # Solver which determines electrostatic potential
      self.ionAcousticPerturbations = Enable('false') # Iaw source term
      self.SRS = Enable('false') # SRS source term
      self.TPD = Enable('false') # TPD source term
      self.spectral = self.Spectral()
      self.maxLightStepsPerStep = 10 # Max light steps per lw step
      self.Labc = 0 # Distance of absorption zone from boundary [microns]
      self.abc = self.Abc()
      self.noise = self.Noise()
      self.landauDamping = Enable('false') # Landau damping switch
      self.collisionalDampingRate = 0.0 # [ps^-1]
      self.__dict__['collisionalDampingRate.isCalculated'] = 'false' # Auto calc
      self.kFilter = self.KFilter()
      self.iclasses = (self.Spectral,self.Abc,self.Noise,Enable,self.KFilter)
    class Spectral:
      def __init__(self):
        self.dt = None # Spectral solver max timestep
        self.iclasses = ()
    class Abc:
      def __init__(self):
        self.type = 'exp' # Absorbing boundary type
        self.iclasses = ()
    class Noise:
      def __init__(self):
        self.enable = 'false' # Lw eq. noise term
        self.isCalculated = 'false' # Auto noise calculation (not working apparently)
        self.amplitude = 0.0 # Amplitude of noise term
        self.iclasses = ()
    class KFilter:
      def __init__(self):
        self.enable = 'false' # Filter spurious high wavenumbers
        self.scale = 1.2 # Envelope factor for cutoff
        self.iclasses = ()

  write = std_write

# Ion acoustic wave control class
class iaw_control:
  def __init__(self):
    self.iaw = self.Iaw()
    self.iclasses = (self.Iaw)

  # Missind additional inner class for velocity profile

  class Iaw:
    def __init__(self):
      self.solver = 'spectral'
      self.sourceTerm = self.Source()
      self.spectral = self.Spectral()
      self.fd = self.Fd()
      self.maxLightStepsPerStep = 10 
      self.maxLwStepsPerStep = 2 
      self.Labc = 0 # Distance of absorption zone from boundary [microns]
      self.abc = self.Abc()
      self.dampingRate = 0.0 # [0.0,1.0] normalised landau damping rate
      self.iclasses = (self.Source,self.Spectral,self.Abc,self.Fd)
    class Spectral:
      def __init__(self):
        self.dt = None # Spectral solver max timestep
        self.iclasses = ()
    class Fd:
      def __init__(self):
        self.numStepsPerLandauUpdate = None
        self.iclasses = ()
    class Source:
      def __init__(self):
        self.lw = Enable('false')
        self.laser = Enable('false')
        self.raman = Enable('false')
        self.iclasses = (Enable)
    class Abc:
      def __init__(self):
        self.type = 'exp' # Absorbing boundary type
        self.iclasses = ()

  write = std_write

# Instrumentation class
class instrumentation:
  def __init__(self):
    self.metrics = self.Metrics()
    self.iclasses = (self.Metrics)

  # Missing additional inner classes for Thomson scattering and light spectra

  class Metrics:
    def __init__(self):
      self.enable = 'false'
      self.file = None
      self.samplePeriod = 0.0 # [ps], if 0 then saved on longest sim timescale
      self.iclasses = ()

  write = std_write

# Light source class
class light_source:
  def __init__(self):
    self.laser = self.Light()
    self.raman = self.Light()
    self.iclasses = (self.Light)

  class Light:
    def __init__(self):
      self.nBeams = 0
      self.intensity = []
      self.phase = []
      self.polarization = []
      self.direction = []
      self.frequencyShift = []
      self.group = []
      self.onTheFlyInjection = None
      self.tableLookupInjection = None
      self.evolution = self.Evolution()
      self.iclasses = (self.Evolution,Enable)
    class Evolution:
      def __init__(self):
        self.source = []
        self.offset = []
        self.width = []
        self.sgOrder = []
        self.iclasses = ()

  # Write method
  def write(self,pout=True):
    kwords, specs = light_package(self)
    std_input(kwords,specs)
    if pout:
      print('Attributes written:')
      adict = {i:j for i,j in zip(kwords,specs)}
      print(adict)

# Extracts keywords and values from class object for writing to file
def light_package(obj,kwords=None,specs=None,prefix=None,nbeams=0):
  dic = obj.__dict__
  kys = list(dic.keys())
  if kwords is None:
    kwords = []
  if specs is None:
    specs = []
  if kys[0] == 'nBeams':
    nbeams = dic[kys[0]]
  for i in kys[:-1]:
    att = dic[i]
    if (att == None):
      continue
    # If attribute an inner class recursively call light_package
    elif (isinstance(att,obj.iclasses)):
      prefixnew = str(prefix or '') + i + '.'
      light_package(att,kwords,specs,prefixnew,nbeams)
    # If attribute a list, add each element with beamlet number
    elif isinstance(att,list):
      for j in range(nbeams):
        new_prefix = prefix[:6] + f'{j+1}.' + prefix[6:]
        kwords.append(new_prefix+i)
        specs.append(att[j])
    else:
      kwords.append(str(prefix or '')+i)
      specs.append(att)
  return kwords, specs


