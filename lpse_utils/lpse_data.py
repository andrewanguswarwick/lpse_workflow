#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import struct
import copy
import os
import write_files as wf

def get_metrics(fname,plot=False):

  # Read file and extract data
  cutoff = 3 # Number of rows of file header to exclude
  cols = []
  ncols = 0
  with open('data/lpse.metrics','r') as fp:
    for cnt, line in enumerate(fp):
      if (cnt < cutoff):
        continue
      ln = line.strip().split()
      if (ln[0] == '#'):
        # New column
        cols.append(ln[2])
        ncols += 1
        dat = np.empty((0,ncols))
      else:
        # Read data row
        ln = np.array([ln],dtype=np.float64)
        dat = np.r_[dat,ln]

  # Create data dictionary and return for further use
  ddict = {}
  for i in range(ncols):
    ddict[cols[i]] = dat[:,i]
  
  # Optionally plot data
  if (plot):
    for i in range(1,ncols):
      plt.plot(dat[:,0],dat[:,i],ls='-')
      plt.xlabel(cols[0])
      plt.ylabel(cols[i])
      plt.show()

  return cols, dat

def get_fields(fname,ddict,dsample,plot=False):

  # If downsampling append suffix
  if (dsample > 1):
    fname = fname + f'.downSample_{dsample}'

  # Strings separating headers and bytes
  splitters = [b'# BeginHeaderSegment;\n',b'# EndHeaderSegment;\n',\
               b'# BeginDataSegment;\n']

  # Specify binary number format and get byte size
  binfmt = 'f'
  dsize = struct.calcsize(binfmt)

  # Header dictionary entries
  ents = ['FileType','isAscii','isBigEndian','isXSpace',\
          'time','Nx','Ny','Nz','gridSize','offset']
  hds = {i:[] for i in ents}

  header = False
  dat = {}; cnt = 0
  # Read file and extract data
  with open(fname,'rb') as fp:
    for line in fp:
      # Check for file splitters
      if (any(line == i for i in splitters)):
        if (line == splitters[0]):
          header = True
          cnt += 1
          dat[cnt] = b''
        if (line == splitters[2]):
          header = False
      else:
        if (header):
          # Read header information into dictionary
          hd = line.decode('ascii')
          hd = hd.strip().split(',')
          hd[-1] = hd[-1].strip(';')
          for i in range(len(hd)):
            shd = hd[i].strip().split('=')
            hds[shd[0]].append(shd[1])
          hds['gridSize'][-1] = np.array(hds['gridSize'][-1].\
                                        split(),'float64')
        else:
          # Concatenate data line into binary data string
          dat[cnt] += line
          
  # Convert necessary dictionary entries to numpy
  hds['time'] = np.array(hds['time'],'float64')
  hds['Nx'] = np.array(hds['Nx'],'int32')
  hds['Ny'] = np.array(hds['Ny'],'int32')
  hds['Nz'] = np.array(hds['Nz'],'int32')

  # Convert bytes into complex numbers
  cmplx = {}; trck = 0
  for i in range(cnt):
    nums = hds['Nx'][i]*hds['Ny'][i]*hds['Nz'][i]*2
    cmplx[i+1] = np.empty(0,dtype='complex64')
    for k in range(nums):
      tmp = trck + dsize
      slc = dat[i+1][trck:tmp]
      flt = struct.unpack(binfmt,slc)[0]
      if (tmp % (2*dsize) == 0):
        num = num + 1j*flt
        cmplx[i+1] = np.append(cmplx[i+1],num)
      else:
        num = flt + 1j*0
      trck = tmp
    trck = 0

  # Populate dictionary for return
  ddict['time'] = hds['time']
  assert hds['isXSpace'][0]=='t','Data in k space, need to modify code.'
  xs = hds['gridSize'][0][0]/2.
  x = np.linspace(-xs,xs,hds['Nx'][0])
  ddict['x'] = x
  ddict['data'] = np.empty((0,hds['Nx'][0]))
  for i in range(cnt):
    ddict['data'] = np.append(ddict['data'],np.array([cmplx[i+1]]),axis=0)
    
  
  # Optionally plot frames
  if (plot):
    for i in range(cnt):
      plt.plot(ddict['x'],np.imag(cmplx[i+1]),\
          label=f'imag {hds["FileType"][i]} {hds["time"][i]:0.3f}')
      plt.plot(ddict['x'],np.real(cmplx[i+1]),\
          label=f'real {hds["FileType"][i]} {hds["time"][i]:0.3}')
       
      plt.legend()
      plt.show()
    
  return ddict

# Class for running lpse, then extracting and plotting results
class lpse_case:
  def __init__(self):
    self.setup_classes = []
    self.np = 1
    self.bin = None
    self.dfp = './data/lpse.'
    self.mkeys = None
    self.mdat = None
    self.fkeys = None
    self.fdat = None
    self.verbose = False

  # Methods 
  def add_class(self,obj):
    exs_types = [type(i) for i in self.setup_classes]
    added = False
    for i in range(len(exs_types)):
      if type(obj) == exs_types[i]:
        self.setup_classes[i] = obj
        added = True
        if self.verbose:
          print('Existing setup class of this type overwritten.')
    if not added:
      self.setup_classes.append(obj)
      if self.verbose:
        print('Setup class added.')

  def write(self,pout=False):
    os.system('> lpse.parms')
    for i in self.setup_classes:
      i.write(pout)
    if len(self.setup_classes) == 0:
      print('write() error: No setup classes specified.')
    if self.verbose:
      print('File \'lpse.parms\' written.')

  def run(self):
    if self.bin == None:
      print('run() error: Must specify binary location.')
      return
    if not os.path.exists('data'):
      os.system('mkdir data')
    os.system('mpirun -np ' + str(self.np) + \
               ' ' + self.bin + ' --parms=lpse.parms')
    if self.verbose:
      print('LPSE run complete.')

  def metrics(self,plot=False):
    # Get file name and extract data
    for i in self.setup_classes:
      if isinstance(i,wf.instrumentation):
        fname = i.metrics.file
    self.mkeys, self.mdat = get_metrics(fname,plot)
    if self.verbose:
      print('Metrics data extracted.')

  def fields(self,plot=False,fname=None):
    # Remove file prefix from dict keys
    for i in self.setup_classes:
      if isinstance(i,wf.io_control):
        fnames = i.fnames()
        dsamp = i.grid.downSampleFactors
    kys = [fnames[i].replace(self.dfp,'') \
                for i in range(len(fnames))]

    # If no filename given do all data files
    if fname == None:
      self.fdat = {i:{} for i in kys}
      self.fkeys = kys
      for i in range(len(fnames)):
          self.fdat[kys[i]] = \
            get_fields(fnames[i],self.fdat[kys[i]],dsamp,plot) 
    else:
      ky = fname.replace(self.dfp,'')
      if self.fdat == None:
        self.fdat = {ky:{}}
        self.fkeys = [ky]
      elif ky not in self.fdat:
        self.fkeys.append(ky)
        self.fdat[ky] = {}
      self.fdat[ky] = \
        get_fields(fname,self.fdat[ky],dsamp,plot) 
    if self.verbose:
      print('Fields data extracted.')
