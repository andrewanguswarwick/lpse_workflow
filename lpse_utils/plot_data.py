#!/bin/python3

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import struct
import copy

def plot_metrics():

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

  # Plot data
  print(cols)
  for i in range(1,ncols):
    plt.plot(dat[:,0],dat[:,i],ls='-')
    plt.xlabel(cols[0])
    plt.ylabel(cols[i])
    plt.show()

def plot_fields(fnames,dsample):

  # If downsampling append suffix
  if (dsample > 1):
    fnames = [i + f'.downSample_{dsample}' for i in fnames]

  # Strings separating headers and bytes
  splitters = [b'# BeginHeaderSegment;\n',b'# EndHeaderSegment;\n',\
               b'# BeginDataSegment;\n']

  # Specify binary number format and get byte size
  binfmt = 'f'
  dsize = struct.calcsize(binfmt)

  # Header dictionary entries
  ents = ['FileType','isAscii','isBigEndian','isXSpace',\
          'time','Nx','Ny','Nz','gridSize','offset']
  def_hds = dict.fromkeys(ents)
  for i in range(len(ents)):
    def_hds[ents[i]] = []

  # Loop over all binary files
  for j in range(len(fnames)):
    header = False
    dat = {}; cnt = 0
    hds = copy.deepcopy(def_hds)
    # Read file and extract data
    with open(fnames[j],'rb') as fp:
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
      
    
      # Define grid and plot frame
      assert hds['isXSpace'][i]=='t','Data in k space, need to modify code.'
      xs = hds['gridSize'][i][0]/2.
      x = np.linspace(-xs,xs,hds['Nx'][i])
      plt.plot(x,np.imag(cmplx[i+1]),\
          label=f'imag {hds["FileType"][i]} {hds["time"][i]:0.3f}')
      plt.plot(x,np.real(cmplx[i+1]),\
          label=f'real {hds["FileType"][i]} {hds["time"][i]:0.3}')
       
      plt.legend()
      plt.show()
    

