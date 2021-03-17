#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import struct
import copy
import os
import write_files as wf
import multiprocessing as mp
import ray
from functools import partial
from matplotlib.widgets import Slider, Button, RadioButtons

# Function which wraps serial function for executing in parallel directories
@ray.remote
def parallel_wrap(inp,idx,func):
  d = f'./parallel/task{idx}'
  os.system(f'mkdir {d}')
  os.chdir(d)
  res = func(inp)
  os.chdir('../..')
  os.system(f'rm -r {d}/data')
  return res

def get_metrics(fname):

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
  
  return cols, ddict

def get_fields(fname,ddict,dsample,zgE):

  # If downsampling append suffix
  if (dsample > 1):
    fname = fname + f'.downSample_{dsample}'

  # Use zgExtractFrames to convert to ascii
  fnamebin = copy.deepcopy(fname)
  fname = fname + '.ascii'
  cmd = f'{zgE} --ascii --file= {fnamebin} > {fname}'
  os.system(cmd)

  # Strings separating headers and data
  splitters = ['# BeginHeaderSegment;\n','# EndHeaderSegment;\n',\
               '# BeginDataSegment;\n','# EndDataSegment;\n']

  # Header dictionary entries
  ents = ['FileType','isAscii','isBigEndian','isXSpace',\
          'time','Nx','Ny','Nz','gridSize','offset']
  hds = {i:[] for i in ents}

  header = False
  dat = {}; cnt = 0
  # Read file and extract data
  with open(fname,'r') as fp:
    for line in fp:
      # Check for file splitters
      if (any(line == i for i in splitters)):
        if (line == splitters[0]):
          header = True
          cnt += 1
          dat[cnt] = np.empty(0,dtype='complex64')
        if (line == splitters[1]):
          header = False
      else:
        if (header):
          # Read header information into dictionary
          hd = line.strip().split(',')
          hd[-1] = hd[-1].strip(';')
          for i in range(len(hd)):
            shd = hd[i].strip().split('=')
            hds[shd[0]].append(shd[1])
          hds['gridSize'][-1] = np.array(hds['gridSize'][-1].\
                                        split(),'float64')
        else:
          # Add data line
          num = line.strip('() \n').split(',')
          num = float(num[0]) + 1j*float(num[1])
          dat[cnt] = np.append(dat[cnt],num)
          
  # Convert necessary dictionary entries to numpy
  hds['time'] = np.array(hds['time'],'float64')
  hds['Nx'] = np.array(hds['Nx'],'int32')
  hds['Ny'] = np.array(hds['Ny'],'int32')
  hds['Nz'] = np.array(hds['Nz'],'int32')

  # Populate dictionary for return (currently only does 1D)
  ddict['time'] = hds['time']
  #assert hds['isXSpace'][0]=='t','Data in k space, need to modify code.'
  xs = hds['gridSize'][0][0]/2.
  x = np.linspace(-xs,xs,hds['Nx'][0])
  ddict['x'] = x
  ddict['data'] = np.empty((0,hds['Nx'][0]))
  for i in range(cnt):
    ddict['data'] = np.append(ddict['data'],np.array([dat[i+1]]),axis=0)
    
  return ddict

def get_fields_old(fname,ddict,dsample):

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
  #assert hds['isXSpace'][0]=='t','Data in k space, need to modify code.'
  xs = hds['gridSize'][0][0]/2.
  x = np.linspace(-xs,xs,hds['Nx'][0])
  ddict['x'] = x
  ddict['data'] = np.empty((0,hds['Nx'][0]))
  for i in range(cnt):
    ddict['data'] = np.append(ddict['data'],np.array([cmplx[i+1]]),axis=0)
    
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
    self.plasmaFrequencyDensity = None

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
    if self.np == 1:
      cmnd = f'{self.bin} --parms=lpse.parms'
    else:
      cmnd = f'mpirun -np {self.np} {self.bin} --parms=lpse.parms'
    if self.verbose:
      os.system(cmnd)
      print('LPSE run complete.')
    else:
      os.system(cmnd + ' > run.log')

  def metrics(self):
    # Get file name and extract data
    for i in self.setup_classes:
      if isinstance(i,wf.instrumentation):
        fname = i.metrics.file
    self.mkeys, self.mdat = get_metrics(fname)
    if self.verbose:
      print('Metrics data extracted.')

  def fields(self,fname=None):
    # Remove file prefix from dict keys
    for i in self.setup_classes:
      if isinstance(i,wf.io_control):
        fnames = i.fnames()
        dsamp = i.grid.downSampleFactors
    kys = [fnames[i].replace(self.dfp,'') \
                for i in range(len(fnames))]

    # Construct zgExtractFrames binary location string
    zgE = copy.deepcopy(self.bin)
    zgE = zgE.replace('lpse_cpu','utils/zgExtractFrames')

    # If no filename given do all data files
    if fname == None:
      self.fdat = {i:{} for i in kys}
      self.fkeys = kys
      for i in range(len(fnames)):
          self.fdat[kys[i]] = \
            get_fields(fnames[i],self.fdat[kys[i]],dsamp,zgE) 
    else:
      ky = fname.replace(self.dfp,'')
      if self.fdat == None:
        self.fdat = {ky:{}}
        self.fkeys = [ky]
      elif ky not in self.fdat:
        self.fkeys.append(ky)
        self.fdat[ky] = {}
      self.fdat[ky] = \
        get_fields(fname,self.fdat[ky],dsamp,zgE) 
    if self.verbose:
      print('Fields data extracted.')

  def plot_metric(self,ky,loglin=False):
    xdat = self.mdat['time']
    ydat = self.mdat[ky]
    if loglin:
      plt.semilogy(xdat,ydat,label=ky)
    else:
      plt.plot(xdat,ydat,label=ky)
    plt.xlabel('Time [ps]')
    plt.legend()
    plt.show()

  def plot_field(self,ky):
    xdat = self.fdat[ky]['x']
    ydat = np.real(self.fdat[ky]['data'])
    tdat = self.fdat[ky]['time']
    argtime = 0
    snaps = len(tdat)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    l, = plt.plot(xdat,ydat[argtime])
    ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.10, 0.05, 0.8, 0.03], facecolor=axcolor)
    delta_t = tdat[1]-tdat[0]
    stime = Slider(axtime, 'Time', tdat[0], tdat[-1], valinit=tdat[0], valstep=delta_t)

    def update(val):
        nonlocal argtime
        time = stime.val
        objf = abs(tdat-time)
        argtime = np.argmin(objf)
        l.set_ydata(ydat[argtime])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    stime.on_changed(update)

    plt.show()
    
  def plot_fields(self):
    ky = self.fkeys[0]
    xdat = self.fdat[ky]['x']
    ydat = np.real(self.fdat[ky]['data'])
    tdat = self.fdat[ky]['time']
    argtime = 0
    snaps = len(tdat)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    l, = plt.plot(xdat,ydat[argtime])
    ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    rax = plt.axes([0.025, 0.5, 0.075, 0.25], facecolor=axcolor)
    radio = RadioButtons(rax, [i for i in self.fkeys], active=0)

    def change_field(ky):
        nonlocal ydat
        nonlocal argtime
        ydat = np.real(self.fdat[ky]['data'])
        l.set_ydata(ydat[argtime])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    radio.on_clicked(change_field)

    axtime = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    delta_t = tdat[1]-tdat[0]
    stime = Slider(axtime, 'Time', tdat[0], tdat[-1], valinit=tdat[0], valstep=delta_t)

    def update(val):
        nonlocal argtime
        time = stime.val
        objf = abs(tdat-time)
        argtime = np.argmin(objf)
        l.set_ydata(ydat[argtime])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    stime.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        stime.reset()
    button.on_clicked(reset)

    plt.show()

  # Method which takes function, and 2D array of inputs
  # Then runs in parallel for each set of inputs
  # Returning 2D array of outputs
  def parallel_runs(self,func,inps,nps):
    
    # Ensure number of requested processors is reasonable
    assert (nps <= mp.cpu_count()),\
        "Error: number of processors selected exceeds available."
    
    # Create parallel directory for tasks
    os.system('mkdir parallel')

    # Run function in parallel    
    ray.init(num_cpus=nps,log_to_driver=False)
    l = len(inps)
    outs = ray.get([parallel_wrap.remote(inps[i],i,func) for i in range(l)])
    ray.shutdown()
    
    # Reshape outputs to 2D array
    if isinstance(outs[0],list):
      ll = len(outs[0])
    else:
      ll = 1
    outs = np.array(outs).reshape((l,ll))

    return outs
