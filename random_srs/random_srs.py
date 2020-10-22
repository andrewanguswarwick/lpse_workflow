#!/bin/python3

import sys; sys.path.insert(0,'..')
import lpse_utils.write_files as wf
import numpy as np

def Isrs(case,tavg):
  # Write lpse.parms and run case
  case.write()
  case.run()

  # Get necessary attributes from setup classes
  for i in case.setup_classes:
    if isinstance(i,wf.io_control):
      fname = i.raman.save.S0.x
      ky = fname.replace(case.dfp,'')
    elif isinstance(i,wf.gridding):
      split = i.grid.sizes/2
    elif isinstance(i,wf.light_control):
      xbuff = i.laser.evolution.Labc+i.laser.evolution.Loff 
    elif isinstance(i,wf.temporal_control):
      touts = int(tavg/i.simulation.samplePeriod)
  xmin = xbuff - split

  # Extract data
  case.fields(fname=fname)

  # Calculate <I_srs> and return
  xdat = case.fdat[ky]['x']
  whe = np.argwhere(xdat > xmin)
  Isrs = 0
  for i in range(touts):
    Sdat = np.real(case.fdat[ky]['data'][-i+1,:])
    Isrs += Sdat[whe][0,0]
  Isrs /= touts  
  return Isrs

def noise_amp(amp,case,tavg):
  # Set lw noise attribute to amp
  for i in case.setup_classes:
    if isinstance(i,wf.lw_control):
      i.lw.noise.amplitude = amp
  
  # Return Isrs
  return Isrs(case,tavg)
