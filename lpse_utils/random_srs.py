#!/bin/python3

import write_files as wf
import calc_inputs as ci
import numpy as np
import copy
from functools import partial
from time import time as stopwatch

def Isrs(case,tavg,splits,cpw=8,verbose=False):
  # Get necessary attributes from setup classes
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      nmax = i.densityProfile.NmaxOverNc
      nmin = i.densityProfile.NminOverNc
    elif isinstance(i,wf.io_control):
      fname = i.raman.save.S0.x
      ky = fname.replace(case.dfp,'')
    elif isinstance(i,wf.gridding):
      split = i.grid.sizes/2
    elif isinstance(i,wf.light_control):
      xbuff = i.laser.evolution.Labc+i.laser.evolution.Loff 
    elif isinstance(i,wf.temporal_control):
      touts = int(tavg/i.simulation.samplePeriod)

  # Get central densities of density range splits
  xmin = xbuff - split
  xmax = split - xbuff
  dx = abs(xmin-xmax)
  xsep = dx/splits
  xcens = np.array([j*xsep + xmin + 0.5*xsep for j in range(splits)])
  Ln = dx/np.log(nmax/nmin)
  ncens = nmax*np.exp(-abs(xcens-xmax)/Ln)

  # Run sim for each envelope density and sum Isrs
  isrstot = 0; t0 = stopwatch()
  for j in range(splits):
    t2 = stopwatch()
    case.plasmaFrequencyDensity = ncens[j]
    freqs = ci.bsrs_lw_envelope(case,cpw,verbose)
    ci.spectral_dt(case,freqs,dt_frac=0.99,verbose=verbose)

    # Write lpse.parms and run case
    case.write()
    case.run()

    # Extract data
    case.fields(fname=fname)

    # Calculate <I_srs> and return
    xdat = case.fdat[ky]['x']
    whe = np.argwhere(xdat > xmin)
    Isrs = 0
    for i in range(touts):
      # Error handling returns Isrs = 0 if dict read fails
      try:
        Sdat = np.real(case.fdat[ky]['data'][-i-1,:])
        Isrs += Sdat[whe][0,0]
      except:
        print('Error: LPSE run terminated prematurely')
        Isrs = 0.0
    Isrs /= touts  
    t3 = stopwatch()
    isrstot += abs(Isrs)
    if verbose: 
      print(f'Time taken: {t3-t2:0.3f} s')
      print(f'<I_srs>: {Isrs:0.3e} W/cm^2')

  # Average
  t1 = stopwatch()
  isrsav = isrstot/splits
  if verbose:
    print(f'Intensity sum: {isrstot:0.3e} W/cm^2')
    print(f'Intensity average: {isrsav:0.3e} W/cm^2')
    print(f'Total time taken: {t1-t0:0.3f} s')

  return isrsav

def Isrs_las(Ilas,case,tavg,splits,cpw):
  # Set laser beam intensity to input
  if not isinstance(Ilas,float):
    Ilas = Ilas[0] # For parallel runs
  for i in case.setup_classes:
    if isinstance(i,wf.light_source):
      i.laser.intensity = [Ilas]
  
  # Return Isrs
  return Isrs(case,tavg,splits,cpw)

def Isrs_curve(case,tavg,Irange,parallel,cpus,splits,cpw):

  # Get Isrs for range of laser intensities
  print('Obtaining <I_srs> for laser intensity range...')
  if parallel:
    func = partial(Isrs_las,case=copy.deepcopy(case),tavg=tavg,\
                  splits=splits,cpw=cpw)
    inps = np.reshape(Irange,(len(Irange),1))
    Isrsvals = case.parallel_runs(func,inps,cpus)
    Isrsvals = np.reshape(Isrsvals,len(Isrsvals))
  else:
    Isrsvals = np.zeros_like(Irange)
    print('0% complete.',end='\r')
    for j,I in enumerate(Irange):
      for i in case.setup_classes:
        if isinstance(i,wf.light_source):
          i.laser.intensity = [str(I)]
      Isrsvals[j] = Isrs(case,tavg)
      print(f'{(j+1)/len(Irange):0.1%} complete.',end='\r')
  print(f'100.0% complete.',end='\r')

  return Isrsvals

# Get Isrs curve for each density profile specified
def Isrs_dens(ocase,dens,dlabs,tavg,Isrs0,Irange,\
              parallel=False,cpus=1,splits=1,cpw=10):

  isrs = {i:None for i in dlabs}
  for i in range(len(dens)):
    case = copy.deepcopy(ocase)
    case.add_class(dens[i])

    # Set raman seed intensity
    print('Fitting raman seed beam intensity...')
    Isrs14(Isrs0[i],case,tavg,splits,cpw)

    # Get srs curves
    isrs[dlabs[i]] = Isrs_curve(case,tavg,Irange,\
                      parallel,cpus,splits,cpw)
  return isrs

# Set raman seed intensity to match EPOCH at 10^14 laser I
def Isrs14(I14,case,tavg,splits,cpw,verbose=False,savingI0=False):
  # Set raman to reference intensity
  baseI = 8e10
  for j in case.setup_classes:
    if isinstance(j,wf.light_source):
      if savingI0:
        saveI0 = j.laser.intensity[0]
      j.laser.intensity = [1e14]
      j.raman.intensity = [baseI]

  # Get gain
  srs14 = Isrs(case,tavg,splits,cpw,verbose)
  gain = srs14/baseI

  # Calculate required raman intensity 
  for j in case.setup_classes:
    if isinstance(j,wf.light_source):
      if savingI0:
        j.laser.intensity = [saveI0]
      j.raman.intensity = [I14/gain]
      if verbose:
        print(f'Raman seed intensity: {I14/gain:0.3e} W/cm^2')
