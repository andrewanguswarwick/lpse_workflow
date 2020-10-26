#!/bin/python3

import write_files as wf
import numpy as np
from GPyOpt.methods import BayesianOptimization
import copy
from functools import partial

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
    # Error handling returns Isrs = 0 if dict read fails
    try:
      Sdat = np.real(case.fdat[ky]['data'][-i+1,:])
      Isrs += Sdat[whe][0,0]
    except:
      print('Error: LPSE run terminated prematurely')
      Isrs = 0.0
  Isrs /= touts  
  return abs(Isrs)

def noise_amp(amp,case,tavg):
  # Set lw noise attribute to amp
  if not isinstance(amp,float):
    amp = amp[0] # For parallel runs
  for i in case.setup_classes:
    if isinstance(i,wf.lw_control):
      i.lw.noise.amplitude = amp
  
  # Return Isrs
  return Isrs(case,tavg)

def test_amp(amp,case,tavg):
  return amp*5.333333e12

def Isrs_curve(case,tavg,Isrs0,Irange):
  # Ensure laser intensity is base value
  for i in case.setup_classes:
    if isinstance(i,wf.light_source):
      i.laser.intensity = [Irange[0]]
  # Perform short golden search iteration to get noise
  print('Finding optimum LW noise amplitude...')
  objf = lambda amp: abs(noise_amp(amp[0,0],case,tavg)-Isrs0)
  domain = [{'name':'amp','type':'continuous','domain':(0.005,0.025)}]
  Bopt = BayesianOptimization(f=objf,domain=domain)
  Bopt.run_optimization(max_iter=15)
  Bopt.plot_acquisition()
  amp0 = Bopt.x_opt[0]
  f0 = Bopt.fx_opt
  print(f'Best LW noise amplitude is: {amp0:0.5f}')
  print(f'Giving base <I_srs>: {f0:0.3e} W/cm^2')

  # Use result and get Isrs for range of laser intensities
  for i in case.setup_classes:
    if isinstance(i,wf.lw_control):
      i.lw.noise.amplitude = amp0
  Isrsvals = np.zeros_like(Irange)
  print('Obtaining <I_srs> for laser intensity range...')
  print('0% complete.',end='\r')
  for j,I in enumerate(Irange):
    for i in case.setup_classes:
      if isinstance(i,wf.light_source):
        i.laser.intensity = [str(I)]
    Isrsvals[j] = Isrs(case,tavg)
    print(f'{(j+1)/len(Irange):0.1%} complete.',end='\r')

  return Isrsvals

def Isrs_dens(case,dens,dlabs,tavg,Isrs0,Irange):
  isrs = {i:None for i in dlabs}
  for i in range(len(dens)):
    case.add_class(dens[i])
    isrs[dlabs[i]] = Isrs_curve(case,tavg,Isrs0,Irange)
  return isrs

# Gets training set for GPyOpt of LW noise amp
def amp_par(case,dens,dlabs,tavg,cpus,train):
  amps = np.linspace(0.005,0.025,train)
  amps = np.reshape(amps,(train,1))
  x0 = {i:amps for i in dlabs}; y0 = {}
  for i in range(len(dlabs)):
    case.add_class(dens[i])
    func = partial(noise_amp,case=case,tavg=tavg)
    y0[dlabs[i]] = case.parallel_runs(func,amps,cpus)
  
  return x0, y0
