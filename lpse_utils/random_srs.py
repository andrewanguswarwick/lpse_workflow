#!/bin/python3

import write_files as wf
import calc_inputs as ci
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

def Isrs_las(Ilas,case,tavg):
  # Set laser beam intensity to input
  if not isinstance(Ilas,float):
    Ilas = Ilas[0] # For parallel runs
  for i in case.setup_classes:
    if isinstance(i,wf.light_source):
      i.laser.intensity = [Ilas]
  
  # Return Isrs
  return Isrs(case,tavg)

def Isrs_curve(case,tavg,Isrs0,Irange,X0,Y0,parallel,cpus):
  # Ensure laser intensity is base value
  for i in case.setup_classes:
    if isinstance(i,wf.light_source):
      i.laser.intensity = [Irange[0]]

  # Use Bayesian optimisation to fit LW noise amplitude
  print('Finding optimum LW noise amplitude...')
  objf = lambda amp: abs(noise_amp(amp[0,0],case,tavg)-Isrs0)
  domain = [{'name':'amp','type':'continuous','domain':(0.005,0.025)}]
  Bopt = BayesianOptimization(f=objf,domain=domain,X=X0,Y=Y0)
  Bopt.run_optimization(max_iter=5)
  Bopt.plot_acquisition()
  amp0 = Bopt.x_opt[0]
  f0 = Bopt.fx_opt
  print(f'Best LW noise amplitude is: {amp0:0.5f}')
  print(f'Giving an <I_srs> error of: {f0:0.3e} W/cm^2')

  # Set case LW noise to optimum
  for i in case.setup_classes:
    if isinstance(i,wf.lw_control):
      i.lw.noise.amplitude = amp0

  # Get Isrs for range of laser intensities
  print('Obtaining <I_srs> for laser intensity range...')
  if parallel:
    func = partial(Isrs_las,case=copy.deepcopy(case),tavg=tavg)
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

def Isrs_dens(ocase,dens,cdens,dlabs,tavg,Isrs0,Irange,\
              x0=None,y0=None,parallel=False,cpus=1,cells_per_wvl=30):
  isrs = {i:None for i in dlabs}
  for i in range(len(dens)):
    case = copy.deepcopy(ocase)
    case.add_class(dens[i])
    case.plasmaFrequencyDensity = cdens[i]
    freqs = ci.bsrs_lw_envelope(case,cells_per_wvl)
    ci.spectral_dt(case,freqs)
    if x0 != None:
      X0 = x0[dlabs[i]]
      Y0 = y0[dlabs[i]]
    else:
      X0 = None; Y0 = None
    isrs[dlabs[i]] = Isrs_curve(case,tavg,Isrs0,Irange,\
                      X0,Y0,parallel,cpus)
  return isrs

# Gets training set for GPyOpt of LW noise amp
def amp_par(ocase,dens,cdens,dlabs,tavg,Isrs0,cpus,train,cells_per_wvl):
  amps = np.linspace(0.005,0.025,train)
  amps = np.reshape(amps,(train,1))
  x0 = {i:amps for i in dlabs}; y0 = {}
  for i in range(len(dlabs)):
    # Add density class
    case = copy.deepcopy(ocase)
    case.add_class(dens[i])

    # Calculate LW envelope density and spectral timesteps
    case.plasmaFrequencyDensity = cdens[i]
    freqs = ci.bsrs_lw_envelope(case,cells_per_wvl)
    ci.spectral_dt(case,freqs)

    # Run case
    func = partial(noise_amp,case=case,tavg=tavg)
    res = case.parallel_runs(func,amps,cpus)
    y0[dlabs[i]] = abs(res-Isrs0)
  
  return x0, y0
