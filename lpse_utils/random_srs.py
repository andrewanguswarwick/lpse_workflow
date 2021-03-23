#!/bin/python3

import write_files as wf
import calc_inputs as ci
import numpy as np
import copy
from functools import partial
from time import time as stopwatch
from scipy.special import erf
import scipy.constants as scc
from scipy.optimize import minimize

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

def ray_trace(case,points=101,noise=True):
  # Extract relevant quantities from case
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      nmax = i.densityProfile.NmaxOverNc
      nmin = i.densityProfile.NminOverNc
    elif isinstance(i,wf.gridding):
      split = i.grid.sizes/2
    elif isinstance(i,wf.light_control):
      xbuff = i.laser.evolution.Labc+i.laser.evolution.Loff 
    elif isinstance(i,wf.light_source):
      I00 = np.float64(i.laser.intensity[0])*1.0e4
      I10 = np.float64(i.raman.intensity[0])*1.0e4

  # Establish density profile
  xmin = xbuff - split
  xmax = split - xbuff
  dr = abs(xmin-xmax)
  Ln = dr/np.log(nmax/nmin)*1e-6
  x = np.linspace(xmin,xmax,points)*1e-6
  r = abs(x-xmax*1e-6)
  n = nmax*np.exp(-r/Ln)

  # Get central resonant info
  case.plasmaFrequencyDensity = n[-int(points/2)-1]
  freqs,wavens,kvac,vth,dby = ci.bsrs_lw_envelope(case,return_all=True)
  omega_pe = vth/dby

  # Calculate coupling constant across domain
  gr = np.zeros_like(x)
  grres = np.zeros_like(x)
  klas = np.zeros_like(x)
  kram = np.zeros_like(x)
  normf = scc.m_e**2*scc.c**2/scc.e**2/scc.mu_0
  for i in range(points):
    # Resonance conditions
    case.plasmaFrequencyDensity = n[i]
    freqst,wavenst,kvact,vtht,dbyt = ci.bsrs_lw_envelope(case,return_all=True)

    # Get new LW wavenumber
    ope = freqs[0]*np.sqrt(n[i])
    
    # Get new wavenumbers
    klas[i] = np.sqrt(freqs[0]**2-ope**2)
    kram[i] = -np.sqrt(freqs[1]**2-ope**2)
    kl = klas[i]-kram[i]

    # Maximise coupling coefficient at local res for noise growth
    def coupling_res(omram):
      omlw = freqs[0]-omram
      kr = -omram*np.sqrt(1-n[i]/omram**2)
      klw = klas[i]+abs(kr)
      zarg = omlw/(klw*vth*np.sqrt(2))
      Z = 1j*np.sqrt(np.pi)*np.exp(-zarg**2)*(1+erf(1j*zarg))
      dZ = -2*(1+zarg*Z)
      esus = -n[i]/(2*klw**2*vth**2)*dZ
      Fij = np.imag(1/4/(1+esus))
      return Fij
    meps = 4*np.finfo(np.float64).eps
    res = minimize(coupling_res,x0=freqst[1])#,bounds=[(ope,1)])#,\
                  #options={'maxcor':10,'ftol':4*meps,'gtol':4*meps,'eps':1e-08,\
                  #'maxfun':15000,'maxiter':15000,'iprint':1,'maxls':20,\
                  #'finite_diff_rel_step':None})
    omram = res.x
    omlw = freqs[0] - res.x
    kr = -omram*np.sqrt(1-n[i]/omram**2)
    klw = wavens[0]+abs(kr)
    grres[i] = -2*res.fun

    # Get coupling factor at raman frequency of interest
    gr[i] = -2*coupling_res(freqs[1])

  # Evolve intensities across domain and iterate to convergence
  dx = (x[1]-x[0])*kvac
  tol = 1e-10; conv = 1
  normf = scc.m_e**2*scc.c**3*kvac**2/scc.e**2/scc.mu_0**2
  omega0 = freqs[0]
  omega1 = freqs[1]
  I1 = np.ones_like(x)
  I0 = np.ones_like(x)
  I1 *= I10/normf; I0 *= I00/normf
  niter = 0
  while (conv > tol and niter < 1000):
    I0old = copy.deepcopy(I0)
    I1old = copy.deepcopy(I1)
    for i in range(points-1):
      if noise:
        I0[i+1] = np.maximum(I0[i]-(1/klas[i])*I0[i]*dx*(gr[i]*I1[i]+grres[i]*I1[-1]),0)
        I1[-2-i] = np.maximum(I1[-1-i]-(1/kram[i])*I0[-1-i]*dx*\
                    (gr[-1-i]*I1[-1-i]+grres[-1-i]*I1[-1]),0)
      else:
        I0[i+1] = np.maximum(I0[i]-(1/klas[i])*I0[i]*dx*gr[i]*I1[i],0)
        I1[-2-i] = np.maximum(I1[-1-i]-(1/kram[i])*I0[-1-i]*dx*\
                    gr[-1-i]*I1[-1-i],0)
    niter += 1
    if niter < 10:
      conv = 1
    else:
      conv = np.sum(abs(I0-I0old)+abs(I1-I1old))

  # Return stolen units
  I0 *= normf*1e-4; I1 *= normf*1e-4
      
  return x,n,I0,I1

    
  
