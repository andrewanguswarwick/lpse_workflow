#!/bin/python3

import write_files as wf
import calc_inputs as ci
import numpy as np
import copy
from functools import partial
from time import time as stopwatch
from scipy.special import wofz
import scipy.constants as scc
from scipy.optimize import minimize,newton
from scipy.integrate import solve_ivp

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
  # Choose SRS function
  if rt:
    srsfunc = ray_trace
  else:
    srsfunc = partial(Isrs,tavg=tavg,splits=splits,cpw=cpw)

  # Set laser beam intensity to input
  if not isinstance(Ilas,float):
    Ilas = Ilas[0] # For parallel runs
  for i in case.setup_classes:
    if isinstance(i,wf.light_source):
      i.laser.intensity = [Ilas]
  
  # Return Isrs
  #return Isrs(case,tavg,splits,cpw)
  return srsfunc(case)

def Isrs_curve(case,tavg,Irange,parallel,cpus,splits,cpw,rt):

  # Get Isrs for range of laser intensities
  print('Obtaining <I_srs> for laser intensity range...')
  if parallel:
    func = partial(Isrs_las,case=copy.deepcopy(case),tavg=tavg,\
                  splits=splits,cpw=cpw)
    inps = np.reshape(Irange,(len(Irange),1))
    Isrsvals = case.parallel_runs(func,inps,cpus)
    Isrsvals = np.reshape(Isrsvals,len(Isrsvals))
  else:
    # Choose SRS function
    if rt:
      srsfunc = ray_trace
    else:
      srsfunc = partial(Isrs,tavg=tavg,splits=splits,cpw=cpw)
    Isrsvals = np.zeros_like(Irange)
    print('0% complete.',end='\r')
    for j,I in enumerate(Irange):
      for i in case.setup_classes:
        if isinstance(i,wf.light_source):
          i.laser.intensity = [str(I)]
      #Isrsvals[j] = Isrs(case,tavg,splits,cpw)
      Isrsvals[j] = srsfunc(case)
      print(f'{(j+1)/len(Irange):0.1%} complete.',end='\r')
  print(f'100.0% complete.',end='\r')

  return Isrsvals

# Get Isrs curve for each density profile specified
def Isrs_dens(ocase,dens,dlabs,tavg,Isrs0,Irange,\
              parallel=False,cpus=1,splits=1,cpw=10,rt=False):

  isrs = {i:None for i in dlabs}
  for i in range(len(dens)):
    case = copy.deepcopy(ocase)
    case.add_class(dens[i])

    # Set raman seed intensity
    print('Fitting raman seed beam intensity...')
    Isrs14(Isrs0[i],case,tavg,splits,cpw,rt)

    # Get srs curves
    isrs[dlabs[i]] = Isrs_curve(case,tavg,Irange,\
                      parallel,cpus,splits,cpw,rt)
  return isrs

# Set raman seed intensity to match EPOCH at 10^14 laser I
def Isrs14(I14,case,tavg,splits,cpw,rt=False,verbose=False,savingI0=False):
  # Choose SRS function
  if rt:
    srsfunc = ray_trace
  else:
    srsfunc = partial(Isrs,tavg=tavg,splits=splits,cpw=cpw,verbose=verbose)

  # Set raman to reference intensity
  baseI = 8e10
  for j in case.setup_classes:
    if isinstance(j,wf.light_source):
      if savingI0:
        saveI0 = j.laser.intensity[0]
      j.laser.intensity = [1e14]
      j.raman.intensity = [baseI]

  # Get gain
  #srs14 = Isrs(case,tavg,splits,cpw,verbose)
  srs14 = srsfunc(case)
  gain = srs14/baseI

  # Calculate required raman intensity 
  for j in case.setup_classes:
    if isinstance(j,wf.light_source):
      if savingI0:
        j.laser.intensity = [saveI0]
      j.raman.intensity = [I14/gain]
      if verbose:
        print(f'Raman seed intensity: {I14/gain:0.3e} W/cm^2')

def ray_trace(ocase,points=101,return_all=False,noise=False):
  # Extract relevant quantities from case
  case = copy.deepcopy(ocase)
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      nmax = i.densityProfile.NmaxOverNc
      nmin = i.densityProfile.NminOverNc
      mime = i.physical.MiOverMe
    elif isinstance(i,wf.gridding):
      split = i.grid.sizes/2*1e-6
    elif isinstance(i,wf.light_control):
      xbuff = (i.laser.evolution.Labc+i.laser.evolution.Loff)*1e-6
    elif isinstance(i,wf.light_source):
      I00 = np.float64(i.laser.intensity[0])*1.0e4
      I10 = np.float64(i.raman.intensity[0])*1.0e4

  # Establish density profile
  xmin = xbuff - split
  xmax = split - xbuff
  dr = abs(xmin-xmax)
  Ln = dr/np.log(nmax/nmin)
  x = np.linspace(xmin,xmax,points)
  r = abs(x-xmax)
  n = nmax*np.exp(-r/Ln)

  # Get central resonant info
  case.plasmaFrequencyDensity = n[-int(points/2)-1]
  freqs,wavens,kvac,vth,dby,LD = ci.bsrs_lw_envelope(case,return_all=True,dispfun=True)
  omega_pe = vth/dby

  # Plasma dispersion function derivative
  def dZfun(zeta):
    Z = 1j*np.sqrt(np.pi)*wofz(zeta)
    return -2*(1+zeta*Z)

  # Calculate coupling constant across domain
  gr = np.zeros_like(x)
  grres = np.zeros_like(x)
  klas = np.zeros_like(x)
  kram = np.zeros_like(x)
  for i in range(points):
    # Resonance conditions
    if noise:
      case.plasmaFrequencyDensity = n[i]
      freqst,wavenst,kvact,vtht,dbyt,LDt = ci.bsrs_lw_envelope(case,return_all=True,dispfun=True)

    # Get new LW wavenumber
    ope = np.sqrt(n[i])
    
    # Get new wavenumbers
    klas[i] = np.sqrt(1-ope**2)
    kram[i] = -np.sqrt(freqs[1]**2-ope**2)
    kl = klas[i]-kram[i]

    # Coupling factor function
    def coupling_res(omlw,klw):
      zeta = omlw/(klw*vth*np.sqrt(2))
      dZ = dZfun(zeta)
      dZi = dZfun(zeta*np.sqrt(mime))
      sus = -n[i]/(2*klw**2*vth**2)
      Fij = np.imag(0.25*(1+sus*dZi)/(1+sus*dZ+sus*dZi))
      #Fij = np.imag(0.25/(1+sus*dZ))
      return Fij
    if noise:
      grres[i] = -2*coupling_res(freqst[2],wavenst[2])

    # Get coupling factor at raman frequency of interest
    gr[i] = -2*coupling_res(freqs[2],wavens[2])

  # Initialise intensity arrays
  normf = scc.m_e**2*scc.c**3*kvac**2/(scc.e**2*scc.mu_0)
  I1 = np.ones_like(x)
  I0 = np.ones_like(x)
  I1 *= I10/normf; I0 *= I00/normf
  I1 /= freqs[1]*abs(kram); I0 /= klas
  #I1[:-1] = 0.0; I0[1:] = 0.0
  I1n = copy.deepcopy(I1)

  # Get divergence of dr/dt
  xn = (x-xmin)*kvac
  dx = (xn[1]-xn[0])
  divlas = np.gradient(klas,dx)
  divram = np.gradient(kram/freqs[1],dx)

  # ODE evolution functions
  def Flas(xi,I0i):
    kl = np.interp(xi,xn,klas)
    gr0 = np.interp(xi,xn,gr)
    gr1 = np.interp(xi,xn,grres)
    I1i = np.interp(xi,xn,I1)
    I1in = np.interp(xi,xn,I1n)
    dla = np.interp(xi,xn,divlas)
    if noise:
      return -I0i/kl*(gr0*I1i+gr1*I1in)-dla*I0i/kl
    return -I0i/kl*(gr0*I1i)-dla*I0i/kl
  def Fram(xi,I1i):
    kr = np.interp(xi,xn,kram)
    gr0 = np.interp(xi,xn,gr)
    gr1 = np.interp(xi,xn,grres)
    I0i = np.interp(xi,xn,I0)
    I1in = np.interp(xi,xn,I1n)
    dra = np.interp(xi,xn,divram)
    if noise:
      return -(I0i/abs(kr)*(gr0*I1i+gr1*I1in)-dra*I1i*freqs[1]/abs(kr))
    return -(I0i/abs(kr)*(gr0*I1i)-dra*I1i*freqs[1]/abs(kr))

  # Evolve intensities across domain and iterate to convergence
  meps = np.finfo(np.float64).eps
  niter = 0; tol = 1e-10; conv = 1
  while (conv > tol and niter < 500):
    I0old = copy.deepcopy(I0)
    I1old = copy.deepcopy(I1)
    for i in range(points-1):
      res = solve_ivp(Flas,(xn[i],xn[i+1]),np.array([I0[i]]),method='RK23')
      I0[i+1] = np.maximum(res.y[0][-1],0)
      res = solve_ivp(Fram,(xn[i+1],xn[i]),np.array([I1[i+1]]),method='RK23')
      I1[i] = res.y[0][-1]
    niter += 1
    conv = np.sum(abs(I0-I0old)+abs(I1-I1old))
    #print(niter,conv)

  # Return stolen units
  I1 *= freqs[1]*abs(kram); I0 *= klas
  I0 *= normf*1e-4; I1 *= normf*1e-4
      
  if return_all:
    return x,n,I0,I1
  else:
    return abs(I1[0])

    
  
