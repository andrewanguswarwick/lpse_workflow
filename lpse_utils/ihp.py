import numpy as np
import write_files as wf
import calc_inputs as ci
import scipy.constants as scc
from scipy.optimize import bisect,differential_evolution,minimize
import matplotlib.pyplot as plt
from functools import partial
import copy

def srs_growth_error(case,gamma,gamma0,ld):
  # Extract relevant quantities
  for i in case.setup_classes:
    if isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)*1.0e-6
  c = scc.c; pi = scc.pi
  kvac = 2*pi/lambda0

  # Write lpse.parms, run case and extract fields data
  case.write()
  case.run() 
  case.metrics()
  case.fields()

  # Get LPSE SRS growth rate and plot regression
  xdat = case.mdat['time']*1e-12*c*kvac
  ydat = case.mdat['E1_max']
  eps = np.finfo(np.float64).eps
  ydat = np.array([np.log(max(i,eps)) for i in ydat])
  xlen = len(xdat)
  p = np.polyfit(xdat[xlen//2:],ydat[xlen//2:],1)
  #p = np.polyfit(xdat[1:4],ydat[1:4],1)
  beta = lambda t: p[0]*t + p[1]
  plt.plot(xdat,ydat,label='LPSE')
  plt.plot(xdat,beta(xdat),label='Linear regression')
  plt.xlabel('Time')
  plt.ylabel('ln E1_max')
  plt.legend()
  plt.show()

  print(f'LPSE SRS growth rate is {p[0]:0.3e}')
  print(f'Undamped theory SRS growth rate is {gamma0:0.3e}')
  if ld:
    print(f'Landau damped theory SRS growth rate is {gamma:0.3e}')
    print(f'LPSE relative error = {(p[0]-gamma)/gamma:0.3%}')
  else:
    print(f'LPSE relative error = {(p[0]-gamma0)/gamma0:0.3%}')

  return p

def srs_theory(case,verbose=True,inflation=True,dispfun=False,relativistic=True):
  # Get case quantities
  den_frac0 = case.plasmaFrequencyDensity
  for i in case.setup_classes:
    if isinstance(i,wf.light_source):
      I0 = np.float64(i.laser.intensity[0])*1.0e4

  # Get resonant frequencies and wavenumbers
  freqs,wavens,kvac,vth,dby,LD = ci.bsrs_lw_envelope(case,30,verbose,True,dispfun,relativistic)
  omega0,omega_s,omega_ek = freqs
  k0,ks,k_ek = wavens
  omega_pe = vth/dby

  # Get quiver velocity and maximum growth rate
  E0 = np.sqrt(2*I0/scc.c/scc.epsilon_0)
  E0 *= scc.e/(scc.m_e*scc.c**2*kvac)
  if inflation:
    infla = 1/np.sqrt(np.sqrt(1-den_frac0))
    E0 *= infla
  vos = E0
  gamma0 = k_ek*vos/4*np.sqrt(omega_pe**2/(omega_ek*(omega0-omega_ek)))
  if verbose:
    print(np.array([wavens])*kvac/1e6)
    print(wavens,kvac)

  # Apply LD correction
  dk = dby*k_ek
  gamma = gamma0*np.sqrt(1+(0.5*LD/gamma0)**2)-LD/2
  if verbose:
    print(f'Frequency matching error: {omega0-omega_s-omega_ek:0.3e}')
    print(f'Wavenumber matching error: {k0-ks-k_ek:0.3e}')
    print(f'Theory undamped SRS growth rate = {gamma0:0.3e}')
    print(f'Theory Landau damped SRS growth rate = {gamma:0.3e}')

  return gamma, gamma0, np.array([k0,ks,k_ek]), dk

def wavelength_matching(case,k,tol,max_iter=100,minints=2,cells_per_wvl=30,\
                        verbose=True,opt=False):
  # Extract relevant quantities
  for i in case.setup_classes:
    if isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)*1.0e-6
  c = scc.c; pi = scc.pi
  kvac = 2*pi/lambda0

  # Get wavelengths
  lams = np.array([abs(2*pi/i) for i in k])
  waves = len(lams)

  # Find domains size which is integer multiple of all
  reldiff = 1.0; n = 0
  ints = minints*np.ones(waves,dtype=np.int64)
  min_lam = np.min(lams)
  diffs = np.empty(0)
  ihist = np.empty((0,waves),dtype=np.int64); vhist = np.empty((0,waves))
  while reldiff > tol and n < max_iter:
    vals = np.array([lams[i]*ints[i] for i in range(waves)])
    diff = np.sum([abs(vals[i]-vals[(i+1) % 3]) for i in range(waves)])
    reldiff = diff/min_lam
    diffs = np.append(diffs,reldiff)
    ihist = np.append(ihist,[ints],axis=0)
    vhist = np.append(vhist,[vals],axis=0)
    if verbose:
      print(f'iter: {n}; ints: {ints}; rel-err: {reldiff:0.3e}')
    if reldiff > tol:
      minval = np.argmin(vals)
      ints[minval] += 1
    n += 1

  # Check if terminated before max_iter
  if (n == max_iter):
    x = np.argmin(diffs)
    if verbose:
      print(f'Tolerance not met, using min error at iter {x}.')
    reldiff = diffs[x]
    ints = ihist[x]
    vals = vhist[x]

  # Get final results
  dsize = np.average(vals)/kvac*1e6 
  max_wvls = np.max(ints)
  if verbose:
    print('')
    print(f'Final normalised values: {np.round(vals,3)}')
    print(f'Final wavelengths mismatch: {reldiff:0.3e}')
    print(f'Domain size: {dsize:0.3f} microns')
    print(f'Max wavelengths in domain: {max_wvls}')

  # If adjusting rho and T only return wavelength mismatch
  # Else specify mesh
  if opt:
    return reldiff
  else:
    for i in case.setup_classes:
      if isinstance(i,wf.gridding):
        i.grid.sizes = dsize
        i.grid.nodes = max_wvls*cells_per_wvl+1
        print(f'Using {i.grid.nodes-1} cells.')

def rhoT_adjust(case,tol,max_iter,minints,dden=0.01,dT=0.5,infl=True,disp=False):
  global inflation, dispfun
  inflation = infl; dispfun = disp
  # Get temperature and density starting points
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      den_frac = np.float64(i.densityProfile.NminOverNc)
      Te = np.float64(i.physical.Te)

  # Minimize wavelength mismatch around originals
  func = partial(rhoT_opt,case=case,tol=tol,\
      max_iter=max_iter,minints=minints)
  bnds = ((max(0.01,den_frac-dden),min(0.24,den_frac+dden)),\
          (max(0.1,Te-dT),Te+dT))
  res = differential_evolution(func,tol=tol,bounds=bnds)

  # Set case parameters to optimum
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      i.densityProfile.NminOverNc = res.x[0]
      i.densityProfile.NmaxOverNc = res.x[0]
      i.physical.Te = res.x[1]
      i.physical.Ti = res.x[1]
  case.plasmaFrequencyDensity = res.x[0]

  print(f'Optimised rho and T: {np.round(res.x,3)}')

def rhoT_opt(x,case,tol,max_iter,minints):
  # Set temperature and density
  global inflation
  rho, T = x
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      i.densityProfile.NminOverNc = rho
      i.densityProfile.NmaxOverNc = rho
      i.physical.Te = T
  case.plasmaFrequencyDensity = rho

  # Get wavenumbers
  scrap,scrapp,k,scrappp = srs_theory(case,verbose=False,inflation=inflation,dispfun=dispfun)

  # Get domain mismatch
  mism = wavelength_matching(case,k,tol=tol,\
      max_iter=max_iter,minints=minints,verbose=False,opt=True)
  return mism
  
def srs_theory_curve(case,zerotemp=False,points=100,I0=2e15,Te=3.5,lambda0=351e-9,\
                    disp=False,infl=False,verbose=False):
  ncase = copy.deepcopy(case)
  den_frac = np.linspace(0.01,0.24,points)
  gammas = np.empty(0)
  gamma0s = np.empty(0)
  keks = np.empty(0)
  LDs = np.empty(0)
  dks = np.empty(0)

  # Set class quantities
  for i in ncase.setup_classes:
    if isinstance(i,wf.light_source):
      i.laser.intensity = [I0]
    elif isinstance(i,wf.light_control):
      i.laser.wavelength = lambda0*1.0e6
    elif isinstance(i,wf.physical_parameters):
      i.physical.Te = Te

  # Get homogeneous SRS growth rates for each density
  for i in range(points):
    # Set density
    ncase.plasmaFrequencyDensity = den_frac[i]

    # Get theory predictions
    gamma,gamma0,wavens,dk = srs_theory(ncase,verbose=verbose,inflation=infl,dispfun=disp)

    # Append histories
    gammas = np.append(gammas,gamma)
    gamma0s = np.append(gamma0s,gamma0)
    dks = np.append(dks,dk)

  return den_frac, gammas, gamma0s, dks

def lw_freq_curve(case,dens,T):
  # Copy case
  ncase = copy.deepcopy(case)

  # Set T
  for i in ncase.setup_classes:
    if isinstance(i,wf.physical_parameters):
      i.physical.Te = T

  # Get LW freqs
  freqs = np.zeros_like(dens)
  for i in range(len(dens)):  
    ncase.plasmaFrequencyDensity = dens[i]
    freqs[i] = ci.bsrs_lw_envelope(ncase,verbose=False)[-1]

  return freqs

