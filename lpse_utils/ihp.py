import numpy as np
import write_files as wf
import scipy.constants as scc
from scipy.optimize import bisect,differential_evolution
import matplotlib.pyplot as plt
from functools import partial

def srs_growth_error(case,gamma):
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
  beta = lambda t: p[0]*t + p[1]
  plt.plot(xdat,ydat,label='LPSE')
  plt.plot(xdat,beta(xdat),label='Linear regression')
  plt.xlabel('Time')
  plt.ylabel('ln E1_max')
  plt.legend()
  plt.show()
  print(f'LPSE SRS growth rate is {p[0]:0.3e}')
  print(f'LPSE relative error = {(p[0]-gamma)/gamma:0.3%}')

  return xdat,ydat,p

def srs_theory(case,verbose=True):
  # Extract relevant quantities from lpse class
  for i in case.setup_classes:
    if isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)*1.0e-6
    elif isinstance(i,wf.physical_parameters):
      den_frac = np.float64(i.densityProfile.NminOverNc)
      Te = np.float64(i.physical.Te)*1e3
    elif isinstance(i,wf.light_source):
      I0 = np.float64(i.laser.intensity[0])*1.0e4

  # Get theory SRS growth rate (dimensionless units)
  # Constants
  c = scc.c; e = scc.e; pi = scc.pi
  me = scc.m_e; epsilon0 = scc.epsilon_0

  # Laser wavenumber in plasma
  kvac = 2*pi/lambda0
  omega0 = 1.0
  omega_pe = np.sqrt(den_frac)*omega0
  k0 = np.sqrt(omega0**2-omega_pe**2)

  # Thermal velocity
  Ek = Te*e/(me*c**2)
  vth = np.sqrt(Ek)

  # Use bisect method to find k roots of SRS dispersion relation
  def bsrs(ks):
    nonlocal omega_pe, vth, k0, omega0
    omega_ek = np.sqrt(omega_pe**2 + 3*vth**2*(k0-ks)**2)
    res = (omega_ek-omega0)**2-ks**2-omega_pe**2
    return res
  ks = bisect(bsrs,-10,0) # Look for negative root for backscatter  

  # Get LW wavenumber by frequency matching and calculate remaining frequencies
  k_ek = k0 - ks
  omega_ek = np.sqrt(omega_pe**2 + 3*vth**2*k_ek**2)
  omega_s = np.sqrt(omega_pe**2 + ks**2)

  # Get quiver velocity and maximum growth rate
  I0star = I0*e**2/(epsilon0*c**5*me**2*kvac**2)
  E0 = np.sqrt(2*I0star/omega0**2)
  vos = E0
  #vos = np.max(np.real(case.fdat['E0_z']['data'][0]))
  gamma = k_ek*vos/4*np.sqrt(omega_pe**2/(omega_ek*(omega0-omega_ek)))
  if verbose:
    print(f'Frequency matching error: {omega0-omega_s-omega_ek:0.3e}')
    print(f'Wavenumber matching error: {k0-ks-k_ek:0.3e}')
    print(f'Theory SRS growth rate = {gamma:0.3e}')

  return gamma, np.array([k_ek,ks,k0])

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

def rhoT_adjust(case,tol,max_iter,minints):
  # Get temperature and density starting points
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      den_frac = np.float64(i.densityProfile.NminOverNc)
      Te = np.float64(i.physical.Te)

  # Minimize wavelength mismatch around originals
  func = partial(rhoT_opt,case=case,tol=tol,\
      max_iter=max_iter,minints=minints)
  bnds = ((max(0.01,den_frac-0.005),min(0.24,den_frac+0.005)),\
          (max(0.1,Te-0.5),Te+0.5))
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
  rho, T = x
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      i.densityProfile.NminOverNc = rho
      i.physical.Te = T

  # Get wavenumbers
  gamma, k = srs_theory(case,verbose=False)

  # Get domain mismatch
  mism = wavelength_matching(case,k,tol=tol,\
      max_iter=max_iter,minints=minints,verbose=False,opt=True)
  return mism

# For running parallel convergence sims
def srs_convergence(args,case,gamma,option='nodes'):
  if option == 'nodes':
    for i in case.setup_classes:
      if isinstance(i,wf.gridding):
        i.grid.nodes = args[0]
  elif option == 'solverOrder':
    for i in case.setup_classes:
      if isinstance(i,wf.light_control):
        i.raman.evolution.solverOrder = args[0]
  return srs_growth_error(case,gamma)[-1]
  


