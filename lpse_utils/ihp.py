import numpy as np
import write_files as wf
import calc_inputs as ci
import scipy.constants as scc
from scipy.optimize import bisect,differential_evolution,minimize
import matplotlib.pyplot as plt
from functools import partial

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

def srs_theory(case,verbose=True,pert = True,Isrs = 8.0e14):
  # Extract relevant quantities from lpse class
  den_frac = case.plasmaFrequencyDensity
  if den_frac == None:
    print("Error: lpse_case.plasmaFrequencyDensity not specified.")
    return
  for i in case.setup_classes:
    if isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)*1.0e-6
    elif isinstance(i,wf.physical_parameters):
      Te = np.float64(i.physical.Te)*1e3
    elif isinstance(i,wf.light_source):
      I0 = np.float64(i.laser.intensity[0])*1.0e4

  # Get theory SRS growth rate (dimensionless units)
  # Constants
  meps = np.finfo(np.float64).eps
  c = scc.c; e = scc.e; pi = scc.pi
  me = scc.m_e; epsilon0 = scc.epsilon_0
  mu0 = scc.mu_0

  # Thermal velocity
  Ek = 0.5*Te*e/(me*c**2)
  vth = np.sqrt(2*Ek)

  # Laser wavenumber in plasma
  kvac = 2*pi/lambda0
  omega0 = 1.0
  omega_pe = np.sqrt(den_frac)*omega0
  k0 = np.sqrt(omega0**2-omega_pe**2)

  # Use bisect method to find k roots of SRS dispersion relation
  kfac = 3.0
  def bsrs(ks):
    omega_ek = np.sqrt(omega_pe**2 + kfac*vth**2*(k0-ks)**2)
    res = (omega_ek-omega0)**2 - ks**2 - omega_pe**2
    return res
  ks = bisect(bsrs,-10,0) # Look for negative root for backscatter  

  # Get LW wavenumber by frequency matching and calculate remaining frequencies
  k_ek = k0 - ks
  omega_ek = np.sqrt(omega_pe**2 + kfac*vth**2*k_ek**2)
  omega_s = np.sqrt(omega_pe**2 + ks**2)

  # Get quiver velocity and maximum growth rate
  Istar = I0/c**3*mu0/(me*omega0*kvac/e)**2
  Evac = np.sqrt(2*Istar)
  infla = 1/np.sqrt(np.sqrt(1-den_frac))
  E0 = Evac*infla
  I0star = I0/c**3*mu0/(me*omega0*kvac/e)**2
  E00 = np.sqrt(2*I0star)
  vos = E0

  gamma0 = k_ek*vos/4*np.sqrt(omega_pe**2/(omega_ek*(omega0-omega_ek)))
  if verbose:
    print(vos/(omega0/k0))
    print(vos/(gamma0/vos))
    print(gamma0/omega_ek)
    print(gamma0/omega_s)
  
  def gammaf(gamma):
    res = gamma**2*(gamma**2-4*omega_ek**2+4*omega_ek*omega0)-omega_pe**2*k_ek**2*vos**2/4
    return res
  gamma00 = bisect(gammaf,0,1)  

  # Get Landau damping and apply correction
  debye = np.sqrt(2*Ek/omega_ek**2)
  dk = debye*k_ek
  LD = np.sqrt(pi/8)*omega_pe/dk**3*(1+1.5*dk**2)*np.exp(-1.5)*np.exp(-0.5/dk**2)
  gamma = gamma0*np.sqrt(1+(0.5*LD/gamma0)**2)-LD/2
  if verbose:
    print(f'Frequency matching error: {omega0-omega_s-omega_ek:0.3e}')
    print(f'Wavenumber matching error: {k0-ks-k_ek:0.3e}')
    print(f'Theory undamped SRS growth rate = {gamma0:0.3e}')
    print(f'Theory Landau damped SRS growth rate = {gamma:0.3e}')

  # Set initial perturbation
  for i in case.setup_classes:
    if isinstance(i,wf.initial_perturbation):
      i.initialPerturbation.wavelength = abs(2*pi/(ks*kvac)*1e6)
      Istar = Isrs/c**3*mu0/(me*omega_s*kvac/e)**2
      Evac = np.sqrt(2*Istar)
      den_frac1 = (omega_pe/omega_s)**2
      infla = 1/np.sqrt(np.sqrt(1-den_frac1))
      i.initialPerturbation.amplitude = Evac*infla

  return gamma, gamma0, np.array([k0,ks,k_ek])

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

def rhoT_adjust(case,tol,max_iter,minints,dden=0.05,dT=0.5):
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
  rho, T = x
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      i.densityProfile.NminOverNc = rho
      i.densityProfile.NmaxOverNc = rho
      i.physical.Te = T
  case.plasmaFrequencyDensity = rho

  # Get wavenumbers
  scrap,scrapp,k = srs_theory(case,verbose=False)

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
  
def srs_theory_curve(case,zerotemp=False):
  points = 100
  meps = np.finfo(np.float64).eps
  den_frac = np.linspace(1e-3,0.24,points)
  gammas = np.empty(0)
  gamma0s = np.empty(0)
  keks = np.empty(0)
  LDs = np.empty(0)
  dks = np.empty(0)
  I0 = 2.0e19
  lambda0 = 351.0e-9
  if zerotemp:
    Te = 0.01
  else:
    Te = 3500.0

  # Get theory SRS growth rate (dimensionless units)
  # Constants
  c = scc.c; e = scc.e; pi = scc.pi
  me = scc.m_e; epsilon0 = scc.epsilon_0
  for i in range(points):
    # Thermal velocity
    A = 0.5
    Ek = A*Te*e
    vth = np.sqrt(2*Ek/me)

    # Laser wavenumber in plasma
    kvac = 2*pi/lambda0
    omega0 = kvac*c
    omega_pe = np.sqrt(den_frac[i])*omega0
    k0 = np.sqrt(omega0**2-omega_pe**2)/c
    def em_dis(omega,k):
      res = omega**2 - c**2*k**2 - omega_pe**2/(1-3*vth**2*k**2/omega**2)
      return res
    light = partial(em_dis,omega0)
    k0 = bisect(light,0,1e20)

    # Use bisect method to find k roots of SRS dispersion relation
    cratio = 3
    def bohmgross(k_ek):
      nonlocal cratio, vth
      return np.sqrt(omega_pe**2 + cratio*vth**2*k_ek**2)
    def bsrs(k_ek):
      nonlocal omega_pe, k0, omega0
      omega_ek = bohmgross(k_ek)
      #res = (omega_ek-omega0)**2-c**2*(k_ek-k0)**2-omega_pe**2
      res = (omega_ek-omega0)**2-c**2*(k_ek-k0)**2-omega_pe**2/(1-3*vth**2*(k_ek-k0)**2/(omega_ek-omega0)**2)
      return res
    k_ek = bisect(bsrs,k0,2*k0) # Look for negative raman root for backscatter  
    kcheck = k0 + omega0/c*np.sqrt(1-2*omega_pe/omega0)

    # Get raman wavenumber by frequency matching and calculate remaining frequencies
    ks = k0 - k_ek
    omega_ek = bohmgross(k_ek)
    raman = partial(em_dis,k=ks)
    omega_s = bisect(raman,meps,1e20)
    #omega_s = np.sqrt(omega_pe**2 + c**2*ks**2)
    vp = omega_ek/k_ek
    vg = 3*vth**2/vp

    # Get quiver velocity and maximum growth rate
    E0 = np.sqrt(2*I0/(c*epsilon0))
    vos = E0*e/(me*omega0)
    gamma0 = k_ek*vos/4*omega_pe/np.sqrt(omega_ek*omega_s)

    # Get Landau damping and apply correction
    debye = vth/omega_pe
    debye = np.sqrt(Te*e/(omega_pe**2*me))
    dk = debye*k_ek
    LD = np.sqrt(pi/8)*omega_pe/dk**3*(1+1.5*dk**2)*np.exp(-1.5-0.5/dk**2)/2
    
    gamma = gamma0*np.sqrt(1+(0.5*LD/gamma0)**2)-0.5*LD
    gammas = np.append(gammas,gamma/omega0)
    gamma0s = np.append(gamma0s,gamma0/omega0)
    keks = np.append(keks,k_ek/kvac)
    LDs = np.append(LDs,LD/omega0)
    dks = np.append(dks,dk)
    #print(f'Frequency matching error: {omega0-omega_s-omega_ek:0.3e}')
    #print(f'Wavenumber matching error: {k0-ks-k_ek:0.3e}')
    #print(f'Theory undamped SRS growth rate = {gamma0:0.3e}')
    #print(f'Theory Landau damped SRS growth rate = {gamma:0.3e}')

  return den_frac, gammas, keks, gamma0s, LDs, dks

def lw_freq_curve(case,dens,T):
  # Set T
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      i.physical.Te = T

  # Get LW freqs
  freqs = np.zeros_like(dens)
  for i in range(len(dens)):  
    case.plasmaFrequencyDensity = dens[i]
    freqs[i] = ci.bsrs_lw_envelope(case,verbose=False)[-1]

  return freqs

