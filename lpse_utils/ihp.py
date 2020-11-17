import numpy as np
import write_files as wf
import scipy.constants as scc
from scipy.optimize import bisect

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

  # Get LPSE SRS growth rate
  xdat = case.fdat['S1_x']['time']*1e-12
  ydat = np.real(case.fdat['S1_x']['data'])
  ydat = np.array([np.average(np.abs(ydat[i,ydat[i]<0]))\
                   for i in range(len(xdat))])
  ydat = np.log(ydat)
  p = np.polyfit(xdat[0:],ydat[0:],1)
  print(f'LPSE SRS growth rate is {p[0]/(c*kvac):0.3e}')
  print(f'LPSE relative error = {abs(gamma-(p[0]/(c*kvac)))/gamma:0.3%}')
  
  return xdat,ydat,p

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
  
def srs_theory(case):
  # Extract relevant quantities from lpse class
  for i in case.setup_classes:
    if isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)*1.0e-6
    elif isinstance(i,wf.physical_parameters):
      den_frac = np.float64(i.lw.envelopeDensity)
      Te = np.float64(i.physical.Te)*1e3
    elif isinstance(i,wf.light_source):
      I0 = np.float64(i.laser.intensity[0])*1.0e4

  # Get theory SRS growth rate (dimensionless units)
  # Constants
  c = scc.c; e = scc.e; pi = scc.pi
  me = scc.m_e; epsilon0 = scc.epsilon_0

  # Laser wavenumber in plasma
  kvac = 2*pi/lambda0
  #omega0 = np.sqrt(kvac**2*c**2)
  omega0 = 1.0
  omega_pe = np.sqrt(den_frac)*omega0
  #k0 = np.sqrt(omega0**2-omega_pe**2)/c
  k0 = np.sqrt(omega0**2-omega_pe**2)

  # Thermal velocity
  #Ek = Te*e
  Ek = Te*e/(me*c**2)
  #vth = np.sqrt(Ek/me)
  vth = np.sqrt(Ek)

  # Use bisect method to find k roots of SRS dispersion relation
  def bsrs(ks):
    nonlocal omega_pe, vth, k0, omega0, c
    omega_ek = np.sqrt(omega_pe**2 + 3*vth**2*(k0-ks)**2)
    #res = (omega_ek-omega0)**2-c**2*ks**2-omega_pe**2
    res = (omega_ek-omega0)**2-ks**2-omega_pe**2
    return res
  #ks = bisect(bsrs,-1e20,0) # Look for negative root for backscatter
  ks = bisect(bsrs,-10,0) # Look for negative root for backscatter  

  # Get LW wavenumber by frequency matching and calculate remaining frequencies
  k_ek = k0 - ks
  omega_ek = np.sqrt(omega_pe**2 + 3*vth**2*k_ek**2)
  #omega = np.sqrt(omega_pe**2 + c**2*ks**2)
  omega_s = np.sqrt(omega_pe**2 + ks**2)

  # Get quiver velocity and maximum growth rate
  I0star = I0*e**2/(epsilon0*c**5*me**2*kvac**2)
  #E0 = np.sqrt(2*I0/(c*epsilon0))
  E0 = np.sqrt(2*I0star/omega0**2)
  #vos = E0*e/(me*omega0)
  vos = E0
  gamma = k_ek*vos/4*np.sqrt(omega_pe**2/(omega_ek*(omega0-omega_ek)))
  #gamma = gamma*c*kvac # Reconvert for comparison
  print(f'Frequency matching error: {omega0-omega_s-omega_ek:0.3e}')
  print(f'Wavenumber matching error: {k0-ks-k_ek:0.3e}')
  print(f'Theory SRS growth rate = {gamma:0.3e}')

  return gamma, np.array([k_ek,ks,k0])

def wavelength_matching(case,k,tol,max_iter=1000):
  # Extract relevant quantities
  for i in case.setup_classes:
    if isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)*1.0e-6
  c = scc.c; pi = scc.pi
  kvac = 2*pi/lambda0

  # Get wavelengths
  lams = np.array([abs(2*pi/i) for i in k])
  print(lams)
  waves = len(lams)

  # Find domains size which is integer multiple of all
  reldiff = 1.0; n = 0
  ints = np.ones(waves,dtype=np.int64)
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
    print(f'iter: {n}; ints: {ints}; rel-err: {reldiff:0.3e}')
    if reldiff > tol:
      minval = np.argmin(vals)
      ints[minval] += 1
    n += 1
  if (n == max_iter):
    x = np.argmin(diffs)
    print(f'Tolerance not met, using min error at iter {x+1}.')
    reldiff = diffs[x]
    ints = ihist[x]
    vals = vhist[x]
  dsize = np.average(vals)/kvac*1e6 
  max_wvls = np.max(ints)
  print('')
  print(f'Final normalised values: {np.round(vals,3)}')
  print(f'Final wavelengths mismatch: {reldiff:0.3e}')
  print(f'Domain size: {dsize:0.3f} microns')
  print(f'Max wavelengths in domain: {max_wvls}')
  return dsize, max_wvls
