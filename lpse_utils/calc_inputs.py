import numpy as np
import write_files as wf
import scipy.constants as scc
from scipy.optimize import bisect,minimize,newton
from scipy.special import wofz,kn,erfc
from functools import partial

# 1D backscattered SRS LW frequency and wavelength calculation
def bsrs_lw_envelope(case,cells_per_wvl=30,verbose=False,return_all=False,dispfun=False,relativistic=True):
  # Extract relevant quantities from lpse class
  den_frac0 = case.plasmaFrequencyDensity
  if den_frac0 == None:
    print("Error: lpse_case.plasmaFrequencyDensity not specified.")
    return
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      Te = np.float64(i.physical.Te)*1.0e3
      mime = i.physical.MiOverMe
    elif isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)
    elif isinstance(i,wf.light_source):
      I0 = np.float64(i.laser.intensity[0])*1.0e4
      I1 = np.float64(i.raman.intensity[0])*1.0e4

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
  omega_pe = np.sqrt(den_frac0)*omega0
  k0 = np.sqrt(omega0**2-omega_pe**2)
  dby = vth/omega_pe

  # Use bisect method to find k roots of SRS dispersion relation
  kfac = 3.0
  def bsrs(ks):
    omega_ek = np.sqrt(omega_pe**2 + kfac*vth**2*(k0-ks)**2)
    K = (k0-ks)*dby
    res = (omega_ek-omega0)**2 - ks**2 - omega_pe**2
    return res
  ks = bisect(bsrs,-5,0) # Look for negative root for backscatter  

  # Get LW wavenumber by frequency matching and calculate remaining frequencies
  k_ek = k0 - ks
  omega_ek = np.sqrt(omega_pe**2 + kfac*vth**2*k_ek**2)
  omega_s = np.sqrt(omega_pe**2 + ks**2)

  # Get Landau damping rate
  dk = dby*k_ek
  LD = np.sqrt(pi/8)/dk**3*omega_ek*np.exp(-0.5*omega_ek**2/(k_ek*vth)**2)/2
  if relativistic:
    a = 1/(vth**2*np.sqrt(1-omega_ek**2/k_ek**2))
    LD = 0.25/dk**3*omega_ek*vth**5*a**3*(kn(0,a)+kn(2,a))*np.exp(1/vth**2)/(1+vth**2)/2

  # Refine analytic calculations with plasma dispersion function
  if dispfun:
    # Define calculation of permittivity using dispersion function
    def Zfun(zeta):
      return 1j*np.sqrt(np.pi)*wofz(zeta)
    def dZfun(zeta):
      return -2*(1+zeta*Zfun(zeta))
    def perm(omega,k):
      re,im = omega
      omega = re + 1j*im
      K = k*dby
      zeta = (omega/omega_pe)/(np.sqrt(2)*K)
      #return abs(1 - dZfun(zeta)/(2*K**2))
      isus = -dZfun(zeta*np.sqrt(mime))/(2*K**2)
      esus = -dZfun(zeta)/(2*K**2)
      return abs(1+esus+isus)

    # Initial omega and permittivity
    if verbose:
      print('Initial')
      print(f'Omega_ek: {omega_ek-1j*LD}')
      print(f'Permittivity: {abs(perm([omega_ek,-LD],k_ek))}\n')

    # Find permittivity root, update k_ek by srs resonance and iterate
    conv = 1; tol = 1e-7; n = 1; om = [omega_ek,-LD]
    while conv > tol and n < 10:
      res = minimize(perm,om,args=(k_ek,))
      om = res.x
      omega_s = 1-om[0]
      ks = -np.sqrt(omega_s**2-omega_pe**2)
      k_ek = k0 - ks
      conv = perm(om,k_ek)
      n += 1
    omega_ek = om[0]; LD = -om[1]

    # Final omega and permittivity
    if verbose:
      print('Final')
      print(f'Omega_ek: {omega_ek-1j*LD}')
      print(f'Permittivity: {abs(perm(om,k_ek))}\n')

  # Set new envelope density
  den_frac = omega_ek**2

  # Set lpse class attribute
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      i.lw.envelopeDensity = den_frac
      if verbose:
        print(f'LW envelope density is: {i.lw.envelopeDensity:0.4f}')
    if isinstance(i,wf.gridding):
      k = np.array([k0,ks,k_ek])
      lams = lambda0/abs(k)
      i.grid.nodes = int(round(cells_per_wvl*i.grid.sizes/np.min(lams)))+1
      if verbose:
        print(f'Using {i.grid.nodes-1} cells.')

  # Set initial perturbation
  for i in case.setup_classes:
    if isinstance(i,wf.initial_perturbation):
      if "E1" in i.initialPerturbation.field:
        ki = ks
        oi = omega_s
        Ii = I1
      elif "E0" in i.initialPerturbation.field:
        ki = k0 
        oi = omega0
        Ii = I0
      i.initialPerturbation.wavelength = abs(2*pi/(ki*kvac))
      Istar = Ii/c**3*mu0/(me*oi*kvac*1e6/e)**2
      Evac = np.sqrt(2*Istar)
      den_frac1 = (omega_pe/oi)**2
      infla = 1/np.sqrt(np.sqrt(1-den_frac1))
      i.initialPerturbation.amplitude = Evac*infla
      
  if return_all:
    return [omega0,omega_s,omega_ek],[k0,ks,k_ek],kvac*1e6,vth,dby,LD
  else:
    return [omega0,omega_s,omega_ek]

# Calculates max spectral dt for avoiding artifacts
def spectral_dt(case,freqs,D=1,dt_frac=0.95,verbose=False):
  # Extract relevant quantities
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      Te = np.float64(i.physical.Te)*1.0e3
      den_frac = i.lw.envelopeDensity
    elif isinstance(i,wf.gridding):
      gsize = i.grid.sizes*1e-6
      nodes = i.grid.nodes
      autoAA = i.grid.antiAliasing.isAutomatic
      if autoAA == 'true':
        autoAA = True
      else:
        autoAA = False
        AA = i.grid.antiAliasing.range
        if verbose:
          print(f'AA range is: {AA:0.3f}')
    elif isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)*1.0e-6

  # Constants
  c = scc.c; e = scc.e; me = scc.m_e; pi = scc.pi

  # Maximum wavenumber on grid
  deltax = gsize/(nodes-1)
  kmax = np.sqrt(D)*pi/deltax

  # If auto AA range perform calculation
  if autoAA:
    AA = auto_AA_range(case,D,verbose)

  # Max dt for each wave
  waves = len(freqs)
  dts = np.zeros(waves)
  for i in range(waves):
    dts[i] = 8*freqs[i]*deltax**2/(lambda0*c*D*(1-AA)**2)*1e12

  # Apply dt cushion and set lpse class attributes
  dts *= dt_frac
  for i in case.setup_classes:
    if isinstance(i,wf.light_control):
      i.laser.spectral.dt = dts[0]
      i.raman.spectral.dt = dts[1]
      if verbose:
        print(f'Laser spectral dt is: {dts[0]:0.2e} ps')
        print(f'Raman spectral dt is: {dts[1]:0.2e} ps')
    elif isinstance(i,wf.lw_control):
      i.lw.spectral.dt = dts[2]
      if verbose:
        print(f'LW spectral dt is: {dts[2]:0.2e} ps')

def auto_AA_range(case,D=1,verbose=False):
  # Extract relevant quantities
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      Te = np.float64(i.physical.Te)*1.0e3
      den_frac = i.lw.envelopeDensity
    elif isinstance(i,wf.gridding):
      gsize = i.grid.sizes*1e-6
      nodes = i.grid.nodes

  # Constants
  c = scc.c; e = scc.e; me = scc.m_e; pi = scc.pi

  # Maximum wavenumber on grid
  deltax = gsize/(nodes-1)
  kmax = np.sqrt(D)*pi/deltax

  # Thermal velocity
  Ek = Te*e/(me*c**2)
  vth = np.sqrt(Ek)

  # Ghost wavenumber size approximation
  K0 = kmax*vth*np.sqrt(1./(1.-den_frac))
  AA = K0/kmax
  if verbose:
    print(f'Automatically calcualted AA range is: {AA:0.3f}')

  return AA

