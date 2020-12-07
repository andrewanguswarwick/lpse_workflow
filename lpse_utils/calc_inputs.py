import numpy as np
import write_files as wf
import scipy.constants as scc
from scipy.optimize import bisect

# 1D backscattered SRS LW frequency and wavelength calculation
def bsrs_lw_envelope(case,cells_per_wvl=30,no_thermal=False):
  # Extract relevant quantities from lpse class
  den_frac = case.plasmaFrequencyDensity
  if den_frac == None:
    print("Error: lpse_case.plasmaFrequencyDensity not specified.")
    return
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      Te = np.float64(i.physical.Te)*1.0e3
    elif isinstance(i,wf.light_control):
      lambda0 = np.float64(i.laser.wavelength)

  # Constants
  c = scc.c; e = scc.e; me = scc.m_e

  # Laser wavenumber in plasma
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

  # Get LW wavenumber by resonance matching and calculate frequency
  k_ek = k0 - ks
  omega_ek = np.sqrt(omega_pe**2 + 3*vth**2*k_ek**2)
  omega_s = np.sqrt(omega_pe**2+ks**2)
  if no_thermal:
    den_frac = omega_pe**2
  else:
    den_frac = omega_ek**2

  # Set lpse class attribute
  for i in case.setup_classes:
    if isinstance(i,wf.physical_parameters):
      i.lw.envelopeDensity = den_frac
      print(f'LW envelope density is: {i.lw.envelopeDensity:0.4f}')
    if isinstance(i,wf.gridding):
      k = np.array([k0,ks,k_ek])
      lams = lambda0/abs(k)
      print(lams)
      i.grid.nodes = int(round(cells_per_wvl*i.grid.sizes/np.min(lams)))+1
      print(f'Using {i.grid.nodes-1} cells.')

  return [omega0,omega_s,omega_ek,0]

# Calculates max spectral dt for avoiding artifacts
def spectral_dt(case,freqs,D=1,dt_frac=0.95):
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
    AA = auto_AA_range(case,D)

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
      print(f'Laser spectral dt is: {dts[0]:0.2e} ps')
      print(f'Raman spectral dt is: {dts[1]:0.2e} ps')
    elif isinstance(i,wf.lw_control):
      i.lw.spectral.dt = dts[2]
      print(f'LW spectral dt is: {dts[2]:0.2e} ps')
    elif isinstance(i,wf.iaw_control):
      i.iaw.spectral.dt = dts[3]
      print(f'IAW spectral dt is: {dts[3]:0.2e} ps')

def auto_AA_range(case,D=1):
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
  print(f'Automatically calcualted AA range is: {AA:0.3f}')

  return AA

