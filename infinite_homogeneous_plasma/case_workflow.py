#!/usr/bin/env python
# coding: utf-8

# # Initialisation

# In[ ]:


import write_files as wf
import ihp
import lpse_data as ld
import matplotlib.pyplot as plt
import numpy as np
from time import time as stopwatch

# Ipython magic features
# %load_ext autoreload
# %autoreload 2
  
# LPSE class
lpse = ld.lpse_case()
lpse.dfp = './data/lpse.' # Data file prefix
lpse.verbose = False # Show prints
lpse.np = 1 # Number of processors
lpse.bin = '/home/space/phrfqm/lpse-3.2.11/bin/lpse_cpu' # Binary


# # Case setup

# In[ ]:


jc = wf.job_control()
jc.version = '3.2.11' 
jc.seed = 1 # 0 for random, otherwise fixed seed
jc.resources.heartbeatInterval = 0.1 # minutes
lpse.add_class(jc)


# In[ ]:


gr = wf.gridding()
gr.grid.sizes = 1.0 # microns
gr.grid.nodes = 10
gr.grid.antiAliasing.isAutomatic = 'false'
gr.grid.antiAliasing.range = 0.333
lpse.add_class(gr)


# In[ ]:


cm = wf.components()
cm.laser.enable = 'true'
cm.raman.enable = 'true'
cm.lw.enable = 'true'
lpse.add_class(cm)


# In[ ]:


tc = wf.temporal_control()
tc.simulation.samplePeriod = 0.05 # ps
tc.simulation.time.end = 0.5 # ps
lpse.add_class(tc)


# In[ ]:


io = wf.io_control()
io.grid.downSampleFactors = 1 # Spatial
io.laser.save.E0.z = lpse.dfp + 'E0_z'
io.raman.save.E0.z = lpse.dfp + 'E1_z'
io.lw.save.pots = lpse.dfp + 'pots'
io.raman.save.S0.x = lpse.dfp + 'S1_x'
lpse.add_class(io)


# In[ ]:


pp = wf.physical_parameters()
pp.physical.Z = 1.0
pp.physical.Te = 4.5 # keV
pp.physical.Ti = 4.5 # keV
pp.physical.MiOverMe = 1836.15
pp.lw.envelopeDensity = 0.15
pp.densityProfile.shape = 'linear'
pp.densityProfile.geometry = 'cartesian'
pp.densityProfile.NminOverNc = 0.15
pp.densityProfile.NmaxOverNc = 0.15
pp.densityProfile.NminLocation = '-50 0 0'
pp.densityProfile.NmaxLocation = '50 0 0'
lpse.add_class(pp)


# In[ ]:


lc = wf.light_control()
lc.laser.wavelength = 0.351 # microns
lc.laser.pumpDepletion.SRS.enable = 'false'
lc.laser.evolution.Labc = 0 # microns
lc.laser.evolution.Loff = 0 # microns
lc.laser.solver = 'static'
lc.laser.evolution.solverOrder = 2
lc.laser.evolution.dtFraction = 0.95
lc.raman.sourceTerm.lw.enable = 'true'
lc.raman.evolution.Labc = 0
lc.raman.evolution.Loff = 0 
# fd solver
# lc.raman.solver = 'fd'
# lc.raman.evolution.solverOrder = 6
# lc.raman.evolution.dtFraction = 0.5
# spectral solver
lc.raman.solver = 'spectral'
lc.raman.spectral.dt = 2e-6
lpse.add_class(lc)


# In[ ]:


ls = wf.light_source()
ls.laser.nBeams = 1
ls.laser.intensity = ['5.0e+15'] # W/cm^2
ls.laser.phase = [0] # degrees
ls.laser.polarization = [90] # degrees
ls.laser.direction = ['1 0 0']
ls.laser.frequencyShift = [0]
ls.laser.group = [0]
ls.laser.evolution.source = ['min.x']
ls.laser.evolution.offset = ['0 0 0'] # microns
ls.laser.evolution.width = [0] # Half-width at 1/e of sgauss [um]
ls.laser.evolution.sgOrder = [4]
lpse.add_class(ls)


# In[ ]:


lwc = wf.lw_control()
lwc.lw.SRS.enable = 'true'
lwc.lw.spectral.dt = 2e-4 # ps
lwc.lw.Labc = 0 # microns
lwc.lw.noise.enable = 'true'
lwc.lw.noise.isCalculated = 'false'
lwc.lw.noise.amplitude = 1e-10
lwc.lw.collisionalDampingRate = 0.02
lwc.lw.__dict__['collisionalDampingRate.isCalculated'] = 'false'
lwc.lw.landauDamping.enable = 'false'
lwc.lw.kFilter.enable = 'true'
lpse.add_class(lwc)


# In[ ]:


ins = wf.instrumentation()
ins.metrics.enable = 'true'
ins.metrics.file = lpse.dfp + 'metrics'
ins.metrics.samplePeriod = 0.05 # ps
lpse.add_class(ins)


# # Calculate theoretical SRS growth rate

# In[ ]:


# Adjust temperature and density slightly to get better wavelength matching
eps = np.finfo(np.float64).eps
max_iter = 20; minints = 1
ihp.rhoT_adjust(lpse,tol=4*eps,max_iter=max_iter,minints=minints)


# In[ ]:


# Theory results
gamma, k, dfrac = ihp.srs_theory(lpse)


# # Adjust domain for wavelength matching

# In[ ]:


# Match domain size to wavelength integer multiples
dsize, maxwvls = ihp.wavelength_matching(lpse,k,tol=1e-6,max_iter=max_iter,minints=minints)


# In[ ]:


# Set class attributes
pp.lw.envelopeDensity = dfrac # Thermal correction to LW frequency
print(f'LW envelope density is now: {pp.lw.envelopeDensity:0.8f}')
lpse.add_class(pp)
cells_per_wvl = 50
gr.grid.sizes = dsize
gr.grid.nodes = maxwvls*cells_per_wvl+1
print(f'Using {gr.grid.nodes-1} cells.')
lpse.add_class(gr)


# # Run case and get LPSE SRS growth rate

# In[ ]:


t1 = stopwatch()
xdat,ydat,pfit = ihp.srs_growth_error(lpse,gamma)
t2 = stopwatch()
print(f'Time taken: {t2-t1:0.3f}')


# In[ ]:


# Check E field of waves
datt = lpse.fdat['E0_z']
plt.plot(datt['x'],np.real(datt['data'][0]))
plt.xlabel('x [um]')
plt.ylabel('E0')
plt.show()
datt = lpse.fdat['E1_z']
xdat = datt['x']
ydat = np.real(datt['data'][-1])
plt.plot(xdat,ydat)
plt.xlabel('x [um]')
plt.ylabel('E1')
plt.show()
datt = lpse.fdat['pots']
xdat = datt['x']
ydat = np.real(datt['data'][-1])
plt.plot(xdat,ydat)
plt.xlabel('x [um]')
plt.ylabel('E_EPW')
plt.show()


# In[ ]:


# LW absorption metrics
lpse.plot_metric('EPW_power_absorbed_by_LD',loglin=True)
lpse.plot_metric('EPW_power_absorbed_by_collisional_damping',loglin=True)
lpse.plot_metric('EPW_energy',loglin=True)

