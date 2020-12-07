#!/usr/bin/env python
# coding: utf-8

# # Initialisation
import write_files as wf
import lpse_data as ld
import random_srs as rs
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle
import ihp
from functools import partial
from time import time as stopwatch
import calc_inputs as ci

# LPSE class
lpse = ld.lpse_case()
lpse.dfp = './data/lpse.' # Data file prefix
lpse.verbose = False # Show prints
lpse.np = 1 # Number of processors
lpse.bin = '/home/space/phrfqm/lpse-3.2.11/bin/lpse_cpu' # Binary

# # Base case setup
jc = wf.job_control()
jc.version = '3.2.11' 
jc.seed = 1 # 0 for random, otherwise fixed seed
jc.resources.heartbeatInterval = 0.1 # minutes
lpse.add_class(jc)

gr = wf.gridding()
gr.grid.sizes = 108 # microns
gr.grid.nodes = 4321
gr.grid.antiAliasing.range = 0.333
gr.grid.antiAliasing.isAutomatic = 'false'
lpse.add_class(gr)

cm = wf.components()
cm.laser.enable = 'true'
cm.raman.enable = 'true'
cm.lw.enable = 'true'
lpse.add_class(cm)

tc = wf.temporal_control()
tc.simulation.samplePeriod = 0.01 # ps
tc.simulation.time.end = 3 # ps
lpse.add_class(tc)

io = wf.io_control()
io.grid.downSampleFactors = 1 # Spatial
io.raman.save.S0.x = lpse.dfp + 'S1_x'
io.laser.save.E0.z = lpse.dfp + 'E0_z'
io.raman.save.E0.z = lpse.dfp + 'E1_z'
io.lw.save.pots = lpse.dfp + 'pots'
lpse.add_class(io)

pp = wf.physical_parameters()
pp.physical.Z = 3.5
pp.physical.Te = 4.5 # keV
pp.physical.Ti = 4.5 # keV
pp.physical.MiOverMe =  11866.0
pp.lw.envelopeDensity = 0.15
pp.densityProfile.shape = 'exp'
pp.densityProfile.geometry = 'cartesian'
pp.densityProfile.NminOverNc = 0.14
pp.densityProfile.NmaxOverNc = 0.17
pp.densityProfile.NminLocation = '-50 0 0'
pp.densityProfile.NmaxLocation = '50 0 0'
lpse.add_class(pp)

lc = wf.light_control()
lc.laser.wavelength = 0.351 # microns
lc.laser.pumpDepletion.SRS.enable = 'true'
lc.laser.evolution.Labc = 3 # microns
lc.laser.evolution.Loff = 1 # microns
lc.raman.sourceTerm.lw.enable = 'true'
lc.raman.evolution.Labc = 3
lc.raman.evolution.Loff = 1 
# fd solvers
# lc.laser.solver = 'fd'
# lc.laser.evolution.solverOrder = 4
# lc.laser.evolution.dtFraction = 0.95
# lc.raman.solver = 'fd'
# lc.raman.evolution.solverOrder = 2
# lc.raman.evolution.dtFraction = 0.95
# spectral solvers
lc.laser.solver = 'spectral'
lc.laser.spectral.dt = 2e-6 # ps
lc.raman.solver = 'spectral'
lc.raman.spectral.dt = 2e-6
lpse.add_class(lc)

ls = wf.light_source()
ls.laser.nBeams = 1
ls.laser.intensity = ['1.0e+14'] # W/cm^2
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

lwc = wf.lw_control()
lwc.lw.SRS.enable = 'true'
lwc.lw.spectral.dt = 2e-6 # ps
lwc.lw.Labc = 3 # microns
lwc.lw.noise.enable = 'true'
lwc.lw.noise.isCalculated = 'false'
lwc.lw.noise.amplitude = 0.015
lwc.lw.collisionalDampingRate = 0.1
lwc.lw.__dict__['collisionalDampingRate.isCalculated'] = 'true'
lwc.lw.landauDamping.enable = 'true'
lwc.lw.kFilter.enable = 'true'
lwc.lw.kFilter.scale = 1.2
lpse.add_class(lwc)

ins = wf.instrumentation()
ins.metrics.enable = 'true'
ins.metrics.file = lpse.dfp + 'metrics'
ins.metrics.samplePeriod = 0.01 # ps
lpse.add_class(ins)

# # List of density profile classes
# format ppxy where x = L_n, y = n_mid, and 1,2,3 = low, mid, high
pp22 = copy.deepcopy(pp)
pp12 = copy.deepcopy(pp)
pp12.densityProfile.NminOverNc = 0.13
pp12.densityProfile.NmaxOverNc = 0.18
pp21 = copy.deepcopy(pp)
pp21.lw.envelopeDensity = 0.12
pp21.densityProfile.NminOverNc = 0.11
pp21.densityProfile.NmaxOverNc = 0.13
pp23 = copy.deepcopy(pp)
pp23.lw.envelopeDensity = 0.20
pp23.densityProfile.NminOverNc = 0.18
pp23.densityProfile.NmaxOverNc = 0.22
pp32 = copy.deepcopy(pp)
pp32.densityProfile.NminOverNc = 0.14
pp32.densityProfile.NmaxOverNc = 0.16
dens = [pp12,pp21,pp22,pp23,pp32]
cdens = [0.15,0.12,0.15,0.20,0.15]
dlabs = ['Ln=300um; nmid=0.15','Ln=500um; nmid=0.12',\
         'Ln=500um; nmid=0.15','Ln=500um; nmid=0.20',\
         'Ln=1000um; nmid=0.15']

# # Run cases and get $<I_{srs}>$ curves
# GPy Opt training set
fresh_training = True
tavg = 2
cpus = 20
train = 20
Isrs0 = 8e10
cells_per_wvl = 30
if fresh_training:
  t0 = stopwatch()
  x0,y0 = rs.amp_par(lpse,dens,cdens,dlabs,tavg,Isrs0,\
                     cpus,train,cells_per_wvl)
  t1 = stopwatch()
  print(f'Time taken: {t1-t0:0.3f}s')
  with open('train.pickle', 'wb') as f:
    pickle.dump([x0,y0], f)
else:
  with open('train.pickle', 'rb') as f:
    x0,y0 = pickle.load(f)

# Isrs curves
fresh_results = True
Irange = np.logspace(14,16,20)
if fresh_results:
  t0 = stopwatch()
  isrs = rs.Isrs_dens(lpse,dens,cdens,dlabs,tavg=tavg,Isrs0=Isrs0,Irange=Irange,x0=x0,y0=y0,\
                      parallel=True,cpus=cpus,cells_per_wvl=cells_per_wvl)
  t1 = stopwatch()
  print(f'Time taken: {t1-t0:0.3f}s')
  with open('isrs.pickle', 'wb') as f:
    pickle.dump(isrs, f)
else:
  with open('isrs.pickle', 'rb') as f:
    isrs = pickle.load(f)
