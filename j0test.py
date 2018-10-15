import sisl
import tqdm

import numpy as np
import numpy.linalg as nl

from scipy.special import roots_legendre
from usful import *

#a simple way of getting argument parser out of the way.. but in a gracefull manner
class args: pass
args.kset=10
args.eset=42
args.esetp=10000
args.infile='picturedat/Ni/SIESTA/Ni.nc'
args.outfile='/dev/null'
args.Ebot=-20 # based on the inspection of the spectrum

# importing the necessary structures from SIESTA output
dat = sisl.get_sile(args.infile)
dh  = dat.read_hamiltonian()
# update datastructure of the hamiltonian 
# this is needed for quick Hk building
dh.hup = dh.tocsr(0).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')
dh.hdo = dh.tocsr(1).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')
dh.sov = dh.tocsr(2).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')
#----------------------------------------------------------------------
# make energy contour 
# we are working in eV now  !
# and sisil shifts E_F to 0 !
cont = make_contour(emin=args.Ebot,enum=args.eset,p=args.esetp)
eran = cont.ze

# generating onsite matrix and overalp elements of all the atoms in the unitcell
# onsite of the origin supercell
orig_indx=np.arange(0,dh.no)+dh.sc_index([0,0,0])*dh.no
# spin up
uc_up    = dh.tocsr(dh.UP   )[:,orig_indx].toarray() 
# spin down
uc_down  = dh.tocsr(dh.DOWN )[:,orig_indx].toarray() 
Hs=[]
# get number of atoms in the unit cell
for i in range(len(dh.atoms)):
    at_indx=dh.a2o(i,all=True)
    Hs.append(
            uc_up[:,at_indx][at_indx,:]-
          uc_down[:,at_indx][at_indx,:]
         )
# generate k space sampling
kset = make_kset(NUMK=args.kset)
wk = 1/len(kset) # weight of a kpoint in BZ integral
kpcs=kset
j0z=[]
for ze in tqdm.tqdm(eran):
    j0z_tmp=0.0j
    for k in kset:
        HKU,HKD,SK=hsk(dh,k=np.array(k))
        Gku=nl.inv(ze*SK-HKU)
        Gkd=nl.inv(ze*SK-HKD)
        j0z_tmp+=np.trace(Hs[0] @ Gku @ Hs[0] @ Gkd)*wk
    j0z.append(j0z_tmp)
    
j0z=np.array(j0z)
J0=np.trapz(np.imag(j0z*cont.we)/(2*np.pi))*1000*sisl.unit_convert('eV','Ry')

print(args.kset,J0)
