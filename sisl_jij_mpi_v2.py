import numpy as np
import numpy.linalg as nl
import sisl
import tqdm
import sys
from mpi4py import MPI
import argparse
from itertools import permutations, product
from timeit import default_timer as timer
from scipy.special import roots_legendre
from usful import hsk,make_kset,make_contour,make_atran

#----------------------------------------------------------------------

start = timer()

# Some input parsing
parser = argparse.ArgumentParser()
parser.add_argument('--kset'    , dest = 'kset'   , default  = 2         , type=int  , help = 'k-space resolution of Jij calculation')
parser.add_argument('--kdirs'   , dest = 'kdirs'  , default  = 'xyz'                 , help = 'Definition of k-space dimensionality')
parser.add_argument('--eset'    , dest = 'eset'   , default  = 42        , type=int  , help = 'Number of energy points on the contour')
parser.add_argument('--eset-p'  , dest = 'esetp'  , default  = 10        , type=int  , help = 'Parameter tuning the distribution on the contour')
parser.add_argument('--input'   , dest = 'infile' , required = True                  , help = 'Input file name')
parser.add_argument('--output'  , dest = 'outfile', required = True                  , help = 'Output file name')
parser.add_argument('--Ebot'    , dest = 'Ebot'   , default  = -20.0     , type=float, help = 'Bottom energy of the contour')
parser.add_argument('--npairs'  , dest = 'npairs' , default  = 1         , type=int  , help = 'Number of unitcell pairs in each direction for Jij calculation')
parser.add_argument('--adirs'   , dest = 'adirs'  , default  = False                 , help = 'Definition of pair directions')
parser.add_argument('--use-tqdm', dest = 'usetqdm', default  = 'not'                 , help = 'Use tqdm for progressbars or not')
parser.add_argument('--cutoff'  , dest = 'cutoff' , default  = 100.0     , type=float, help = 'Real space cutoff for pair generation in Angs')
args = parser.parse_args()
#----------------------------------------------------------------------

# MPI init
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root_node = 0
if rank == root_node:
    print('Number of nodes in the parallel cluster: ',size)
#----------------------------------------------------------------------

# importing the necessary structures from SIESTA output
dat = sisl.get_sile(args.infile)
dh  = dat.read_hamiltonian()
# update datastructure of the hamiltonian 
# this is needed for quick Hk building
dh.hup = dh.tocsr(0).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')
dh.hdo = dh.tocsr(1).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')
dh.sov = dh.tocsr(2).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')
#----------------------------------------------------------------------

# generate k space sampling
kset=make_kset(dirs=args.kdirs,NUMK=args.kset)
wk = 1/len(kset) # weight of a kpoint in BZ integral
kpcs = np.array_split(kset,size)
if 'k' in args.usetqdm:
    kpcs[root_node] = tqdm.tqdm(kpcs[root_node],desc='k loop')
#----------------------------------------------------------------------
# generate pairs
args.adirs = args.adirs if args.adirs else args.kdirs # Take pair directions for k directions if adirs is not set
atran=make_atran(len(dh.atoms),args.adirs,dist=args.npairs) # definition of pairs in terms of integer coordinates refering to unicell distances and atomic positions       
pairs=[]
for i,j,uc in atran:
    if nl.norm(np.dot(uc,dh.cell)) < args.cutoff:
        pairs.append(dict(
        offset = uc,    # lattice vector offset between the unitcells the two atoms are
        aiij   = [i,j], # indecies of the atoms in the unitcell
        noij   = [dh.atoms.orbitals[i],dh.atoms.orbitals[j]], # number of orbitals on the appropriate atoms
        slij   = [slice( *(lambda x:[min(x),max(x)+1])(dh.a2o(i,all=True)) ),  # slices for 
                  slice( *(lambda x:[min(x),max(x)+1])(dh.a2o(j,all=True)) )], # appropriate orbitals           
        rirj   = [dh.axyz()[i],dh.axyz()[j]], # real space vectors of atoms in the unit cell
        Rij    = np.dot(uc,dh.cell),          # real space distance vector between unit cells
        rij    = np.dot(uc,dh.cell)-dh.axyz()[i]+dh.axyz()[j], # real space vector between atoms
        Jijz   = [], # in this empty list are we going to gather the integrad of the energy integral
        Jij    = 0   # the final results of the calculation are going to be here on the root node
            ))

if rank == root_node:
    print('Number of pairs beeing calculated: ',len(pairs))
#----------------------------------------------------------------------

# make energy contour 
# we are working in eV now  !
# and sisil shifts E_F to 0 !
cont = make_contour(emin=args.Ebot,enum=args.eset,p=args.esetp)
eran = cont.ze
#----------------------------------------------------------------------

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

#----------------------------------------------------------------------

# sampling the integrand on the contour and the BZ

for pair in pairs:
    noi , noj =pair['noij']
    pair['Guij'] =  np.zeros((args.eset,noi,noj),dtype='complex128')
    pair['Gdji'] =  np.zeros((args.eset,noj,noi),dtype='complex128')
    pair['Guij_tmp'] =  np.zeros((args.eset,noi,noj),dtype='complex128')
    pair['Gdji_tmp'] =  np.zeros((args.eset,noj,noi),dtype='complex128')


for k in kpcs[rank]:
    HKU,HKD,SK = hsk(dh,k)
    Gku = nl.inv(SK*eran.reshape(args.eset,1,1)-HKU)
    Gkd = nl.inv(SK*eran.reshape(args.eset,1,1)-HKD)
    for pair in pairs:
        phase=np.exp(1j*np.dot(np.dot(k,dh.rcell),pair['Rij']))
        pair['Guij_tmp'] += Gku*phase*wk
        pair['Gdji_tmp'] += Gkd/phase*wk

# summ reduce partial results of mpi nodes
for pair in pairs:
    comm.Reduce(pair['Guij_tmp'],pair['Guij'],root=root_node)
    comm.Reduce(pair['Gdji_tmp'],pair['Gdji'],root=root_node)

    
if rank==root_node:
    for pair in pairs:
        i,j = pair['aiij']
        # The Szunyogh-Lichtenstein formula
        pair['Jijz']=np.trace((Hs[i] @ pair['Guij'] ) @ (Hs[j] @ pair['Gdji'] ),axis1=1,axis2=2)
        # evaluation of the contour integral
        pair['Jij']  = np.trapz(np.imag(pair['Jijz']*cont.we)/(2*np.pi))
    end = timer()        
#----------------------------------------------------------------------
# and saveing output of the calculation
    np.savetxt(args.outfile,
               np.array([ [nl.norm(p['rij']),
                           p['Jij']*sisl.unit_convert('eV','Ry')*1000]+
                           p['aiij']+list(p['offset'])+list(p['rij'])                         
                        for p in pairs],
                        dtype=object),
               header=str(args)+
                      '\nnumber of cores = '+str(size)+
                      '\ntime of calculation = '+str(end-start)+
                      '\nnorm(rij),Jij[mRy],aiij,offset,rij',
               fmt="%s")


