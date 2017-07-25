import numpy as np
import numpy.linalg as nl
import sisl
import tqdm
import sys
from mpi4py import MPI
import argparse
from itertools import permutations, product
from timeit import default_timer as timer

# define some useful functions
def hsk(dh,k=(0,0,0)):
    '''
    One way to speed up Hk and Sk generation
    '''
    k = np.asarray(k, np.float64) # this two conversion lines are from the sisl source  
    k.shape = (-1,)               # 
    phases = np.exp(-1j * np.dot(np.dot(np.dot(dh.rcell, k), dh.cell), dh.sc.sc_off.T)) # this generates the list of phases
           
    HKU = np.einsum('abc,c->ab',dh.hup,phases)
    HKD = np.einsum('abc,c->ab',dh.hdo,phases)
    SK  = np.einsum('abc,c->ab',dh.sov,phases)

    return HKU,HKD,SK

def make_contour(emin=-20,emax=0.0,enum=42,p=10):
    '''
    A simple routine to generate energy contours
    '''
    R  = (emax-emin)/2;
    z0 = (emax+emin)/2;
    ie = (np.exp(-np.linspace(0,p,enum))-1);
    ze = z0-R*np.exp(1.0j*ie*np.pi);
    we = -1.0j*R*np.pi*np.exp(1.0j*np.pi*ie);

    class ccont:
        #just an empty container class
        pass
    cont    = ccont();
    cont.R  = R;
    cont.z0 = z0;
    cont.ie = ie;
    cont.ze = ze;
    cont.we = we;
    return cont
#----------------------------------------------------------------------

start = timer()

# Some input parsing
parser = argparse.ArgumentParser()
parser.add_argument('--kset'    , dest = 'kset'   , default  = 2         , type=int  , help = 'k-space resolution of Jij calculation')
parser.add_argument('--eset'    , dest = 'eset'   , default  = 42        , type=int  , help = 'Number of energy points on the contour')
parser.add_argument('--eset-p'  , dest = 'esetp'  , default  = 10        , type=int  , help = 'Parameter tuning the distribution on the contour')
parser.add_argument('--input'   , dest = 'infile' , required = True                  , help = 'Input file name')
parser.add_argument('--output'  , dest = 'outfile', required = True                  , help = 'Output file name')
parser.add_argument('--Ebot'    , dest = 'Ebot'   , default  = -20.0     , type=float, help = 'Bottom energy of the contour')
parser.add_argument('--npairs'  , dest = 'npairs' , default  = 1         , type=int  , help = 'Number of unitcell pairs in each direction for Jij calculation')
parser.add_argument('--use-tqdm', dest = 'usetqdm', default  = False                 , help = 'Use tqdm for progressbars or not')
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
NUMK = args.kset # INPUT
kran = np.linspace(0,1,NUMK,endpoint=False)
kx,ky,kz = np.meshgrid(kran,kran,kran)
kset = np.array([kx.flatten(),ky.flatten(),kz.flatten()]).T
wk = 1/len(kset) # weight of a kpoint in BZ integral
kpcs = np.array_split(kset,size)
if 'k' in args.usetqdm:
    kpcs[root_node] = tqdm.tqdm(kpcs[root_node],desc='k loop')
#----------------------------------------------------------------------

# generate pairs
NPAIRS = args.npairs # INPUT
neibrange = np.arange(-NPAIRS,NPAIRS+1);
[xi,yi,zi] = np.meshgrid(neibrange,neibrange,neibrange);
neigh_uc_list = np.array([xi.flatten(),yi.flatten(),zi.flatten()]).T

pairs = [] # we are going to collect usefull informations regarding the pairs in this list
           # each pair is going to have a dict 

for uc in neigh_uc_list:    
    
    if np.allclose(uc,[0,0,0]):
        # in the middle unit cell only nonequivalent neighbors are defined
        atran = permutations(range(len(dh.atoms)),2)
    else:
        # if unit cells are further apart all pairs are calculated
        atran = product(range(len(dh.atoms)),repeat=2)
        
    for i,j in atran:
        pairs.append(dict(
        offset = uc,    # lattice vector offset between the unitcells the two atoms are
        aiij   = [i,j], # indecies of the atoms in the unitcell
        noij   = [dh.atoms[i].orbs,dh.atoms[j].orbs], # number of orbitals on the appropriate atoms
        slij   = [slice( *(lambda x:[min(x),max(x)+1])(dh.a2o(i,all=True)) ),  # slices for 
                  slice( *(lambda x:[min(x),max(x)+1])(dh.a2o(j,all=True)) )], # appropriate orbitals           
        rirj   = [dh.axyz()[i],dh.axyz()[j]], # real space vectors of atoms in the unit cell
        Rij    = np.dot(dh.cell,uc),          # real space distance vector between unit cells
        rij    = np.dot(dh.cell,uc)-dh.axyz()[i]+dh.axyz()[j], # real space vector between atoms
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
if (rank==root_node) and ('E' in args.usetqdm):
 eran = tqdm.tqdm(range(len(cont.ie)),desc='E loop')
else:
 eran = range(len(cont.ie))
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

# sampling the integrand on the contour
for e_i in eran:

    ze = cont.ze[e_i]

    # reset G-s
    for pair in pairs:

        noi , noj =pair['noij']
        pair['Guij'] =  np.zeros((noi,noj),dtype='complex128')
        pair['Gdji'] =  np.zeros((noj,noi),dtype='complex128')
        pair['Guij_tmp'] =  np.zeros((noi,noj),dtype='complex128')
        pair['Gdji_tmp'] =  np.zeros((noj,noi),dtype='complex128')

    # doing parallel BZ integral
    for k in kpcs[rank]:

        HKU,HKD,SK = hsk(dh,k)
        Gku = nl.inv((ze*SK-HKU))
        Gkd = nl.inv((ze*SK-HKD))

        for pair in pairs:
            phase=np.exp(1j*np.dot(np.dot(dh.rcell,k),pair['Rij']))
            si,sj=pair['slij']
            # Fourier transform to real space
            pair['Guij_tmp'] +=  Gku[si,sj]*phase*wk # ij gets exp(+i k R) 
            pair['Gdji_tmp'] +=  Gkd[sj,si]/phase*wk # ji gets exp(-i k R)

    # summ reduce partial results of mpi nodes
    for pair in pairs:
        comm.Reduce(pair['Guij_tmp'],pair['Guij'],root=root_node)
        comm.Reduce(pair['Gdji_tmp'],pair['Gdji'],root=root_node)

    if rank==root_node:
        # The Szunyogh-Lichtenstein formula
        for pair in pairs:
            i,j = pair['aiij']
            pair['Jijz'].append( 
                         np.trace(np.dot(
                         np.dot(Hs[i],pair['Guij']),
                         np.dot(Hs[j],pair['Gdji'])
                              )))

#----------------------------------------------------------------------

# evaluation of the contour integral on the root node
# and saveing output of the calculation
if rank==root_node:
    for pair in pairs:
        pair['Jijz'] = np.array(pair['Jijz']) 
        pair['Jij']  = np.trapz(np.imag(pair['Jijz']*cont.we)/(2*np.pi),cont.ie)
    end = timer()
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



