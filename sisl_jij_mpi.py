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

def make_contour(emin=-20,emax=0.0,enum=42,p=150):
    '''
    A more sophisticated contour generator
    '''
    
    x,wl = roots_legendre(enum)
    R  = (emax-emin)/2
    z0 = (emax+emin)/2
    y1 = -np.log(1+np.pi*p)
    y2 = 0

    y   = (y2-y1)/2*x+(y2+y1)/2
    phi = (np.exp(-y)-1)/p
    ze  = z0+R*np.exp(1j*phi)
    we  = -(y2-y1)/2*np.exp(-y)/p*1j*(ze-z0)*wl
    
    class ccont:
        #just an empty container class
        pass
    cont    = ccont()
    cont.R  = R
    cont.z0 = z0
    cont.ze = ze
    cont.we = we
    cont.enum = enum
        
    return cont

def make_kset(dirs='xyz',NUMK=20):
    '''
    Simple k-grid generator. Depending on the value of the dirs
    argument k sampling in 1,2 or 3 dimensions is generated.
    If dirs argument does not contain either of x,y or z
    a kset of a single k-pont at the origin is returend.
    '''
    if not(sum([d in dirs for d in 'xyz'])):
        return array([[0,0,0]])
        
    kran=len(dirs)*[np.linspace(0,1,NUMK,endpoint=False)]
    mg=meshgrid(*kran)
    dirsdict=dict()
    
    for d in enumerate(dirs):
        dirsdict[d[1]]=mg[d[0]].flatten()
    for d in 'xyz':
        if not(d in dirs):
            dirsdict[d]=0*dirsdict[dirs[0]]
    kset = np.array([dirsdict[d] for d in 'xyz']).T

    return kset

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
parser.add_argument('--kdirs'   , dest = 'kdirs'  , default  = 'xyz'                 , help = 'Definition of k-space dimensionality')
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
NPAIRS = args.npairs # INPUT
if NPAIRS == 0:
    neibrange = np.arange(0,2);
    [xi,yi,zi] = np.meshgrid(neibrange,0,0);
else:
    neibrange = np.arange(-NPAIRS,NPAIRS+1);
    [xi,yi,zi] = np.meshgrid(neibrange,neibrange,neibrange);
neigh_uc_list = np.array([xi.flatten(),yi.flatten(),zi.flatten()]).T

pairs = [] # we are going to collect usefull informations regarding the pairs in this list
           # each pair is going to have a dict 

for uc in neigh_uc_list:    
    uc=np.array(uc)
    uc.shape=(-1,)   
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
if (rank==root_node) and ('E' in args.usetqdm):
 eran = tqdm.tqdm(cont.ze,desc='E loop')
else:
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

# sampling the integrand on the contour
for ze in eran:

    # reset G-s
    for pair in pairs:

        noi , noj =pair['noij']
        pair['Guij'] =  np.zeros((noi,noj),dtype='complex128')
        pair['Gdji'] =  np.zeros((noj,noi),dtype='complex128')
        pair['Guij_tmp'] =  np.zeros((noi,noj),dtype='complex128')
        pair['Gdji_tmp'] =  np.zeros((noj,noi),dtype='complex128')

    # doing parallel BZ integral
    for k in kpcs[rank]:
        k=np.array(k)
        k.shape=(-1,)
        HKU,HKD,SK = hsk(dh,k)
        Gku = nl.inv((ze*SK-HKU))
        Gkd = nl.inv((ze*SK-HKD))

        for pair in pairs:
            phase=np.exp(1j*np.dot(np.dot(k,dh.rcell),pair['Rij']))
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
        pair['Jij']  = np.trapz(np.imag(pair['Jijz']*cont.we)/(2*np.pi))
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



