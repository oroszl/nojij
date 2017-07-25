import numpy as np
import numpy.linalg as nl
import sisl
import tqdm
import sys
from mpi4py import MPI
import argparse

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



# Some input parsing
parser = argparse.ArgumentParser()
parser.add_argument('--kset'  , dest = 'kset'   , default  = 2      , type=int  , help = 'k-space resolution of Jij calculation')
parser.add_argument('--eset'  , dest = 'eset'   , default  = 42     , type=int  , help = 'Number of energy points on the contour')
parser.add_argument('--eset-p', dest = 'esetp'  , default  = 10     , type=int  , help = 'Parameter tuning the distribution on the contour')
parser.add_argument('--input' , dest = 'infile' , required = True               , help = 'Input file name')
parser.add_argument('--Ebot'  , dest = 'Ebot'   , default  = -20.0  , type=float, help = 'Bottom energy of the contour')

args = parser.parse_args()
# MPI init
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root_node=0
if rank==root_node:
#    print('Number of nodes in the parallel cluster: ',size)
    print('cores: ',size)


# importing the necessary structures
dat = sisl.get_sile(args.infile)
dh  = dat.read_hamiltonian()

# update datastructure of the hamiltonian 
# this is needed for quick Hk building
dh.hup = dh.tocsr(0).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')
dh.hdo = dh.tocsr(1).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')
dh.sov = dh.tocsr(2).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1).astype('complex128')

# generate k space sampling
NUMK = args.kset # INPUT
kran = np.linspace(0,1,NUMK,endpoint=False)
kx,ky,kz = np.meshgrid(kran,kran,kran)
kset = np.array([kx.flatten(),ky.flatten(),kz.flatten()]).T
wk = 1/len(kset)
kpcs = np.array_split(kset,size)
#kpcs[root_node] = tqdm.tqdm(kpcs[root_node],desc='k loop')


# make energy contour 
# we are working in eV now  !
# and sisil shifts E_F to 0 !
cont = make_contour(emin=args.Ebot,enum=args.eset,p=args.esetp)
eran = range(len(cont.ie))
#if rank==root_node:
# eran = tqdm.tqdm(range(len(cont.ie)),desc='Energy loop')
#else:
# eran = range(len(cont.ie))

# initialize rhoz for contour integration
if rank==root_node:

    rhoz_up = []
    rhoz_do = []

# sampling the integrand on the contour
for e_i in eran:
    ze = cont.ze[e_i]
    # reset G-s
    Gu = np.zeros((dh.no,dh.no),dtype='complex128');
    Gd = np.zeros((dh.no,dh.no),dtype='complex128');
    Gu_tmp = np.zeros((dh.no,dh.no),dtype='complex128');
    Gd_tmp = np.zeros((dh.no,dh.no),dtype='complex128');


    # doing parallel BZ integral
    for k in kpcs[rank]:

        HKU,HKD,SK = hsk(dh,k)
        Gku = nl.inv((ze*SK-HKU))
        Gkd = nl.inv((ze*SK-HKD))

        Gu_tmp += np.dot(Gku,SK)*wk
        Gd_tmp += np.dot(Gkd,SK)*wk        

    # summ reduce partial results of mpi nodes
    comm.Reduce(Gu_tmp,Gu,root=root_node)
    comm.Reduce(Gd_tmp,Gd,root=root_node)

    if rank==root_node:

        rhoz_up.append(np.trace(Gu))
        rhoz_do.append(np.trace(Gd))

# evaluation of the contour integral on the root node
if rank==root_node:

    rhoz_up = np.array(rhoz_up)    
    rhoz_do = np.array(rhoz_do)    
    # doing the contour integral
    rho_up = -np.trapz(np.imag((rhoz_up)*(cont.we))/(np.pi),cont.ie);   
    rho_do = -np.trapz(np.imag((rhoz_do)*(cont.we))/(np.pi),cont.ie);   
    #print('')
    #print(rho_up,rho_do)



