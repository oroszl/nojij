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
from usful import hsk,make_kset,make_contour

#----------------------------------------------------------------------

start = timer()

# Some input parsing
parser = argparse.ArgumentParser()

parser.add_argument('--kset'    , dest = 'kset'   , default  = 2         ,
 type=int  , help = 'k-space resolution of Jq calculation')

parser.add_argument('--kdirs'   , dest = 'kdirs'  , default  = 'xyz'
            , help = 'Definition of k-space dimensionality')

parser.add_argument('--eset'    , dest = 'eset'   , default  = 42        ,
 type=int  , help = 'Number of energy points on the contour')

parser.add_argument('--eset-p'  , dest = 'esetp'  , default  = 10000     ,
 type=int  , help = 'Parameter tuning the distribution on the contour')

parser.add_argument('--input'   , dest = 'infile' , required = True
            , help = 'Input file name')

parser.add_argument('--output'  , dest = 'outfile', required = True
           , help = 'Output file name')

parser.add_argument('--Ebot'    , dest = 'Ebot'   , default  = -20.0     ,
 type=float, help = 'Bottom energy of the contour')

parser.add_argument('--qnum'    , dest = 'qnum'   , default  = 10        ,
 type=int  , help = 'Number of q points')

parser.add_argument('--qdir'    , dest = 'qdir'   , default  = 0         ,
 type=int  , help = 'Direction of q vectors')

parser.add_argument('--qmax'    , dest = 'qmax'   , default  = 0.1       ,
 type=float, help = 'Maximum of q vector')

parser.add_argument('--use-tqdm', dest = 'usetqdm', default  = 'not'
           , help = 'Use tqdm for progressbars or not')

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
dh.hup = dh.tocsr(0).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1)
dh.hdo = dh.tocsr(1).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1)
dh.sov = dh.tocsr(2).toarray().reshape(dh.no,dh.n_s,dh.no).transpose(0,2,1)
dh.hup = dh.hup.astype('complex128')
dh.hdo = dh.hdo.astype('complex128')
dh.sov = dh.sov.astype('complex128')
#----------------------------------------------------------------------

# generate k space sampling
kset=make_kset(dirs=args.kdirs,NUMK=args.kset)
wk = 1/len(kset) # weight of a kpoint in BZ integral
kpcs = np.array_split(kset,size)
if 'k' in args.usetqdm:
    kpcs[root_node] = tqdm.tqdm(kpcs[root_node],desc='k loop')
#----------------------------------------------------------------------
# generate q-vectors
# TODO: sofar ony single atom in the unitcell is implemented!!
# TODO: sofar direction of the qpath is limited!!
# generation of q-vectors strictly from points of the k-sampling !

qdir = np.array([[0],[0],[0]])
qdir[args.qdir,0] = 1
kran = np.linspace(0,1,args.kset,endpoint=False)
qran = np.sort(np.hstack((kran[ :min(int(args.qnum/2),args.kset)],
                         -kran[1:min(int(args.qnum/2),args.kset)])))
qvecs = (qran*qdir).T
qs = []
for qi in range(len(qran)):
    i,j = 0,0
    qs.append(dict(
        aiij   = [i,j], # indecies of the atoms in the unitcell
        noij   = [dh.atoms.orbitals[i],dh.atoms.orbitals[j]], # number of orbitals on the appropriate atoms
        slij   = [slice( *(lambda x:[min(x),max(x)+1])(dh.a2o(i,all=True)) ),  # slices for
                  slice( *(lambda x:[min(x),max(x)+1])(dh.a2o(j,all=True)) )], # appropriate orbitals
        qvec  = qvecs[qi], # qvector in the BZ
        Jqz   = [], # in this empty list are we going to gather the integrad of the energy integral
        Jq    = 0   # the final results of the calculation are going to be here on the root node
            ))


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

    for q in qs:
        q['Jqz_tmp_kloop']=np.zeros((1,1),dtype='complex128')
        q['Jqz_tmp_qloop']=np.zeros((1,1),dtype='complex128')
    # doing parallel BZ integral
    for k in kpcs[rank]:
        k=np.array(k)
        k.shape=(-1,)
        HKU,HKD,SK = hsk(dh,k)
#        Gku = nl.inv((ze*SK-HKU))
        Gkd = nl.inv((ze*SK-HKD))

        for q in qs:
            si,sj=q['slij']
            HKpQU,HKpQD,SKpQ = hsk(dh,k+q['qvec'])
            Gkpqu = nl.inv((ze*SKpQ-HKpQU))

            i,j = q['aiij']
            q['Jqz_tmp_kloop'][0,0] += (
                         np.trace(np.dot(
                         np.dot(Hs[i],Gkpqu[si,sj]),
                         np.dot(Hs[j],Gkd[sj,si])
                              )))*wk

    # summ reduce partial results of mpi nodes
    for q in qs:

        comm.Reduce(q['Jqz_tmp_kloop'],q['Jqz_tmp_qloop'],root=root_node)

        # append contributions to contour dependent list
        if rank==root_node:
            q['Jqz'].append(q['Jqz_tmp_qloop'][0,0])


#----------------------------------------------------------------------

# evaluation of the contour integral on the root node
# and saveing output of the calculation
if rank==root_node:
    for q in qs:
        q['Jqz'] = np.array(q['Jqz'])
        q['Jq']  = np.trapz(np.imag(q['Jqz']*cont.we)/(2*np.pi))
    end = timer()

    np.savetxt(args.outfile,
               np.array([ [nl.norm(q['qvec']),
                           q['Jq']*sisl.unit_convert('eV','Ry')*1000]+
                          list(q['qvec'])
                        for q in qs],
                        dtype=object),
               header=str(args)+
                      '\nnumber of cores = '+str(size)+
                      '\ntime of calculation = '+str(end-start)+
                      '\nnorm(q),Jq[mRy],qvec',
               fmt="%s")
