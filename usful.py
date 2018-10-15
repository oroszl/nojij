import numpy as np
from scipy.special import roots_legendre
# define some useful functions
def hsk(dh,k=(0,0,0)):
    '''
    One way to speed up Hk and Sk generation
    '''
    k = np.asarray(k, np.float64) # this two conversion lines
    k.shape = (-1,)               # are from the sisl source

    # this generates the list of phases
    phases = np.exp(-1j * np.dot(np.dot(np.dot(dh.rcell, k), dh.cell),
                                 dh.sc.sc_off.T))

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
        return np.array([[0,0,0]])

    kran=len(dirs)*[np.linspace(0,1,NUMK,endpoint=False)]
    mg=np.meshgrid(*kran)
    dirsdict=dict()

    for d in enumerate(dirs):
        dirsdict[d[1]]=mg[d[0]].flatten()
    for d in 'xyz':
        if not(d in dirs):
            dirsdict[d]=0*dirsdict[dirs[0]]
    kset = np.array([dirsdict[d] for d in 'xyz']).T

    return kset

def make_atran(nauc,dirs='xyz',dist=1):
    '''
    Simple pair generator. Depending on the value of the dirs
    argument sampling in 1,2 or 3 dimensions is generated.
    If dirs argument does not contain either of x,y or z
    a single pair is returend.
    '''
    if not(sum([d in dirs for d in 'xyz'])):
        return (0,0,[1,0,0])

    dran=len(dirs)*[np.arange(-dist,dist+1)]
    mg=np.meshgrid(*dran)
    dirsdict=dict()

    for d in enumerate(dirs):
        dirsdict[d[1]]=mg[d[0]].flatten()
    for d in 'xyz':
        if not(d in dirs):
            dirsdict[d]=0*dirsdict[dirs[0]]

    ucran = np.array([dirsdict[d] for d in 'xyz']).T
    atran=[]
    for i,j in list(product(range(nauc),repeat=2)):
        for u in ucran:
            if (abs(i-j)+sum(abs(u)))>0:
                atran.append((i,j,list(u)))

    return atran
