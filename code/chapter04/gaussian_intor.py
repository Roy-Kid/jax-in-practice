

'''
For a Gaussian Basis, we have a series of parameters to describe it:
In the context of GTO basis, such parameters (to describe a very basis) turns out to be rather simple:

    g = N (x-rx)^i (y-ry)^j (z-rz)^k exp(-alpha * ((x-rx)^2+(y-ry)^2+(z-rz)^2))

i,j,k:
    -- int
    the power of x,y,z before exp(-alpha * (x^2+y^2+z^2)), should be integers

R = (rx, ry, rz):
    -- Union[List[float], float]
    describing the center of Gaussian function

alpha:
    -- Union[float, jnp.ndarray]
    describing the variance of the gaussian basis, if we simply let
    g ~ exp(-alpha * x^2), then alpha = 0.5 * variance^(-1)

In the following, the parameters of the Gaussian functions are arranged in the order:
    -- [i,j,k,Rx,Ry,Rz,alpha]
'''
import jax
import jax.numpy as jnp
from jax import jit, lax
import functools
from abc import abstractmethod
from typing import Tuple, Union, Callable

# Type specification
I = J = K = int
RType = Rx = Ry = Rz = Union[float, jnp.ndarray]
AlphaType = Union[float, jnp.ndarray]
GaussianParameters = Tuple[I, J, K, Rx, Ry, Rz, AlphaType]
GaussianParameters_require_grad = Tuple[Rx, Ry, Rz, AlphaType]
GaussianParameters2 = Tuple[I, J, K, Rx, Ry, Rz, AlphaType, I, J, K, Rx, Ry, Rz, AlphaType]
GaussianParameters2_require_grad = Tuple[Rx, Ry, Rz, AlphaType, Rx, Ry, Rz, AlphaType]
int1e_nuc_Parameters = Tuple[Rx, Ry, Rz, AlphaType, Rx, Ry, Rz, AlphaType, Rx, Ry, Rz]
int1e_nuc_1s_Parameters = Tuple[Rx, Ry, Rz, AlphaType, Rx, Ry, Rz, AlphaType]


# i:int,j:int,k:int,

class Gaussian_Intor_Base(object):
    # @abstractmethod
    # def gaussian_int(self, n:int, alpha: float)->float:
    #     ''' int_0^inf x^n exp(-alpha x^2) dx '''
    #     ...

    @abstractmethod
    def gaussian_int(self, n:float, alpha: float)->float:
        ''' int_0^inf x^n exp(-alpha x^2) dx '''
        ...
    @abstractmethod
    def gaussian_normal_factor(self, i: int, j: int, k: int, alpha: AlphaType)->jnp.ndarray:
        '''get the nornalization factor N'''
        ...
    @abstractmethod
    def int1e_ovlp(self,
                   i1: int, j1: int, k1: int, r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                   i2: int, j2: int, k2: int, r2x: RType, r2y: RType, r2z: RType, a2: AlphaType)->Union[float,jnp.asarray]:

        '''calculate the overlapping integral'''
        # ( \ | \ )
        ...

    @abstractmethod
    def int1e_kin(self,
                  i1: int, j1: int, k1: int, r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                  i2: int, j2: int, k2: int, r2x: RType, r2y: RType, r2z: RType, a2: AlphaType)->Union[float,jnp.asarray]:
        '''calculate the kinetic energy integrol'''
        # 0.5 ( \ | p dot p | \ )  or  -0.5 ( \ | nabla^2 | \ )
        ...

    #  -----------------------   function generators  ----------------------- #

    @abstractmethod
    def gen_int1e_ovlp(self, i1:int, j1:int, k1:int, i2:int, j2:int, k2:int, normalize = True):
        '''returns a Callable function to calculate the int1e overlapping integral, with predefined ijk parameters'''
        ...

    @abstractmethod
    def gen_int1e_kin(self, i1: int, j1: int, k1: int, i2: int, j2: int, k2: int, normalize:bool = True) ->Callable:
        '''returns a Callable function to calculate the int1e_kinetic
        energy integral, with predefined ijk parameters'''
        ...

    @abstractmethod
    def gen_int1e_nuc(self, i1:int, j1:int, k1:int, i2:int, j2:int, k2:int,
                      normalize:bool = True, Zc_Bool:bool = True) -> Callable:
        '''returns a Callable function to calculate the int1e_nuclear
        attraction integral, with predefined ijk parameters'''
        ...

    @abstractmethod
    def gen_int2e(self, i1:int, j1:int, k1:int, i2:int, j2:int, k2:int,
                  i3:int, j3:int, k3:int, i4:int, j4:int, k4:int,
                  normalize:bool = True) -> Callable:
        '''returns a Callable function to calculate the int2e repulsion
        integral, with predefined ijk parameters'''
        ...



def doublefactorial(n:int):
    # return n!! (Better not to use this, it's tooooo slow...)
    return lax.while_loop(cond_fun=lambda ls: ls[0] > 0,
                          body_fun=lambda ls: [ls[0]-2,ls[1]*ls[0]],
                          init_val=[n,1])[1]

def factorial(n:int):
    # return n!
    if n == 0 or n == 1: return 1.
    if n == 2: return 2.
    if n == 3: return 6.
    if n == 4: return 24.
    if n == 5: return 120.
    if n == 6: return 720.
    if n == 7: return 5040.
    body_fun = lambda i, x: x * i
    return lax.fori_loop(lower=1, upper = n+1, body_fun = body_fun, init_val = 1)

def _factorial(n:int):
    # return n!
    body_fun = lambda i, x: x * i
    return lax.fori_loop(lower=1, upper = n+1, body_fun = body_fun, init_val = 1)

class Gaussian_Intor(Gaussian_Intor_Base):
    int1e_ovlp_function_dictionary = {}
    int1e_kin_function_dictionary = {}

    ## ---------------------      Some basic integrals:          --------------------  ##
    @classmethod
    def gaussian_int(cls, n:float, alpha: float) ->float:
        n1 = (n + 1) * 0.5
        return jnp.exp(jax.scipy.special.gammaln(n1)) / (2 * alpha ** n1)


    @classmethod
    def gaussian_normal_factor(cls, i: int, j: int, k: int,
                               alpha: AlphaType) -> jnp.ndarray:
        i1,j1,k1 = (2*i+1)/2, (2*j+1)/2, (2*k+1)/2
        return jnp.sqrt((2*alpha)**(i1+j1+k1)/(jnp.exp(jax.scipy.special.gammaln(i1))*
                                               jnp.exp(jax.scipy.special.gammaln(j1))*
                                               jnp.exp(jax.scipy.special.gammaln(k1))))


    ## ------------------------   overlapping integral   ----------------------- ##
    # @classmethod
    # def gen_int1e_ovlp(cls, i1:int, j1:int, k1:int, i2:int, j2:int, k2:int, normalize = True):
    #     idx = cls.__encode_int1e_ovlp(i1, j1, k1, i2, j2, k2, normalize)  # encoding the input into an int
    #     if idx in cls.int1e_ovlp_function_dictionary:
    #         return cls.int1e_ovlp_function_dictionary[idx]
    #     else:
    #         function = cls._gen_int1e_ovlp(i1, j1, k1, i2, j2, k2, normalize)
    #         cls.int1e_ovlp_function_dictionary[idx] = function
    #         return function

    @classmethod
    def gen_int1e_ovlp(cls,
                      i1:int, j1:int, k1:int,
                      i2:int, j2:int, k2:int, normalize = True):
        '''return a valid overlapping integral's Callable function'''
        if i1<0 or i2<0 or j1<0 or j2 < 0 or k1 < 0 or k2 < 0:
            return cls.zero
        else:
            def int1e_ovlp(r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                           r2x: RType, r2y: RType, r2z: RType, a2: AlphaType) -> Union[float, jnp.asarray]:
                # Ix = cls.gaussian_int_ovlp(i1, i2, a2 * (r2x - r1x) / (a1 + a2), a1 * (r1x - r2x) / (a1 + a2), a1, a2)
                # Iy = cls.gaussian_int_ovlp(j1, j2, a2 * (r2y - r1y) / (a1 + a2), a1 * (r1y - r2y) / (a1 + a2), a1, a2)
                # Iz = cls.gaussian_int_ovlp(k1, k2, a2 * (r2z - r1z) / (a1 + a2), a1 * (r1z - r2z) / (a1 + a2), a1, a2)
                ''' int_{-inf}{inf} (x+r1x)^i1 * (x+r2x)^i2 e^(-(alpha1+alpha2)x^2) dx'''
                Ix = 0.
                for m in range(i1 + 1):
                    for n in range(i2 + 1):
                        if (m + n) % 2 == 0:
                            r1 = a2 * (r2x - r1x) / (a1 + a2)
                            r2 = a1 * (r1x - r2x) / (a1 + a2)
                            n1 = (n + m + 1) * 0.5
                            Ix = Ix + r1 ** (i1 - m) * r2 ** (i2 - n) * 2 * \
                                 jnp.exp(jax.scipy.special.gammaln(n1)) / (2 * (a1 + a2) ** n1) / \
                                 (factorial(m) * factorial(n) * factorial(i1 - m) * factorial(i2 - n))
                Ix = Ix * factorial(i1) * factorial(i2)

                ''' int_{-inf}{inf} (y+r1)^i1 * (y+r2)^i2 e^(-(alpha1+alpha2)y^2) dy'''
                Iy = 0.
                for m in range(j1 + 1):
                    for n in range(j2 + 1):
                        if (m + n) % 2 == 0:
                            r1 = a2 * (r2y - r1y) / (a1 + a2)
                            r2 = a1 * (r1y - r2y) / (a1 + a2)
                            n1 = (n + m + 1) * 0.5
                            Iy = Iy + r1 ** (j1 - m) * r2 ** (j2 - n) * 2 * \
                                 jnp.exp(jax.scipy.special.gammaln(n1)) / (2 * (a1 + a2) ** n1) / \
                                 (factorial(m) * factorial(n) * factorial(j1 - m) * factorial(j2 - n))
                Iy = Iy * factorial(j1) * factorial(j2)

                ''' int_{-inf}{inf} (y+r1)^i1 * (y+r2)^i2 e^(-(alpha1+alpha2)y^2) dy'''
                Iz = 0.
                for m in range(k1 + 1):
                    for n in range(k2 + 1):
                        if (m + n) % 2 == 0:
                            r1 = a2 * (r2z - r1z) / (a1 + a2)
                            r2 = a1 * (r1z - r2z) / (a1 + a2)
                            n1 = (n + m + 1) * 0.5
                            Iz = Iz + r1 ** (k1 - m) * r2 ** (k2 - n) * 2 * \
                                 jnp.exp(jax.scipy.special.gammaln(n1)) / (2 * (a1 + a2) ** n1) / \
                                 (factorial(m) * factorial(n) * factorial(k1 - m) * factorial(k2 - n))
                Iz = Iz * factorial(k1) * factorial(k2)

                K = jnp.exp(-a1 * a2 / (a1 + a2) * ((r1x - r2x) ** 2 + (r1y - r2y) ** 2 + (r1z - r2z) ** 2))
                if normalize:
                    N1 = cls.gaussian_normal_factor(i1, j1, k1, a1)
                    N2 = cls.gaussian_normal_factor(i2, j2, k2, a2)
                    return N1 * N2 * Ix * Iy * Iz * K
                else:
                    return Ix * Iy * Iz * K
            return int1e_ovlp


    @classmethod
    def int1e_ovlp(cls,
                   i1: int, j1: int, k1: int, r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                   i2: int, j2: int, k2: int, r2x: RType, r2y: RType, r2z: RType, a2: AlphaType,
                   normalize = True) ->Union[float,jnp.asarray]:
        ''' calculate the overlap integral, but will not check whether ijk is no less than 0'''

        Ix = cls.gaussian_int_ovlp(i1, i2, a2*(r2x-r1x)/(a1+a2), a1*(r1x-r2x)/(a1+a2), a1, a2)
        Iy = cls.gaussian_int_ovlp(j1, j2, a2*(r2y-r1y)/(a1+a2), a1*(r1y-r2y)/(a1+a2), a1, a2)
        Iz = cls.gaussian_int_ovlp(k1, k2, a2*(r2z-r1z)/(a1+a2), a1*(r1z-r2z)/(a1+a2), a1, a2)
        K = jnp.exp(-a1 * a2/(a1 + a2) * ((r1x-r2x)**2 + (r1y-r2y)**2 + (r1z-r2z)**2))
        if not normalize:
            return Ix * Iy * Iz * K
        else:
            N1 = cls.gaussian_normal_factor(i1, j1, k1, a1)
            N2 = cls.gaussian_normal_factor(i2, j2, k2, a2)
            return N1 * N2 * Ix * Iy * Iz * K

    @classmethod
    def gaussian_int_ovlp(cls,i1:int,i2:int,r1:RType,r2:RType,alpha1:AlphaType,alpha2:AlphaType):
        ''' int_{-inf}{inf} (x+r1)^i1 * (x+r2)^i2 e^(-(alpha1+alpha2)x^2) dx'''

        ##  faster with jit, but will be mush slower without:
        # def i1_loop(val1: float, m:int)->Tuple:
        #     def i2_loop(val2:float, n:int)->Tuple:
        #         return val2 + cls.__gaussian_int1e_ovlp_with_params(m,n,i1,i2,r1,r2,alpha1,alpha2), None
        #     return lax.scan(f=i2_loop, init=val1, xs=jnp.arange(i2 + 1,dtype=int),)[0], None
        # return lax.scan(f=i1_loop,init=0,xs=jnp.arange(i1+1, dtype = int))[0] * _factorial(i1) * _factorial(i2)

        ## 2 times slower with jit, but will be mush faster without:
        val = 0
        for m in range(i1+1):
            for n in range(i2+1):
                val = val + cls.__gaussian_int1e_ovlp_with_params(m,n,i1,i2,r1,r2,alpha1,alpha2)
        return val * _factorial(i1) * _factorial(i2)

        # # Cannot be differentiated backward, but can still be jitted anyway:
        # def i1_loop(m:int, val1: float)->float:
        #     def i2_loop(n:int, val2:float)->float:
        #         return val2 + cls.__gaussian_int1e_ovlp_with_params(m,n,i1,i2,r1,r2,alpha1,alpha2)
        #     return lax.fori_loop(lower=0, upper=i2+1,body_fun=i2_loop,init_val=val1)
        # return lax.fori_loop(lower=0,upper=i1+1,body_fun=i1_loop,init_val=0) * _factorial(i1) * _factorial(i2)

    @classmethod
    def __gaussian_int1e_ovlp_with_params(cls, m:int ,n:int,i1:int,i2:int,
                                          r1:RType, r2:RType, alpha1:AlphaType, alpha2:AlphaType):
        return lax.cond((m+n)%2==1, cls.zero, cls.__gaussian_int1e_ovlp_with_params_nonzeros, (m,n,i1,i2,r1,r2,alpha1,alpha2))

    @classmethod
    def __gaussian_int1e_ovlp_with_params_nonzeros(cls,args):
        m, n, i1, i2, r1, r2, alpha1, alpha2 = args
        return r1**(i1-m)*r2**(i2-n) *2 * cls.gaussian_int(m+n,alpha1+alpha2) / \
               (_factorial(m)*_factorial(n)*_factorial(i1-m)*_factorial(i2-n))

    ## ------------------------   Kinetic integral   ----------------------- ##
    # @classmethod
    # def gen_int1e_kin(cls, i1:int, j1:int, k1:int, i2:int, j2:int, k2:int, normalize = True):
    #     idx = cls.__encode_int1e_kin(i1, j1, k1, i2, j2, k2, normalize)  # encoding the input into a int
    #     if idx in cls.int1e_kin_function_dictionary:
    #         return cls.int1e_kin_function_dictionary[idx]
    #     else:
    #         function = cls._gen_int1e_kin(i1, j1, k1, i2, j2, k2, normalize)
    #         cls.int1e_kin_function_dictionary[idx] = function
    #         return function

    @classmethod
    def gen_int1e_kin(cls,
                      i1: int, j1: int, k1: int,
                      i2: int, j2: int, k2: int,
                      normalize:bool = True) ->Callable:
        ''' Calculate the kinetic energy integral '''
        # get the overlaping integral calculator
        I11 = cls.gen_int1e_ovlp(i1, j1, k1, i2-2, j2, k2  , normalize=False)
        I12 = cls.gen_int1e_ovlp(i1, j1, k1, i2,   j2, k2  , normalize=False)
        I13 = cls.gen_int1e_ovlp(i1, j1, k1, i2+2, j2, k2  , normalize=False)
        I21 = cls.gen_int1e_ovlp(i1, j1, k1, i2, j2-2, k2  , normalize=False)
        I22 = cls.gen_int1e_ovlp(i1, j1, k1, i2, j2  , k2  , normalize=False)
        I23 = cls.gen_int1e_ovlp(i1, j1, k1, i2, j2+2, k2  , normalize=False)
        I31 = cls.gen_int1e_ovlp(i1, j1, k1, i2, j2  , k2-2, normalize=False)
        I32 = cls.gen_int1e_ovlp(i1, j1, k1, i2, j2  , k2  , normalize=False)
        I33 = cls.gen_int1e_ovlp(i1, j1, k1, i2, j2  , k2+2, normalize=False)
        def int1e_kin(r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                      r2x: RType, r2y: RType, r2z: RType, a2: AlphaType) -> Union[float,jnp.ndarray]:
            Ix = -0.5 * i2 * (i2 - 1) * I11(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2) \
                 + a2 * (2 * i2 + 1)  * I12(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2) \
                 - 2 * a2 ** 2        * I13(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2)
            Iy = -0.5 * j2 * (j2 - 1) * I21(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2) \
                 + a2 * (2 * j2 + 1)  * I22(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2) \
                 - 2 * a2 ** 2        * I23(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2)
            Iz = -0.5 * k2 * (k2 - 1) * I31(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2) \
                 + a2 * (2 * k2 + 1)  * I32(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2) \
                 - 2 * a2 ** 2        * I33(r1x, r1y, r1z, a1, r2x, r2y, r2z, a2)
            if normalize:
                N1 = cls.gaussian_normal_factor(i1, j1, k1, a1)
                N2 = cls.gaussian_normal_factor(i2, j2, k2, a2)
                return (Ix + Iy + Iz) * N1 * N2
            else:
                return (Ix + Iy + Iz)

        return int1e_kin


    ## ------------------------  nuclues attraction integral   ----------------------- ##
    @classmethod
    def gen_int1e_nuc(cls,
                      i1:int, j1:int, k1:int,
                      i2:int, j2:int, k2:int,
                      normalize:bool = True, Zc_Bool:bool = True) -> Callable:
        # (i1,j1,k1 | i2,j2,k2)
        # This is the function that the generator is going to return.
        def int1e_nuc(r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                      r2x: RType, r2y: RType, r2z: RType, a2: AlphaType,
                      r3x: RType, r3y: RType, r3z: RType, Zc: Union[float,int]) -> Union[float, jnp.ndarray]:
            """
                r1 = (r1x, r1y, r1z): the center of the first gaussian basis
                r2 = (r2x, r2y, r2z): the center of the second gaussian basis
                r3 = (r3x, r3y, r3z): the center of the atom's core
                Zc: charge of the atom's core
                Zc_bool: to ensure that Z_c are only calculated once
            """

            Normal_factor = 1. if not normalize else cls.gaussian_normal_factor(i1,j1,k1,a1) * cls.gaussian_normal_factor(i2,j2,k2,a2)
            Zc_factor = 1. if not Zc_Bool else -Zc

            # argnums  =   0    1    2   3    4    5    6   7    8    9    10  11
            params_all = (r1x, r1y, r1z, a1, r2x, r2y, r2z, a2, r3x, r3y, r3z, Zc)
            I_res = None  # integral's result

            if i1 == 0 and j1 == 0 and k1 == 0 and i2 == 0 and j2 == 0 and k2 == 0:
                # params = r1x, r1y, r1z, a1, r2x, r2y, r2z, a2
                Rp_abs = cls.__distance((a1 * r1x + a2 * r2x) / (a1 + a2), (a1 * r1y + a2 * r2y) / (a1 + a2),
                                        (a1 * r1z + a2 * r2z) / (a1 + a2), r3x, r3y, r3z)

                # These are the int1e_nuc functions that are going to be returned when ijk=0
                def __int1e_nuc_1s_(void: int = 0) -> Union[float, jnp.ndarray]:
                    return (jnp.pi / (a1 + a2)) ** (3 / 2) * jax.scipy.special.erf(jnp.sqrt(a1 + a2) * Rp_abs) / Rp_abs * \
                           jnp.exp(-((r1x - r2x)**2 + (r1y - r2y)**2 +(r1z - r2z)**2) * a1 * a2 / (a1 + a2))

                def __int1e_nuc_1s_div0_(void: int = 0) -> Union[float, jnp.ndarray]:
                    return 2 * jnp.pi / (a1 + a2) * \
                           jnp.exp(-((r1x - r2x)**2 + (r1y - r2y)**2 +(r1z - r2z)**2) * a1 * a2 / (a1 + a2))
                I_res =  lax.cond(Rp_abs < 1E-6, __int1e_nuc_1s_div0_, __int1e_nuc_1s_, 0)

            elif i1 < 0 or j1 < 0 or k1 < 0 or i2 < 0 or j2 < 0 or k2 < 0:
                I_res = 0.
            elif i1 > 0:
                I_res = (jax.grad(cls.gen_int1e_nuc(i1-1, j1, k1, i2, j2, k2, normalize=False, Zc_Bool=False), argnums=0)(*params_all) +
                        (i1 - 1) * cls.gen_int1e_nuc(i1-2, j1, k1, i2, j2, k2, normalize=False, Zc_Bool=False)(*params_all))/ (2*a1)
            elif j1 > 0:
                I_res = (jax.grad(cls.gen_int1e_nuc(i1, j1-1, k1, i2, j2, k2, normalize=False, Zc_Bool=False), argnums=1)(*params_all) +
                        (j1 - 1) * cls.gen_int1e_nuc(i1, j1-2, k1, i2, j2, k2, normalize=False, Zc_Bool=False)(*params_all))/ (2*a1)
            elif k1 > 0:
                I_res = (jax.grad(cls.gen_int1e_nuc(i1, j1, k1-1, i2, j2, k2, normalize=False, Zc_Bool=False), argnums=2)(*params_all) +
                        (k1 - 1) * cls.gen_int1e_nuc(i1, j1, k1-2, i2, j2, k2, normalize=False, Zc_Bool=False)(*params_all))/ (2*a1)
            elif i2 > 0:
                I_res = (jax.grad(cls.gen_int1e_nuc(i1, j1, k1, i2-1, j2, k2, normalize=False, Zc_Bool=False), argnums=4)(*params_all) +
                        (i2 - 1) * cls.gen_int1e_nuc(i1, j1, k1, i2-2, j2, k2, normalize=False, Zc_Bool=False)(*params_all))/ (2*a2)
            elif j2 > 0:
                I_res = (jax.grad(cls.gen_int1e_nuc(i1, j1, k1, i2, j2-1, k2, normalize=False, Zc_Bool=False), argnums=5)(*params_all) +
                        (j2 - 1) * cls.gen_int1e_nuc(i1, j1, k1, i2, j2-2, k2, normalize=False, Zc_Bool=False)(*params_all))/ (2*a2)
            elif k2 > 0:
                I_res = (jax.grad(cls.gen_int1e_nuc(i1, j1, k1, i2, j2, k2-1, normalize=False, Zc_Bool=False), argnums=6)(*params_all) +
                        (k2 - 1) * cls.gen_int1e_nuc(i1, j1, k1, i2, j2, k2-2, normalize=False, Zc_Bool=False)(*params_all))/ (2*a2)
            else:
                import logging
                logging.fatal("Falal error occurs while doing the iteration in function gen_int1e_nuc")
                raise ValueError("Fatal!")
            return I_res * Normal_factor * Zc_factor
        return int1e_nuc


    ## ------------------------  Two electrons' repulsion integral   ----------------------- ##

    @classmethod
    def gen_int2e(cls,
                  i1:int, j1:int, k1:int,
                  i2:int, j2:int, k2:int,
                  i3:int, j3:int, k3:int,
                  i4:int, j4:int, k4:int,
                  normalize: bool = True) -> Callable:
        '''   (i1,j1,k1, i2,j2,k2 | rinv | i3,j3,k3, i4,j4,k4) '''
        """
        normalize: determining whether to normalize the gaussian basis or not.
        Zc_bool: to ensure that Z_c are only calculated once, highly recommended to set as its default value
        """
        # This is the function that the generator is going to return.
        def int2e(r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                  r2x: RType, r2y: RType, r2z: RType, a2: AlphaType,
                  r3x: RType, r3y: RType, r3z: RType, a3: AlphaType,
                  r4x: RType, r4y: RType, r4z: RType, a4: AlphaType,) -> Union[float, jnp.ndarray]:
            """
                r1 = (r1x, r1y, r1z): the center of the first gaussian basis
                r2 = (r2x, r2y, r2z): the center of the second gaussian basis
                r3 = (r3x, r3y, r3z): the center of the third gaussian basis
                r4 = (r4x, r4y, r4z): the center of the fourth gaussian basis
            """

            Normal_factor = 1. if not normalize else cls.gaussian_normal_factor(i1,j1,k1,a1) * cls.gaussian_normal_factor(i2,j2,k2,a2) * \
                                                     cls.gaussian_normal_factor(i3,j3,k3,a3) * cls.gaussian_normal_factor(i4,j4,k4,a4)

            # argnums  =   0    1    2    3   4    5    6   7    8    9   10   11   12   13  14   15
            params_all = (r1x, r1y, r1z, a1, r2x, r2y, r2z, a2, r3x, r3y, r3z, a3, r4x, r4y, r4z, a4)
            I_res = None  # integral's result

            if i1 == 0 and j1 == 0 and k1 == 0 and i2 == 0 and j2 == 0 and k2 == 0 and \
               i3 == 0 and j3 == 0 and k3 == 0 and i4 == 0 and j4 == 0 and k4 == 0:

                Px, Py, Pz = (a1 * r1x + a2 * r2x) / (a1 + a2), (a1 * r1y + a2 * r2y) / (a1 + a2), (a1 * r1z + a2 * r2z) / (a1 + a2)
                Qx, Qy, Qz = (a3 * r3x + a4 * r4x) / (a3 + a4), (a3 * r3y + a4 * r4y) / (a3 + a4), (a3 * r3z + a4 * r4z) / (a3 + a4)
                PQ = jnp.sqrt((Px - Qx) ** 2 + (Py - Qy) ** 2 + (Pz - Qz) ** 2)
                # These are the int2e functions that are going to be returned when ijk=0
                def __int2e_1s(void: int = 0) -> Union[float, jnp.ndarray]:
                    return jnp.pi ** 3 / ((a1+a2) * (a3+a4)*jnp.sqrt(a1+a2+a3+a4)) * \
                           jax.scipy.special.erf(PQ * jnp.sqrt((a1+a2)*(a3+a4)/(a1+a2+a3+a4))) / \
                                                (PQ * jnp.sqrt((a1+a2)*(a3+a4)/(a1+a2+a3+a4))) * \
                           jnp.exp(-((r1x - r2x)**2 + (r1y - r2y)**2 +(r1z - r2z)**2) * a1 * a2 / (a1 + a2)) * \
                           jnp.exp(-((r3x - r4x)**2 + (r3y - r4y)**2 +(r3z - r4z)**2) * a3 * a4 / (a3 + a4))

                def __int2e_1s_div0(void: int = 0) -> Union[float, jnp.ndarray]:
                    return 2 * jnp.pi ** (2.5) / ((a1+a2) * (a3+a4)*jnp.sqrt(a1+a2+a3+a4)) * \
                           jnp.exp(-((r1x - r2x)**2 + (r1y - r2y)**2 +(r1z - r2z)**2) * a1 * a2 / (a1 + a2)) * \
                           jnp.exp(-((r3x - r4x)**2 + (r3y - r4y)**2 +(r3z - r4z)**2) * a3 * a4 / (a3 + a4))

                I_res =  lax.cond(PQ < 1E-6, __int2e_1s_div0, __int2e_1s, 0)

            elif i1 < 0 or j1 < 0 or k1 < 0 or i2 < 0 or j2 < 0 or k2 < 0 or \
                 i3 < 0 or j3 < 0 or k3 < 0 or i4 < 0 or j4 < 0 or k4 < 0:
                I_res = 0.
            elif i1 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1-1, j1, k1, i2, j2, k2, i3, j3, k3, i4, j4, k4, normalize=False), argnums=0)(*params_all) +
                        (i1 - 1) * cls.gen_int2e(i1-2, j1, k1, i2, j2, k2, i3, j3, k3, i4, j4, k4,  normalize=False)(*params_all))/ (2*a1)
            elif j1 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1-1, k1, i2, j2, k2, i3, j3, k3, i4, j4, k4, normalize=False), argnums=1)(*params_all) +
                        (j1 - 1) * cls.gen_int2e(i1, j1-2, k1, i2, j2, k2, i3, j3, k3, i4, j4, k4,  normalize=False)(*params_all))/ (2*a1)
            elif k1 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1-1, i2, j2, k2, i3, j3, k3, i4, j4, k4, normalize=False), argnums=2)(*params_all) +
                        (k1 - 1) * cls.gen_int2e(i1, j1, k1-2, i2, j2, k2, i3, j3, k3, i4, j4, k4,  normalize=False)(*params_all))/ (2*a1)
            elif i2 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2-1, j2, k2, i3, j3, k3, i4, j4, k4, normalize=False), argnums=4)(*params_all) +
                        (i2 - 1) * cls.gen_int2e(i1, j1, k1, i2-2, j2, k2, i3, j3, k3, i4, j4, k4,  normalize=False)(*params_all))/ (2*a2)
            elif j2 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2, j2-1, k2, i3, j3, k3, i4, j4, k4, normalize=False), argnums=5)(*params_all) +
                        (j2 - 1) * cls.gen_int2e(i1, j1, k1, i2, j2-2, k2, i3, j3, k3, i4, j4, k4,  normalize=False)(*params_all))/ (2*a2)
            elif k2 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2, j2, k2-1, i3, j3, k3, i4, j4, k4, normalize=False), argnums=6)(*params_all) +
                        (k2 - 1) * cls.gen_int2e(i1, j1, k1, i2, j2, k2-2, i3, j3, k3, i4, j4, k4,  normalize=False)(*params_all))/ (2*a2)
            elif i3 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3-1, j3, k3, i4, j4, k4, normalize=False), argnums=8)(*params_all) +
                        (i1 - 1) * cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3-2, j3, k3, i4, j4, k4,  normalize=False)(*params_all))/ (2*a3)
            elif j3 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3-1, k3, i4, j4, k4, normalize=False), argnums=9)(*params_all) +
                        (j1 - 1) * cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3-2, k3, i4, j4, k4,  normalize=False)(*params_all))/ (2*a3)
            elif k3 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3, k3-1, i4, j4, k4, normalize=False), argnums=10)(*params_all) +
                        (k1 - 1) * cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3, k3-2, i4, j4, k4,  normalize=False)(*params_all))/ (2*a3)
            elif i4 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3, k3, i4-1, j4, k4, normalize=False), argnums=12)(*params_all) +
                        (i2 - 1) * cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3, k3, i4-2, j4, k4,  normalize=False)(*params_all))/ (2*a4)
            elif j4 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3, k3, i4, j4-1, k4, normalize=False), argnums=13)(*params_all) +
                        (j2 - 1) * cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3, k3, i4, j4-2, k4,  normalize=False)(*params_all))/ (2*a4)
            elif k4 > 0:
                I_res = (jax.grad(cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3, k3, i4, j4, k4-1, normalize=False), argnums=14)(*params_all) +
                        (k2 - 1) * cls.gen_int2e(i1, j1, k1, i2, j2, k2, i3, j3, k3, i4, j4, k4-2,  normalize=False)(*params_all))/ (2*a4)
            else:
                import logging
                logging.fatal("Falal error occurs while doing the iteration in function gen_int2e")
                raise ValueError("Fatal!")
            return I_res * Normal_factor
        return int2e

    ## ------------------------     other functions    ----------------------- ##

    @classmethod
    def __distance(cls, r1x: RType, r1y: RType, r1z: RType,
                        r2x: RType, r2y: RType, r2z: RType):
        ''' return the distance between two 3D vectors'''
        return jnp.sqrt((r1x - r2x)**2 + (r1y -r2y)**2 + (r1z - r2z)**2)

    @classmethod
    def zero(*args): return 0.

    @classmethod
    def __encode_int1e_ovlp(cls,
                            i1: int, j1: int, k1: int,
                            i2: int, j2: int, k2: int, normalize=True) -> int:
        assert i1 < 10 and i2 < 10 and j1 < 10 and j2 < 10 and k1 < 10 and k2 < 10
        flag = 1 if normalize else 0
        return int(flag * 1E6 + i1 * 1E5 + j1 * 1E4 + k1 * 1E3 + i2 * 1E2 + j2 * 1E1 + k2)

    @classmethod
    def __encode_int1e_kin(cls,
                           i1: int, j1: int, k1: int,
                           i2: int, j2: int, k2: int, normalize=True) -> int:
        return cls.__encode_int1e_ovlp(i1, j1, k1, i2, j2, k2, normalize)


# when ijk = 000
def S_GF(r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
         r2x: RType, r2y: RType, r2z: RType, a2: AlphaType): # S_GF(para_1,para_2) is the overlap integral of GFs
    alpha_1,alpha_2 = a1,a2
    x_1 = 0
    x_2 = jnp.sqrt((r1x-r2x)**2+(r1y-r2y)**2+(r1z-r2z)**2)
    return (4 * alpha_1 * alpha_2 / (alpha_1 + alpha_2) ** 2) ** (3 / 4) * jnp.exp(-1 * alpha_1 * alpha_2 * (x_1 - x_2) ** 2 / (alpha_1 + alpha_2)) # checked


# when ijk = 000
def T_GF(r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
         r2x: RType, r2y: RType, r2z: RType, a2: AlphaType): # T(para_1,para_2) is the kinetic energy integral of GFS, para_1, para_2 is the parameters of the 2 input GFs
    alpha_1,alpha_2 = a1,a2
    x_1 = 0
    x_2 = jnp.sqrt((r1x-r2x)**2+(r1y-r2y)**2+(r1z-r2z)**2)
    K = jnp.exp(-1 * alpha_1 * alpha_2 * (x_1 - x_2) ** 2 / (alpha_1 + alpha_2))
    p = alpha_1 + alpha_2
    return 2 ** (3 / 2) * (alpha_1 * alpha_2) ** (7 / 4) * K * p ** (-5 / 2) * (3 - 2 * alpha_1 * alpha_2 * (x_1 - x_2) ** 2 / p) # checked


# when ijk = 000
def V_one_nucl_GF_000(r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                      r2x: RType, r2y: RType, r2z: RType, a2: AlphaType,
                      r3x: RType, r3y: RType, r3z: RType, Zc): # Z_C is the charge of the nucleus
    alpha_1,alpha_2 = a1, a2
    K = jnp.exp(-1 * alpha_1 * alpha_2 * ((r1x - r2x) ** 2 + (r1y - r2y)**2 + (r1z - r2z)**2) / (alpha_1 + alpha_2))
    p = alpha_1 + alpha_2
    Rpx = (alpha_1 * r1x + alpha_2 * r2x) / (alpha_1 + alpha_2)
    Rpy = (alpha_1 * r1y + alpha_2 * r2y) / (alpha_1 + alpha_2)
    Rpz = (alpha_1 * r1z + alpha_2 * r2z) / (alpha_1 + alpha_2)
    R_Q = jnp.sqrt((Rpx - r3x) ** 2 + (Rpy - r3y)**2 + (Rpz - r3z)**2) # R_Q is always positive
    if R_Q > 10 ** (-6):
        return -1 * (4 * alpha_1 * alpha_2 / p ** 2) ** (3 / 4) * Zc * K * jax.scipy.special.erf(R_Q * p ** 0.5) / R_Q
    else:
        return -1 * (4 * alpha_1 * alpha_2) ** (3 / 4) * 2 * Zc * K / (p * jnp.pi ** 0.5)


# when ijk = 100
def V_one_nucl_GF_100(r1x: RType, r1y: RType, r1z: RType, a1: AlphaType,
                     r2x: RType, r2y: RType, r2z: RType, a2: AlphaType,
                     r3x: RType, r3y: RType, r3z: RType, Zc): # Z_C is the charge of the nucleus, while R_C is the coordinate (here only the x coordinate) of the nucleus
    alpha_1,alpha_2 = a1, a2
    K = jnp.exp(-1 * alpha_1 * alpha_2 * ((r1x - r2x) ** 2 + (r1y - r2y)**2 + (r1z - r2z)**2) / (alpha_1 + alpha_2))
    p = alpha_1 + alpha_2
    Rpx = (alpha_1 * r1x + alpha_2 * r2x) / (alpha_1 + alpha_2)
    Rpy = (alpha_1 * r1y + alpha_2 * r2y) / (alpha_1 + alpha_2)
    Rpz = (alpha_1 * r1z + alpha_2 * r2z) / (alpha_1 + alpha_2)
    PC = jnp.sqrt((Rpx - r3x) ** 2 + (Rpy - r3y)**2 + (Rpz - r3z)**2) # R_Q is always positive
    I1 = 2 * jnp.pi / (a1 + a2) * K * Fm(1., (a1+a2) * PC**2) * (r3x - Rpx)
    I2 = - 2 * jnp.pi / ((a1 + a2) * a1) * Fm(0.,(a1+a2) * PC**2) * K * a1*a2/(a1+a2)*(r1x-r2x)
    return (I1+I2)*(-Zc)

def Fm(m,omega):
    return 1/(2 * omega**(m+0.5)) * jax.scipy.special.gammainc(m+0.5,omega) * jnp.exp(jax.scipy.special.gammaln(m+0.5))



if __name__ == "__main__":
    pass
    '''
    import time
    params_ijk = (0, 0, 0,
                  0, 0, 0)
    params_Ra = (1., 2., 3., 4.,
                 2., 2., 3., 4.,)
    ts = time.time()
    int1e_ovlp = Gaussian_Intor.gen_int1e_ovlp(*params_ijk)
    t0 = time.time() - ts
    print("overlap integral = ", int1e_ovlp(*params_Ra))
    print("check value:", S_GF(*params_Ra))
    print("overlap integral calculation time = ", t0, "\n\n")
    '''
    # int1e_kin = Gaussian_Intor.gen_int1e_kin(*params_ijk)
    # ts = time.time()
    # for i in range(1):
    #     int1e_kin(*params_Ra)
    # t0 = time.time() - ts
    # print(int1e_kin(*params_Ra))


    '''
    int1e_kin = jit(Gaussian_Intor.gen_int1e_kin(*params_ijk))
    ts = time.time()
    for i in range(1000):
        a = int1e_kin(*params_Ra)
    t0, ts = time.time() - ts, time.time()
    print("kinetic energy integral = ", int1e_kin(*params_Ra))
    print("check value:", T_GF(*params_Ra))
    print("kinetic energy integral time (with jit)", t0 / 1000, "\n\n")
    '''

    # int1e_grad = jit(jax.grad(Gaussian_Intor.gen_int1e_kin(*params_ijk)))
    # ts = time.time()
    # for i in range(1000):
    #     a = int1e_grad(*params_Ra)
    # t0, ts = time.time() - ts, time.time()
    # print("grad of kinetic energy's integral time (with jit)", t0 / 1000)
    # print(int1e_grad(*params_Ra))

    '''
    ijk = [0.] * 12
    int2e = Gaussian_Intor.gen_int2e(*ijk)

    coefficient = [0.444635, 0.535328, 0.154329]
    zeta = 1.24
    R = 1.4  # we place atom 1 at x = 0 and atom 2 at x = R (this is a 1D system)

    GF = [[None for i in range(3)] for j in range(2)]
    GF[0][0] = [zeta ** 2 * 0.109818, 0]  # format: [alpha, x_coordinate_of_the_nucleus]
    GF[0][1] = [zeta ** 2 * 0.405771, 0]  # format: [alpha, x_coordinate_of_the_nucleus]
    GF[0][2] = [zeta ** 2 * 2.22766, 0]  # format: [alpha, x_coordinate_of_the_nucleus]
    GF[1][0] = [zeta ** 2 * 0.109818, R]  # format: [alpha, x_coordinate_of_the_nucleus]
    GF[1][1] = [zeta ** 2 * 0.405771, R]  # format: [alpha, x_coordinate_of_the_nucleus]
    GF[1][2] = [zeta ** 2 * 2.22766, R]  # format: [alpha, x_coordinate_of_the_nucleus]
    output = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    output += coefficient[i] * coefficient[j] * coefficient[k] * coefficient[l] * int2e(GF[0][i][1],0.,0.,GF[0][i][0],
                                                                                                        GF[0][j][1],0.,0.,GF[0][j][0],
                                                                                                        GF[0][k][1],0.,0.,GF[0][k][0],
                                                                                                        GF[0][l][1],0.,0.,GF[0][l][0],Zc = 1)
    ee_integral_tensor = [[[[None for i in range(2)] for j in range(2)] for k in range(2)] for l in range(2)]
    ee_integral_tensor[0][0][0][0] = ee_integral_tensor[1][1][1][1] = output
    print('(CGF_0 CGF_0 | CGF_0 CGF_0) = ', ee_integral_tensor[0][0][0][0])  # checked
    '''

