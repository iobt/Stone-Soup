import copy
from collections.abc import Sequence

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from .base import TransitionModel
from ..base import GaussianModel, TimeVariantModel, STVGaussianModel
from ...base import Property
from ...types.array import CovarianceMatrix, StateVector, StateVectors


class GaussianTransitionModel(TransitionModel, GaussianModel):
    pass

class STVGaussianTransitionModel(TransitionModel, STVGaussianModel):
    pass

class CTRV_STV(STVGaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a discrete, time-variant 2D Constant
    Turn Rate and Velocity Model (CTRV).

    The target is assumed to move with (nearly) constant velocity and also
    an unknown (nearly) constant turn rate. This implementation uses a
    state-dependent noise covariance matrix.
    """
    linear_noise_coeff: float = Property(
        doc=r"The linear acceleration noise in the heading direction :math:`q_l`")
    turn_noise_coeff: float = Property(
        doc=r"The turn rate noise coefficient :math:`q_\omega`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = None

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 5


    def function(self, state, noise=False, **kwargs) -> StateVector:
        dt       = kwargs['time_interval'].total_seconds()

        Px       = state.state_vector[0,:]
        Py       = state.state_vector[1,:]
        psi      = state.state_vector[2,:]
        v        = state.state_vector[3,:]
        omega    = state.state_vector[4,:]
    
        w            = dt * omega / 2       # half-angle term
        psi_mid      = psi + w              # mid-point angle
        sinc_val     = np.sinc(w / np.pi)   # numpy's sinc(x) = sin(π*x)/(π*x)
        cos_psi_mid  = np.cos(psi_mid)
        sin_psi_mid  = np.sin(psi_mid)
        Pxnew        = Px + dt*v*cos_psi_mid*sinc_val
        Pynew        = Py + dt*v*sin_psi_mid*sinc_val
        psinew       = psi + dt*omega

        svnew        = StateVectors([Pxnew,Pynew,psinew,v,omega])

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(svnew, num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        
        return svnew + noise
    
    def jacobian(self, state, **kwargs):

        dt       = kwargs['time_interval'].total_seconds()

        Px       = state.state_vector[0,:]
        Py       = state.state_vector[1,:]
        psi      = state.state_vector[2,:]
        v        = state.state_vector[3,:]
        omega    = state.state_vector[4,:]

        w            = dt * omega / 2       # half-angle term
        psi_mid      = psi + w              # mid-point angle
        sinc_val     = np.sinc(w / np.pi)   # numpy's sinc(x) = sin(π*x)/(π*x)
        cos_psi_mid  = np.cos(psi_mid)
        sin_psi_mid  = np.sin(psi_mid)

        j = np.zeros((5,5))
        j[0,0] = 1 #dfx/dx
        j[1,1] = 1 #df[y]/dy

        j[0,2] = -dt*v*sin_psi_mid*sinc_val #df[x]/dpsi
        j[1,2] = dt*v*cos_psi_mid*sinc_val #df[y]/dpsi
        j[2,2] = 1 #df[psi]/dpsi
        j[3,2] = 0 #df[v]/dpsi
        j[4,2] = 0 #df[omega]/dpsi

        j[0,3] = dt*cos_psi_mid*sinc_val #df[x]/dv
        j[1,3] = dt*sin_psi_mid*sinc_val #df[y]/dv
        j[2,3] = 0 #df[psi]/dv
        j[3,3] = 1 #df[v]/dv
        j[4,3] = 0 #df[omega]/dv

        if np.abs(w) < 1e-10:
            dsinc_dw = 0.0
        else:
            dsinc_dw = (w * np.cos(w) - np.sin(w)) / (w**2)

        j[0,4] = 0.5*(dt**2)*v*(-sin_psi_mid*sinc_val + cos_psi_mid*dsinc_dw) #df[x]/domega
        j[1,4] = 0.5*(dt**2)*v*(cos_psi_mid*sinc_val + sin_psi_mid*dsinc_dw) #df[y]/domega
        j[2,4] = dt #df[psi]/domega
        j[3,4] = 0 #df[v]/domega
        j[4,4] = 1 #df[omega]/domega

        #Store Q for this state
        #kwargs["state_vectors"] = StateVectors([Px,Py,psi,v,omega])
        #self.Q = self.covar(**kwargs)

        return j
    
    def rvs(self, state_vectors, num_samples=1, **kwargs):

        time_interval = kwargs["time_interval"]

        Qs = self.covar(time_interval,state_vectors)
        Qs = Qs.reshape(-1,5,5)

        if(Qs.shape[0]>1):
            num_samples=1
            
        noise = [multivariate_normal(np.zeros(self.ndim), Q).rvs(num_samples).reshape(-1,num_samples) for Q in Qs]

        return np.hstack(noise)

    def covar(self, time_interval, state_vectors, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        dt       = time_interval.total_seconds()
        var_accl = self.linear_noise_coeff
        var_accy = self.turn_noise_coeff

        if(len(state_vectors.shape))==1:
            state_vectors = state_vectors.reshape(-1,1)

        state_dim, batch_size = state_vectors.shape

        if(state_dim != self.ndim_state):
            raise ValueError("state dimension mismatch")

        psik = state_vectors[2,:]
        vk   = state_vectors[3,:]

        si = np.sin(psik)
        co = np.cos(psik)
        cs = np.cos(psik)*np.sin(psik)

        Q = np.zeros((batch_size,state_dim,state_dim))

        Q[:,0,0] = (var_accl * ((1/3) * dt**3 * co**2) + var_accy * ((1/20) * dt**5 * vk**2 * si**2)).squeeze()
        Q[:,1,0] = (var_accl*((1/3) * dt**3 * cs) - var_accy*( (1/20) * dt**5 * vk**2 * cs)).squeeze()
        Q[:,1,1] = (var_accl*((1/3) * dt**3 * si**2) + var_accy*( (1/20) * dt**5 * vk**2 * co**2)).squeeze()
        Q[:,2,0] = (-var_accy*((1/8) * vk * dt**4 * si)).squeeze()
        Q[:,2,1] = (var_accy*((1/8) * vk * dt**4 * co)).squeeze()
        Q[:,2,2] = (var_accy*((1/3) * dt**3))
        Q[:,3,0] = (var_accl*((1/2) * dt**2 * co)).squeeze()
        Q[:,3,1] = (var_accl*((1/2) * dt**2 * si)).squeeze()
        Q[:,3,2] = 0
        Q[:,3,3] = (var_accl * dt)
        Q[:,4,0] = (-var_accy*(vk * (1/6) * dt**3 * si)).squeeze()
        Q[:,4,1] = (var_accy*(vk * (1/6) * dt**3 * co)).squeeze()
        Q[:,4,2] = (var_accy*((1/2) * dt**2))
        Q[:,4,3] = 0
        Q[:,4,4] = (var_accy * dt)

        Q = np.tril(Q) + np.transpose(np.tril(Q,k=-1),axes=(0,2,1)) + 1e-10*np.reshape(np.eye(state_dim),(1,state_dim,state_dim))    
        if(batch_size == 1):
            return Q[0,:,:]
        else:
            return Q


class CTRV_Hack(GaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a discrete, time-variant 2D Constant
    Turn Rate and Velocity Model (CTRV).

    The target is assumed to move with (nearly) constant velocity and also
    an unknown (nearly) constant turn rate. This implementation uses a
    state-dependent noise covariance matrix.
    """
    linear_noise_coeff: float = Property(
        doc=r"The linear acceleration noise in the heading direction :math:`q_l`")
    turn_noise_coeff: float = Property(
        doc=r"The turn rate noise coefficient :math:`q_\omega`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = None

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 5


    def function(self, state, noise=False, **kwargs) -> StateVector:
        dt       = kwargs['time_interval'].total_seconds()

        Px       = state.state_vector[0,:]
        Py       = state.state_vector[1,:]
        psi      = state.state_vector[2,:]
        v        = state.state_vector[3,:]
        omega    = state.state_vector[4,:]
    
        #Pxnew    = Px + dt*v*np.cos(psi + dt*omega/2)*np.sinc(dt*omega/2)
        #Pynew    = Py + dt*v*np.sin(psi + dt*omega/2)*np.sinc(dt*omega/2)        
        #psinew   = psi + dt*omega

        w            = dt * omega / 2       # half-angle term
        psi_mid      = psi + w              # mid-point angle
        sinc_val     = np.sinc(w / np.pi)   # numpy's sinc(x) = sin(π*x)/(π*x)
        cos_psi_mid  = np.cos(psi_mid)
        sin_psi_mid  = np.sin(psi_mid)
        Pxnew        = Px + dt*v*cos_psi_mid*sinc_val
        Pynew        = Py + dt*v*sin_psi_mid*sinc_val
        psinew       = psi + dt*omega

        svnew        = StateVectors([Pxnew,Pynew,psinew,v,omega])

        if isinstance(noise, bool) or noise is None:
            if noise:
                kwargs["state_vectors"] = svnew 
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        
        return svnew + noise
    
    def jacobian(self, state, **kwargs):

        dt       = kwargs['time_interval'].total_seconds()

        Px       = state.state_vector[0,:]
        Py       = state.state_vector[1,:]
        psi      = state.state_vector[2,:]
        v        = state.state_vector[3,:]
        omega    = state.state_vector[4,:]

        w            = dt * omega / 2       # half-angle term
        psi_mid      = psi + w              # mid-point angle
        sinc_val     = np.sinc(w / np.pi)   # numpy's sinc(x) = sin(π*x)/(π*x)
        cos_psi_mid  = np.cos(psi_mid)
        sin_psi_mid  = np.sin(psi_mid)

        j = np.zeros((5,5))
        j[0,0] = 1 #dfx/dx
        j[1,1] = 1 #df[y]/dy

        j[0,2] = -dt*v*sin_psi_mid*sinc_val #df[x]/dpsi
        j[1,2] = dt*v*cos_psi_mid*sinc_val #df[y]/dpsi
        j[2,2] = 1 #df[psi]/dpsi
        j[3,2] = 0 #df[v]/dpsi
        j[4,2] = 0 #df[omega]/dpsi

        j[0,3] = dt*cos_psi_mid*sinc_val #df[x]/dv
        j[1,3] = dt*sin_psi_mid*sinc_val #df[y]/dv
        j[2,3] = 0 #df[psi]/dv
        j[3,3] = 1 #df[v]/dv
        j[4,3] = 0 #df[omega]/dv

        if np.abs(w) < 1e-10:
            dsinc_dw = 0.0
        else:
            dsinc_dw = (w * np.cos(w) - np.sin(w)) / (w**2)

        j[0,4] = 0.5*(dt**2)*v*(-sin_psi_mid*sinc_val + cos_psi_mid*dsinc_dw) #df[x]/domega
        j[1,4] = 0.5*(dt**2)*v*(cos_psi_mid*sinc_val + sin_psi_mid*dsinc_dw) #df[y]/domega
        j[2,4] = dt #df[psi]/domega
        j[3,4] = 0 #df[v]/domega
        j[4,4] = 1 #df[omega]/domega

        #Store Q for this state
        kwargs["state_vectors"] = StateVectors([Px,Py,psi,v,omega])
        self.Q = self.covar(**kwargs)

        return j
    
    def rvs(self, num_samples=1, **kwargs):

        Qs = self.covar(**kwargs)
        Qs = Qs.reshape(-1,5,5)

        if(Qs.shape[0]>1):
            num_samples=1
            
        noise = [multivariate_normal(np.zeros(self.ndim), Q).rvs(num_samples).reshape(-1,num_samples) for Q in Qs]

        return np.hstack(noise)

    def covar(self, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        dt       = time_interval.total_seconds()
        var_accl = self.linear_noise_coeff
        var_accy = self.turn_noise_coeff

        state_vectors    = kwargs.get("state_vectors",None)
        if state_vectors is None:
            if(self.Q is None):
                raise ValueError("state_vectors must be provided to compute the covariance matrix if covar is not cached")
            else:
                #Assume Q for the current state has been cached
                #Works for EKF that compute jacobian before covar is called 
                return self.Q

        if(len(state_vectors.shape))==1:
            state_vectors = state_vectors.reshape(-1,1)

        state_dim, batch_size = state_vectors.shape

        if(state_dim != self.ndim_state):
            raise ValueError("state dimension mismatch")

        psik = state_vectors[2,:]
        vk   = state_vectors[3,:]

        si = np.sin(psik)
        co = np.cos(psik)
        cs = np.cos(psik)*np.sin(psik)

        Q = np.zeros((batch_size,state_dim,state_dim))

        Q[:,0,0] = (var_accl * ((1/3) * dt**3 * co**2) + var_accy * ((1/20) * dt**5 * vk**2 * si**2)).squeeze()
        Q[:,1,0] = (var_accl*((1/3) * dt**3 * cs) - var_accy*( (1/20) * dt**5 * vk**2 * cs)).squeeze()
        Q[:,1,1] = (var_accl*((1/3) * dt**3 * si**2) + var_accy*( (1/20) * dt**5 * vk**2 * co**2)).squeeze()
        Q[:,2,0] = (-var_accy*((1/8) * vk * dt**4 * si)).squeeze()
        Q[:,2,1] = (var_accy*((1/8) * vk * dt**4 * co)).squeeze()
        Q[:,2,2] = (var_accy*((1/3) * dt**3))
        Q[:,3,0] = (var_accl*((1/2) * dt**2 * co)).squeeze()
        Q[:,3,1] = (var_accl*((1/2) * dt**2 * si)).squeeze()
        Q[:,3,2] = 0
        Q[:,3,3] = (var_accl * dt)
        Q[:,4,0] = (-var_accy*(vk * (1/6) * dt**3 * si)).squeeze()
        Q[:,4,1] = (var_accy*(vk * (1/6) * dt**3 * co)).squeeze()
        Q[:,4,2] = (var_accy*((1/2) * dt**2))
        Q[:,4,3] = 0
        Q[:,4,4] = (var_accy * dt)

        Q = np.tril(Q) + np.transpose(np.tril(Q,k=-1),axes=(0,2,1)) + 1e-10*np.reshape(np.eye(state_dim),(1,state_dim,state_dim))    
        if(batch_size == 1):
            return Q[0,:,:]
        else:
            return Q

class kinematic_bicycle(GaussianTransitionModel, TimeVariantModel):

    r"""This is a class implementation of a discrete, time-variant 2D kinematic bicycle model.

    The target is assumed to move with (nearly) constant velocity and also
    an unknown (nearly) constant steering angle. This implementation uses a
    state-dependent noise covariance matrix.
    """
    std_accl: float = Property(
        doc=r"The linear acceleration noise in the heading direction :math:`q_l`")
    std_accd: float = Property(
        doc=r"The turn rate noise coefficient :math:`q_\omega`")
    use_circular: bool = Property(
        default=True,
        doc=r"Whether to use circular motion model for angle normalization")
    L: float = Property(
        default=5,
        doc=r"Wheel base length")
    verbose: bool = Property(
        default=True,
        doc=r"Whether to produce verbose output.")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 5

    def function(self, state, noise=False, **kwargs) -> StateVector:
        dt       = kwargs['time_interval'].total_seconds()

        Px       = state.state_vector[0,:]
        Py       = state.state_vector[1,:]
        v        = state.state_vector[2,:]
        theta    = state.state_vector[3,:]
        delta    = state.state_vector[4,:]

        L        = self.L
        l_r      = self.L/2
        
        ta       = np.tan(delta)
        S        = L/ta
        beta     = np.arctan(l_r*ta/L)
        co       = np.cos(beta)
        R        = S/co

        Pxnew    = Px + dt*v*np.cos((beta + theta) + dt*v*ta*co/(2*L))*np.sinc(dt*v*ta*co/(2*L))
        Pynew    = Py + dt*v*np.sin((beta + theta) + dt*v*ta*co/(2*L))*np.sinc(dt*v*ta*co/(2*L))
        vnew     = v  
        
        thetanew  = theta + dt*v*ta*co/L
        #thetanew = (thetanew) % (np.pi*2)  -np.pi
        deltanew  = delta
        deltanew = np.clip(deltanew, -np.pi/4, np.pi/4)

        svnew     = StateVectors([Pxnew,Pynew,vnew,thetanew,deltanew])

        if isinstance(noise, bool) or noise is None:
            if noise:
                kwargs["state_vectors"] = svnew 
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        Pxnew    = Pxnew + noise[0,:]
        Pynew    = Pynew + noise[1,:]
        vnew     = vnew  + noise[2,:]
        thetanew = (thetanew + noise[3,:]) 
        #thetanew = (thetanew % (np.pi*2)) -np.pi
        deltanew = np.clip(deltanew + noise[4,:], -np.pi/4, np.pi/4)

        svnew    = StateVectors([Pxnew,Pynew,vnew,thetanew,deltanew])

        return svnew

    def rvs(self, num_samples=1, **kwargs):

        Qs = self.covar(**kwargs)
        Qs = Qs.reshape(-1,5,5)

        if(Qs.shape[0]>1):
            num_samples=1
            
        noise = [multivariate_normal(np.zeros(self.ndim), Q).rvs(num_samples).reshape(-1,num_samples) for Q in Qs]

        return np.hstack(noise)

    def covar(self, time_interval, **kwargs):
        dt = time_interval.total_seconds()
        state_vectors    = kwargs.get("state_vectors",None)

        q_v = self.std_accl**2
        q_d = self.std_accd**2

        state_vectors    = kwargs.get("state_vectors",None)
        if state_vectors is None:
            raise ValueError("state_vectprs must be provided to compute the covariance matrix")

        if(len(state_vectors.shape))==1:
            state_vectors = state_vectors.reshape(-1,1)

        state_dim, batch_size = state_vectors.shape

        if(state_dim != self.ndim_state):
            raise ValueError("state dimension mismatch")

        Q = np.zeros((batch_size,state_dim,state_dim))


        Px       = state_vectors[0,:]
        Py       = state_vectors[1,:]
        v        = state_vectors[2,:]
        theta    = state_vectors[3,:]
        delta    = state_vectors[4,:]
        L        = self.L
        l_r      = self.L/2
        
        ta       = np.tan(delta)
        S        = L/ta
        beta     = np.arctan(l_r*ta/L)
        co       = np.cos(beta)
        R        = S/co
        sith     = np.sin(theta+beta)
        coth     = np.cos(theta+beta)
        cs       = coth*sith
        sec      = 1/np.cos(delta)
        d_beta   = L * l_r * sec**2/(L**2 + l_r**2 * ta**2)
        m        = sec**2 * co - ta * np.sin(beta) * d_beta
        
        
        Q[:,0,0] = ((1/3) * dt**3 * (q_v * (coth**2) + q_d * (sith**2 * d_beta**2 * v**2)) + (1/4) * dt**4 * (q_v * (-cs * v * ta * co/L) + q_d * (sith**2 * d_beta * v**3 * m/L))  + (1/20) * dt**5 * (q_v * (v**2 * sith**2 * ta**2 * co**2/L**2) + q_d * (v**4 * sith**2  * m**2/L**2))).squeeze() 
        
        Q[:,1,0] = ((1/3) * dt**3 * (q_v * (cs) - q_d * (cs * d_beta**2 * v**2)) + (1/4) * dt**4 * (q_v * (v * ta * co * (coth**2 - sith**2)/(2*L)) + q_d * (-cs * d_beta * v**3 * m/L))  + (1/20) * dt**5 * (q_v * (-cs * v**2 * ta**2 * co**2/L**2) + q_d * (-cs * v**4 * m**2/L**2))).squeeze() 
        
        Q[:,1,1] = ((1/3) * dt**3 * (q_v * (sith**2) + q_d * (coth**2 * d_beta**2 * v**2)) + (1/4) * dt**4 * (q_v * (cs * v * ta * co/L) + q_d * (coth**2 * d_beta * v**3 * m/L))  + (1/20) * dt**5 * (q_v * (v**2 * coth**2 * ta**2 * co**2/L**2) + q_d * (v**4 * coth**2  * m**2/L**2))).squeeze() 
        
        Q[:,2,0] = ((1/2) * dt**2 * q_v * (coth) + (1/6) * dt**3 * q_v * (-sith * v * ta * co/L )).squeeze()
        
        Q[:,2,1] = ((1/2) * dt**2 * q_v * (sith) + (1/6) * dt**3 * q_v * (coth * v * ta * co/L )).squeeze()
        
        Q[:,2,2] = q_v * dt #This is a float scalar
        
        Q[:,3,0] = ((1/3) * dt**3 * (q_v * (coth * ta * co/L) + q_d * (-sith * v**2 * m * d_beta/L)) + (1/4) * dt**4 * (q_v * (-sith * v * ta**2 * co**2/(2 * L**2)) + q_d * (-sith * v**3 * m**2/(2 * L**2)))).squeeze() 
        
        Q[:,3,1] = ((1/3) * dt**3 * (q_v * (sith * ta * co/L) + q_d * (coth * v**2 * m * d_beta/L)) + (1/4) * dt**4 * (q_v * (coth * v * ta**2 * co**2/(2 * L**2)) + q_d * (coth * v**3 * m**2/(2 * L**2)))).squeeze() 
        
        Q[:,3,2] = ((1/2) * dt**2 * q_v * ta * co/L).squeeze()
        
        Q[:,3,3] = ((1/3) * dt**3 * (q_v * (ta**2 * co**2/L**2) + q_d * (v**2 * m**2/L**2))).squeeze()
        
        Q[:,4,0] = ((1/2) * dt**2 * (q_d * (-sith * d_beta * v)) + (1/3) * dt**3 * (q_d * (-sith * v**2 * m/(2*L)))).squeeze() 
        
        Q[:,4,1] = ((1/2) * dt**2 * (q_d * (coth * d_beta * v)) + (1/3) * dt**3 * (q_d * (coth * v**2 * m/(2*L)))).squeeze() 
        
        Q[:,4,2] = 0
        
        Q[:,4,3] = ((1/2) * dt**2 * q_d * v * m/L).squeeze()
        
        Q[:,4,4] = q_d * dt #This is a float scalar

        Q = np.tril(Q) + np.transpose(np.tril(Q,k=-1),axes=(0,2,1)) + 1e-8*np.reshape(np.eye(state_dim),(1,state_dim,state_dim))    
        if(batch_size == 1):
            return Q[0,:,:]
        else:
            return Q



class kinematic_bicycle2(GaussianTransitionModel, TimeVariantModel):

    r"""This is a class implementation of a discrete, time-variant 2D kinematic bicycle model.

    The target is assumed to move with (nearly) constant velocity and also
    an unknown (nearly) constant steering angle. This implementation uses a
    state-dependent noise covariance matrix.
    """
    std_accl: float = Property(
        doc=r"The linear acceleration noise in the heading direction :math:`q_l`")
    std_accd: float = Property(
        doc=r"The turn rate noise coefficient :math:`q_\omega`")
    use_circular: bool = Property(
        default=True,
        doc=r"Whether to use circular motion model for angle normalization")
    L: float = Property(
        default=5,
        doc=r"Wheel base length")
    verbose: bool = Property(
        default=True,
        doc=r"Whether to produce verbose output.")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 5

    def function(self, state, noise=False, **kwargs) -> StateVector:
        dt       = kwargs['time_interval'].total_seconds()

        Px       = state.state_vector[0,:]
        Py       = state.state_vector[1,:]
        v        = state.state_vector[2,:]
        theta    = state.state_vector[3,:]
        delta    = state.state_vector[4,:]

        L = self.L
        l_r = L/2
        ta = np.tan(delta)
        S = L/ta
        beta = np.arctan(l_r*ta/L)
        co = np.cos(beta)
        R = S/co

        Pxnew    = Px + dt*v*np.cos((beta + theta) + dt*v*ta*co/(2*L))*np.sinc(dt*v*ta*co/(2*L))
        Pynew    = Py + dt*v*np.sin((beta + theta) + dt*v*ta*co/(2*L))*np.sinc(dt*v*ta*co/(2*L))
        vnew     = v  
        
        # thetanew   = np.atan2(np.sin(theta + dt*v*ta*co/L),np.cos(theta + dt*v*ta*co/L))
        thetanew   = theta + dt*v*ta*co/L
        deltanew = delta

        svnew     = StateVectors([Pxnew,Pynew,vnew,thetanew,deltanew])

        if isinstance(noise, bool) or noise is None:
            if noise:
                kwargs["state_vectors"] = svnew 
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)

                Pxnew    = Pxnew + noise[0,:]
                Pynew    = Pynew + noise[1,:]
                vnew     = vnew  + noise[2,:]
                thetanew = (thetanew + noise[3,:]) 
                #thetanew = (thetanew % (np.pi*2)) -np.pi
                deltanew = np.clip(deltanew + noise[4,:], -np.pi/4, np.pi/4)
                svnew    = StateVectors([Pxnew,Pynew,vnew,thetanew,deltanew])

        return svnew

    def rvs(self, num_samples=1, **kwargs):

        Qs = self.covar(**kwargs)
        Qs = Qs.reshape(-1,5,5)

        if(Qs.shape[0]>1):
            num_samples=1
            
        noise = [multivariate_normal(np.zeros(self.ndim), Q).rvs(num_samples).reshape(-1,num_samples) for Q in Qs]

        return np.hstack(noise)

    def covar(self, time_interval, **kwargs):
        dt = time_interval.total_seconds()
        state_vectors    = kwargs.get("state_vectors",None)

        q_v = self.std_accl**2
        q_d = self.std_accd**2

        state_vectors    = kwargs.get("state_vectors",None)
        if state_vectors is None:
            raise ValueError("state_vectprs must be provided to compute the covariance matrix")

        if(len(state_vectors.shape))==1:
            state_vectors = state_vectors.reshape(-1,1)

        state_dim, batch_size = state_vectors.shape

        if(state_dim != self.ndim_state):
            raise ValueError("state dimension mismatch")

        Q = np.zeros((batch_size,state_dim,state_dim))

        Px       = state_vectors[0,:]
        Py       = state_vectors[1,:]
        v        = state_vectors[2,:]
        theta    = state_vectors[3,:]
        delta    = state_vectors[4,:]

        L = self.L
        l_r = L/2
        ta = np.tan(delta)
        S = L/ta
        beta = np.arctan(l_r*ta/L)
        co = np.cos(beta)
        R = S/co
        sith = np.sin(theta+beta)
        coth = np.cos(theta+beta)
        cs = coth*sith
        sec = 1/np.cos(delta)
        d_beta = L * l_r * sec**2/(L**2 + l_r**2 * ta**2)
        m = sec**2 * co - ta * np.sin(beta) * d_beta
               
        Q[:,0,0] = ((1/3) * dt**3 * (q_v * (coth**2) + q_d * (sith**2 * d_beta**2 * v**2)) + (1/4) * dt**4 * (q_v * (-cs * v * ta * co/L) + q_d * (sith**2 * d_beta * v**3 * m/L))  + (1/20) * dt**5 * (q_v * (v**2 * sith**2 * ta**2 * co**2/L**2) + q_d * (v**4 * sith**2  * m**2/L**2))).squeeze() 
        
        Q[:,1,0] = ((1/3) * dt**3 * (q_v * (cs) - q_d * (cs * d_beta**2 * v**2)) + (1/4) * dt**4 * (q_v * (v * ta * co * (coth**2 - sith**2)/(2*L)) + q_d * (-cs * d_beta * v**3 * m/L))  + (1/20) * dt**5 * (q_v * (-cs * v**2 * ta**2 * co**2/L**2) + q_d * (-cs * v**4 * m**2/L**2))).squeeze() 
        
        Q[:,1,1] = ((1/3) * dt**3 * (q_v * (sith**2) + q_d * (coth**2 * d_beta**2 * v**2)) + (1/4) * dt**4 * (q_v * (cs * v * ta * co/L) + q_d * (coth**2 * d_beta * v**3 * m/L))  + (1/20) * dt**5 * (q_v * (v**2 * coth**2 * ta**2 * co**2/L**2) + q_d * (v**4 * coth**2  * m**2/L**2))).squeeze() 
        
        Q[:,2,0] = ((1/2) * dt**2 * q_v * (coth) + (1/6) * dt**3 * q_v * (-sith * v * ta * co/L )).squeeze()
        
        Q[:,2,1] = ((1/2) * dt**2 * q_v * (sith) + (1/6) * dt**3 * q_v * (coth * v * ta * co/L )).squeeze()
        
        Q[:,2,2] = (q_v * dt)
        
        Q[:,3,0] = ((1/3) * dt**3 * (q_v * (coth * ta * co/L) + q_d * (-sith * v**2 * m * d_beta/L)) + (1/4) * dt**4 * (q_v * (-sith * v * ta**2 * co**2/(2 * L**2)) + q_d * (-sith * v**3 * m**2/(2 * L**2)))).squeeze() 
        
        Q[:,3,1] = ((1/3) * dt**3 * (q_v * (sith * ta * co/L) + q_d * (coth * v**2 * m * d_beta/L)) + (1/4) * dt**4 * (q_v * (coth * v * ta**2 * co**2/(2 * L**2)) + q_d * (coth * v**3 * m**2/(2 * L**2)))).squeeze() 
        
        Q[:,3,2] = ((1/2) * dt**2 * q_v * ta * co/L).squeeze()
        
        Q[:,3,3] = ((1/3) * dt**3 * (q_v * (ta**2 * co**2/L**2) + q_d * (v**2 * m**2/L**2))).squeeze()
        
        Q[:,4,0] = ((1/2) * dt**2 * (q_d * (-sith * d_beta * v)) + (1/3) * dt**3 * (q_d * (-sith * v**2 * m/(2*L)))).squeeze() 
        
        Q[:,4,1] = ((1/2) * dt**2 * (q_d * (coth * d_beta * v)) + (1/3) * dt**3 * (q_d * (coth * v**2 * m/(2*L)))).squeeze() 
        
        Q[:,4,2] = 0
        
        Q[:,4,3] = ((1/2) * dt**2 * q_d * v * m/L).squeeze()
        
        Q[:,4,4] = (q_d * dt)


        Q = np.tril(Q) + np.transpose(np.tril(Q,k=-1),axes=(0,2,1)) + 1e-8*np.reshape(np.eye(state_dim),(1,state_dim,state_dim))    
        if(batch_size == 1):
            return Q[0,:,:]
        else:
            return Q



class ConstantTurn(GaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a discrete, time-variant 2D Constant
    Turn Model.

    The target is assumed to move with (nearly) constant velocity and also
    an unknown (nearly) constant turn rate.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{align}
                dx_{pos} & =  x_{vel} d  \quad | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = -\omega y_{pos} d \quad | {Speed \
                on\ X-axis (m/s)} &\\
                dy_{pos} & =  y_{vel} d  \quad | {Position \ on \
                Y-axis (m)} \\
                dy_{vel} & =  \omega x_{pos} d \quad | {Speed \
                on\ Y-axis (m/s)} \\
                d\omega & = q_\omega dt  \quad | {Position \ on \ X,Y-axes (rad/sec)}
            \end{align}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel} \\
                        y_{pos} \\
                        y_{vel} \\
                        \omega
                    \end{bmatrix}

        .. math::
            F(x) & = & \begin{bmatrix}
                          1 & \frac{\sin\omega dt}{\omega} & 0 & -
                            \frac{(1-\cos\omega dt)}{\omega} & 0 \\
                          0 & \cos\omega dt & 0 & - \sin\omega dt & 0 \\
                          0 & \frac{(1-\cos\omega dt)}{\omega} & 1 &
                            \frac{\sin\omega dt}{\omega} & 0 \\
                          0 & \sin\omega dt & 0 & \sin\omega dt & 0 \\
                          0 & 0 & 0 & 0 & 1
                      \end{bmatrix}

        .. math::
             Q_t & = & \begin{bmatrix}
                          q_x\frac{dt^3}{3} & q_x\frac{dt^2}{2} & 0 & 0 & 0 \\
                          q_x\frac{dt^2}{2} & q_xdt & 0 & 0 & 0 \\
                          0 & 0 & q_y\frac{dt^3}{3} & q_y\frac{dt^2}{2} & 0 \\
                          0 & 0 & q_y\frac{dt^2}{2} & q_ydt & 0 \\
                          0 & 0 & 0 & 0 & q_\omega dt
                     \end{bmatrix}
    """
    linear_noise_coeffs: np.ndarray = Property(
        doc=r"The acceleration noise diffusion coefficients :math:`[q_x, \: q_y]^T`")
    turn_noise_coeff: float = Property(
        doc=r"The turn rate noise coefficient :math:`q_\omega`")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 5

    def function(self, state, noise=False, **kwargs) -> StateVector:
        time_interval_sec = kwargs['time_interval'].total_seconds()
        sv1 = state.state_vector
        turn_rate = sv1[4, :]
        # Avoid divide by zero in the function evaluation
        if turn_rate.dtype != float:
            turn_rate = turn_rate.astype(float)
        turn_rate[turn_rate == 0.] = np.finfo(float).eps
        dAngle = turn_rate * time_interval_sec
        cos_dAngle = np.cos(dAngle)
        sin_dAngle = np.sin(dAngle)
        sv2 = StateVectors(
            [sv1[0, :] + sin_dAngle/turn_rate * sv1[1, :] - sv1[3, :] / turn_rate *
             (1. - cos_dAngle),
             sv1[1, :] * cos_dAngle - sv1[3, :] * sin_dAngle,
             sv1[1, :] / turn_rate * (1. - cos_dAngle) + sv1[2, :] + sv1[3, :] * sin_dAngle
             / turn_rate,
             sv1[1, :] * sin_dAngle + sv1[3, :] * cos_dAngle,
             turn_rate])
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        return sv2 + noise

    def covar(self, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """
        q_x, q_y = self.linear_noise_coeffs
        q = self.turn_noise_coeff
        dt = abs(time_interval.total_seconds())

        Q = np.array([[dt**3 / 3., dt**2 / 2.],
                      [dt**2 / 2., dt]])
        C = block_diag(Q*q_x, Q*q_y, dt*q)

        return CovarianceMatrix(C)


class ConstantTurnSandwich(ConstantTurn):
    r"""This is a class implementation of a time-variant 2D Constant Turn
    Model. This model is used, as opposed to the normal :class:`~.ConstantTurn`
    model, when the turn occurs in 2 dimensions that are not adjacent in the
    state vector, eg if the turn occurs in the x-z plane but the state vector
    is of the form :math:`(x,y,z)`. The list of transition models are to be
    applied to any state variables that lie in between, eg if for the above
    example you wanted the y component to move with constant velocity, you
    would put a :class:`~.ConstantVelocity` model in the list.

    The target is assumed to move with (nearly) constant velocity and also
    unknown (nearly) constant turn rate.
    """
    model_list: Sequence[GaussianTransitionModel] = Property(
        doc="List of Transition Models.")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list) + 5

    def function(self, state, noise=False, **kwargs) -> StateVector:
        state_tmp = copy.copy(state)
        sv_in = state.state_vector
        sv1 = np.concatenate((sv_in[0:2, 0:], sv_in[-3:, 0:]))
        state_tmp.state_vector = sv1
        # Calculate state vector for CT model
        sv_ct = super().function(state_tmp, noise=False, **kwargs)

        # Calculate state vector for model list
        idx1 = 2
        sv_list = [sv_ct[0:2, 0:]]
        for model in self.model_list:
            idx2 = idx1 + model.ndim
            state_tmp.state_vector = sv_in[idx1:idx2, 0:]
            sv_list.append(model.function(state_tmp, noise=False, **kwargs))
            idx1 = idx2
        sv_list.append(sv_ct[-3:, 0:])
        sv_out = StateVectors(np.concatenate(sv_list))
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        return sv_out + noise

    def covar(self, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """
        C_t = np.zeros([self.ndim, self.ndim])
        C_ct = super().covar(time_interval, **kwargs)
        covar_list = [model.covar(time_interval) for model in self.model_list]

        # Assemble diag block components
        C_t[2:-3, 2:-3] = block_diag(*covar_list)
        C_t[0:2, 0:2] = C_ct[0:2, 0:2]
        C_t[-3:, -3:] = C_ct[-3:, -3:]
        # Reorder offdiagonal elements
        C_t[0:2:, -3:] = C_ct[0:2, -3:]
        C_t[-3:, 0:2] = C_ct[-3:, 0:2]

        return CovarianceMatrix(C_t)
