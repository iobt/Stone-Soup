import copy
from collections.abc import Sequence

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from .base import TransitionModel
from ..base import GaussianModel, TimeVariantModel
from ...base import Property
from ...types.array import CovarianceMatrix, StateVector, StateVectors


class GaussianTransitionModel(TransitionModel, GaussianModel):
    pass



class CTRV(GaussianTransitionModel, TimeVariantModel):
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

        Pxnew    = Px + dt*v*np.cos(psi + dt*omega/2)*np.sinc(dt*omega/2)
        Pynew    = Py + dt*v*np.sin(psi + dt*omega/2)*np.sinc(dt*omega/2)        
        psinew   = psi + dt*omega

        svnew    = StateVectors([Pxnew,Pynew,psinew,v,omega])

        if isinstance(noise, bool) or noise is None:
            if noise:
                kwargs["state_vectors"] = svnew 
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        #print(svnew.T)
        #print(noise.T)
        #print()
        
        return svnew + noise
    
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
            raise ValueError("state_vectprs must be provided to compute the covariance matrix")

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
