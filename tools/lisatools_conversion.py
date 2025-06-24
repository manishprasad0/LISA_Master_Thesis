#
# Copyright (C) 2019 Sylvain Marsat.
#
#


"""
    Python tools for LISA: L-frame, Jacobian, degenerate sky positions.
"""

from __future__ import absolute_import, division, print_function
import sys
if sys.version_info[0] == 2:
    from future_builtins import map, filter


import copy
import numpy as np

import scipy
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline


def modpi(x):
    return np.remainder(x, np.pi)


# import lisabeta.tools.pytools as pytools
# import lisabeta.pyconstants as pyconstants
# import lisabeta.lisa.pyresponse as pyresponse

# Constants
R = 1.49597870700e11 #pyconstants.AU_SI
c = 299792458. #pyconstants.C_SI
Omega = 1.9909865927683785e-07 #pyconstants.EarthOrbitOmega_SI
msols = 4.925491025543575903411922162094833998e-6 #pyconstants.MTSUN_SI

################################################################################
# Functions for the L-frame conversion for LISA
################################################################################

# Conversions between SSB-frame and L-frame parameters (for initial position alpha=0)
def functLfromtSSB(tSSB, lambd, beta):
    return tSSB - R/c*np.cos(beta)*np.cos(Omega*tSSB - lambd)
def functSSBfromtL(tL, lambd, beta):
    return tL + R/c*np.cos(beta)*np.cos(Omega*tL - lambd) - 1./2*Omega*(R/c*np.cos(beta))**2*np.sin(2.*(Omega*tL - lambd))
# Option SetphiRefSSBAtfRef indicates the meaning of phase in the input SSB frame
# L-frame phase is always a pure observer phase shift, equivalent to having SetphiRefSSBAtfRef=False AND flipping the sign
# So phiL = -phiSSB(SetphiRefSSBAtfRef=False)
# We set phiL' = -phiSSB(SetphiRefSSBAtfRef=True) + pi*tSSB*fRef to remove the tRef-dependence
# NOTE: phiL != phiL'
# the two still differ by fRef-dependent terms that require access to the waveform to evaluate
# NOTE: phiL' only supports fRef set at its default, the maximum 22 frequency covered by the ROM
# NOTE: fliphase=False is to be used when the SSB phase is already 'correct' (phiRef=phi0 entering the Ylm in code updated from pyFDresponse), in which case we do not touch the phase -- using both SetphiRefSSBAtfRef=True and flipphase=False will not really be use, but we allow it
def funcphiL(m1, m2, tSSB, phiSSB, SetphiRefSSBAtfRef=False, flipphase=True): # note: mod to [0,2pi]
    if not SetphiRefSSBAtfRef:
        if flipphase:
            return -phiSSB
        else:
            return phiSSB
    else:
        MfROMmax22 = 0.14
        fRef = MfROMmax22/((m1 + m2)*msols)
        if flipphase:
            return pytools.mod2pi(-phiSSB + np.pi*tSSB*fRef)
        else:
            return pytools.mod2pi(phiSSB + np.pi*tSSB*fRef)
# Inverse transformation of the phase
# NOTE: we take tSSB as an argument, not tL - because computing tSSB requires the sky position as well
def funcphiSSB(m1, m2, tSSB, phiL, SetphiRefSSBAtfRef=False, flipphase=True): # note: mod to [0,2pi]
    if not SetphiRefSSBAtfRef:
        if flipphase:
            return -phiL
        else:
            return phiL
    else:
        MfROMmax22 = 0.14
        fRef = MfROMmax22/((m1 + m2)*msols)
        if flipphase:
            return pytools.mod2pi(-phiL + np.pi*tSSB*fRef)
        else:
            return pytools.mod2pi(phiL - np.pi*tSSB*fRef)
# NOTE: simple relation between L-frame definitions
# lambdaL_old = lambdaL_paper - pi/2
def funclambdaL(lambd, beta, defLframe='paper'):
    if defLframe=='paper':
        return np.arctan2(np.cos(beta)*np.sin(lambd), np.cos(beta)*np.cos(lambd)*np.cos(np.pi/3) + np.sin(beta)*np.sin(np.pi/3))
    elif defLframe=='old':
        return -np.arctan2(np.cos(beta)*np.cos(lambd)*np.cos(np.pi/3) + np.sin(beta)*np.sin(np.pi/3), np.cos(beta)*np.sin(lambd))
    else:
        raise ValueError('Value %s for defLframe is not recognized.' % defLframe)
def funcbetaL(lambd, beta):
    return -np.arcsin(np.cos(beta)*np.cos(lambd)*np.sin(np.pi/3) - np.sin(beta)*np.cos(np.pi/3))
# NOTE: old equivalent writing
# modpi(np.arctan2(np.cos(np.pi/3)*np.cos(beta)*np.sin(psi) - np.sin(np.pi/3)*(np.sin(lambd)*np.cos(psi) - np.cos(lambd)*np.sin(beta)*np.sin(psi)), np.cos(np.pi/3)*np.cos(beta)*np.cos(psi) + np.sin(np.pi/3)*(np.sin(lambd)*np.sin(psi) + np.cos(lambd)*np.sin(beta)*np.cos(psi))))
def funcpsiL(lambd, beta, psi): # note: mod to [0,pi]
    return pytools.modpi(psi + np.arctan2(-np.sin(np.pi/3)*np.sin(lambd), np.cos(np.pi/3)*np.cos(beta) + np.sin(np.pi/3)*np.cos(lambd)*np.sin(beta)))

# Copy of C functions for translation between frames
# We modify constellation variant, keeping only the initial constellation phase : Omega is fixed, constellation_ini_phase replaces variant->ConstPhi0
# NOTE: C function for time duplicate python functions functLfromtSSB and functSSBfromtL
# Compute Solar System Barycenter time tSSB from retarded time at the center of the LISA constellation tL
# NOTE: depends on the sky position given in SSB parameters
def tSSBfromLframe(tL, lambdaSSB, betaSSB, constellation_ini_phase=0., frozenLISA=False, tfrozenLISA=None):
    if frozenLISA:
        if tfrozenLISA is None:
            alpha = constellation_ini_phase
        else:
            alpha = Omega * (tfrozenLISA) + constellation_ini_phase
    else:
        alpha = Omega * (tL) + constellation_ini_phase
    phase = alpha - lambdaSSB
    RoC = R/c
    if frozenLISA:
        correction_implicit = 0.
    else:
        correction_implicit = -1./2*Omega*pow(RoC*np.cos(betaSSB), 2)*np.sin(2.*phase)
    return tL + RoC*np.cos(betaSSB)*np.cos(phase) + correction_implicit;
# Compute retarded time at the center of the LISA constellation tL from Solar System Barycenter time tSSB */
def tLfromSSBframe(tSSB, lambdaSSB, betaSSB, constellation_ini_phase=0., frozenLISA=False, tfrozenLISA=None):
    if frozenLISA:
        if tfrozenLISA is None:
            alpha = constellation_ini_phase
        else:
            alpha = Omega * (tfrozenLISA) + constellation_ini_phase
    else:
        alpha = Omega * (tSSB) + constellation_ini_phase
    phase = alpha - lambdaSSB
    RoC = R/c
    return tSSB - RoC*np.cos(betaSSB)*np.cos(phase)
# Convert L-frame params to SSB-frame params
# NOTE: no transformation of the phase -- approximant-dependence with e.g. EOBNRv2HMROM setting phiRef at fRef, and freedom in definition
def ConvertLframeParamsToSSBframe(
    tL,
    lambdaL,
    betaL,
    psiL,
    constellation_ini_phase=0.,
    tSSB = None,
    frozenLISA=False,
    tfrozenLISA=None):

    alpha = 0.; cosalpha = 0; sinalpha = 0.; coslambdaL = 0; sinlambdaL = 0.; cosbetaL = 0.; sinbetaL = 0.; cospsiL = 0.; sinpsiL = 0.;
    coszeta = np.cos(np.pi/3.)
    sinzeta = np.sin(np.pi/3.)
    coslambdaL = np.cos(lambdaL)
    sinlambdaL = np.sin(lambdaL)
    cosbetaL = np.cos(betaL)
    sinbetaL = np.sin(betaL)
    cospsiL = np.cos(psiL)
    sinpsiL = np.sin(psiL)
    lambdaSSB_approx = 0.
    betaSSB_approx = 0.
    # Initially, approximate alpha using tL instead of tSSB - then iterate
    # NOTE: iteration is not useful when using frozenLISA - will just repeat
    if tSSB is None:
        tSSB_approx = tL
        for k in range(3):
            if frozenLISA:
                if tfrozenLISA is None:
                    alpha = constellation_ini_phase
                else:
                    alpha = Omega * (tfrozenLISA) + constellation_ini_phase
            else:
                alpha = Omega * (tSSB_approx) + constellation_ini_phase
            cosalpha = np.cos(alpha)
            sinalpha = np.sin(alpha)
            lambdaSSB_approx = np.arctan2(cosalpha*cosalpha*cosbetaL*sinlambdaL -sinalpha*sinbetaL*sinzeta + cosbetaL*coszeta*sinalpha*sinalpha*sinlambdaL -cosalpha*cosbetaL*coslambdaL*sinalpha + cosalpha*cosbetaL*coszeta*coslambdaL*sinalpha, cosbetaL*coslambdaL*sinalpha*sinalpha -cosalpha*sinbetaL*sinzeta + cosalpha*cosalpha*cosbetaL*coszeta*coslambdaL -cosalpha*cosbetaL*sinalpha*sinlambdaL + cosalpha*cosbetaL*coszeta*sinalpha*sinlambdaL)
            betaSSB_approx = np.arcsin(coszeta*sinbetaL + cosalpha*cosbetaL*coslambdaL*sinzeta + cosbetaL*sinalpha*sinzeta*sinlambdaL)
            tSSB_approx = tSSBfromLframe(tL, lambdaSSB_approx, betaSSB_approx, constellation_ini_phase=constellation_ini_phase, frozenLISA=frozenLISA, tfrozenLISA=tfrozenLISA)
        tSSB = tSSB_approx
        lambdaSSB = lambdaSSB_approx
        betaSSB = betaSSB_approx
    else:
        if frozenLISA:
            if tfrozenLISA is None:
                alpha = constellation_ini_phase
            else:
                alpha = Omega * (tfrozenLISA) + constellation_ini_phase
        else:
            alpha = Omega * (tSSB) + constellation_ini_phase
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        lambdaSSB = np.arctan2(cosalpha*cosalpha*cosbetaL*sinlambdaL -sinalpha*sinbetaL*sinzeta + cosbetaL*coszeta*sinalpha*sinalpha*sinlambdaL -cosalpha*cosbetaL*coslambdaL*sinalpha + cosalpha*cosbetaL*coszeta*coslambdaL*sinalpha, cosbetaL*coslambdaL*sinalpha*sinalpha -cosalpha*sinbetaL*sinzeta + cosalpha*cosalpha*cosbetaL*coszeta*coslambdaL -cosalpha*cosbetaL*sinalpha*sinlambdaL + cosalpha*cosbetaL*coszeta*sinalpha*sinlambdaL)
        betaSSB = np.arcsin(coszeta*sinbetaL + cosalpha*cosbetaL*coslambdaL*sinzeta + cosbetaL*sinalpha*sinzeta*sinlambdaL) 
    # Polarization
    psiSSB = modpi(psiL + np.arctan2(cosalpha*sinzeta*sinlambdaL -coslambdaL*sinalpha*sinzeta, cosbetaL*coszeta -cosalpha*coslambdaL*sinbetaL*sinzeta -sinalpha*sinbetaL*sinzeta*sinlambdaL))
    return [tSSB, lambdaSSB, betaSSB, psiSSB]
# Convert SSB-frame params to L-frame params
# NOTE: no transformation of the phase -- approximant-dependence with e.g. EOBNRv2HMROM setting phiRef at fRef, and freedom in definition
def ConvertSSBframeParamsToLframe(
    tSSB,
    lambdaSSB,
    betaSSB,
    psiSSB,
    constellation_ini_phase=0.,
    frozenLISA=False,
    tfrozenLISA=None):

    alpha = 0.; cosalpha = 0; sinalpha = 0.; coslambda = 0; sinlambda = 0.; cosbeta = 0.; sinbeta = 0.; cospsi = 0.; sinpsi = 0.;
    coszeta = np.cos(np.pi/3.)
    sinzeta = np.sin(np.pi/3.)
    coslambda = np.cos(lambdaSSB)
    sinlambda = np.sin(lambdaSSB)
    cosbeta = np.cos(betaSSB)
    sinbeta = np.sin(betaSSB)
    cospsi = np.cos(psiSSB)
    sinpsi = np.sin(psiSSB)
    if frozenLISA:
        if tfrozenLISA is None:
            alpha = constellation_ini_phase
        else:
            alpha = Omega * (tfrozenLISA) + constellation_ini_phase
    else:
        alpha = Omega * (tSSB) + constellation_ini_phase
    cosalpha = np.cos(alpha)
    sinalpha = np.sin(alpha)
    tL = tLfromSSBframe(tSSB, lambdaSSB, betaSSB, constellation_ini_phase=constellation_ini_phase, frozenLISA=frozenLISA, tfrozenLISA=tfrozenLISA)
    lambdaL = np.arctan2(cosalpha*cosalpha*cosbeta*sinlambda + sinalpha*sinbeta*sinzeta + cosbeta*coszeta*sinalpha*sinalpha*sinlambda -cosalpha*cosbeta*coslambda*sinalpha + cosalpha*cosbeta*coszeta*coslambda*sinalpha, cosalpha*sinbeta*sinzeta + cosbeta*coslambda*sinalpha*sinalpha + cosalpha*cosalpha*cosbeta*coszeta*coslambda -cosalpha*cosbeta*sinalpha*sinlambda + cosalpha*cosbeta*coszeta*sinalpha*sinlambda)
    betaL = np.arcsin(coszeta*sinbeta -cosalpha*cosbeta*coslambda*sinzeta -cosbeta*sinalpha*sinzeta*sinlambda)
    psiL = modpi(psiSSB + np.arctan2(coslambda*sinalpha*sinzeta -cosalpha*sinzeta*sinlambda, cosbeta*coszeta + cosalpha*coslambda*sinbeta*sinzeta + sinalpha*sinbeta*sinzeta*sinlambda))
    return [tL, lambdaL, betaL, psiL]
# Wrapper functions for frame conversion, used with dictionary of parameters
# NOTE: t0 in yr
# def convert_Lframe_to_SSBframe(params_Lframe, t0=0., frozenLISA=False, LISAconst=pyresponse.LISAconstProposal):
#     if not (params_Lframe.get('Lframe', False)):
#         raise ValueError('Input params already given in the SSB-frame.')
#     t0_s = t0 * pyconstants.YRSID_SI
#     tL = params_Lframe['Deltat'] + t0_s
#     lambdaL = params_Lframe['lambda']
#     betaL = params_Lframe['beta']
#     psiL = params_Lframe['psi']
#     tSSB, lambdaSSB, betaSSB, psiSSB = ConvertLframeParamsToSSBframe(
#         tL,
#         lambdaL,
#         betaL,
#         psiL,
#         constellation_ini_phase=0.,
#         frozenLISA=frozenLISA,
#         tfrozenLISA=t0_s)
#     params_SSBframe = copy.deepcopy(params_Lframe)
#     params_SSBframe['Deltat'] = tSSB - t0_s
#     params_SSBframe['lambda'] = lambdaSSB
#     params_SSBframe['beta'] = betaSSB
#     params_SSBframe['psi'] = psiSSB
#     params_SSBframe['Lframe'] = False
#     return params_SSBframe
# NOTE: t0 in yr
# def convert_SSBframe_to_Lframe(params_SSBframe, t0=0., frozenLISA=False, LISAconst=pyresponse.LISAconstProposal):
#     if(params_SSBframe.get('Lframe', False)):
#         raise ValueError('Input params already given in the L-frame.')
#     t0_s = t0 * pyconstants.YRSID_SI
#     tSSB = params_SSBframe['Deltat'] + t0_s
#     lambdaSSB = params_SSBframe['lambda']
#     betaSSB = params_SSBframe['beta']
#     psiSSB = params_SSBframe['psi']
#     tL, lambdaL, betaL, psiL = ConvertSSBframeParamsToLframe(
#         tSSB,
#         lambdaSSB,
#         betaSSB,
#         psiSSB,
#         constellation_ini_phase=0.,
#         frozenLISA=frozenLISA,
#         tfrozenLISA=t0_s)
#     params_Lframe = copy.deepcopy(params_SSBframe)
#     params_Lframe['Deltat'] = tL - t0_s
#     params_Lframe['lambda'] = lambdaL
#     params_Lframe['beta'] = betaL
#     params_Lframe['psi'] = psiL
#     params_Lframe['Lframe'] = True
#     return params_Lframe
# Compute DeltatL_cut for a given time to merger
# When DeltatSSB is given, we have to compute DeltatL first
# time_to_merger > 0 in s
def DeltatL_cut_from_time_to_merger(time_to_merger, params, **waveform_params):
    if time_to_merger < 0:
        raise ValueError('Must have time_to_merger >= 0, and given in s.')
    Lframe = params.get('Lframe', False)
    # DeltatL (+t0*yr) is the arrival time of the merger at LISA
    if Lframe:
        DeltatL = params['Deltat']
    else:
        DeltatL = tLfromSSBframe(params['Deltat'] + waveform_params['t0']*pyconstants.YRSID_SI, params['lambda'], params['beta']) - waveform_params['t0']*pyconstants.YRSID_SI
    return DeltatL - time_to_merger

################################################################################
# Sky modes in the L-frame
################################################################################

# 8-modes sky degeneracy, index=[a,b] for betaL *=a and lambdaL += b*pi/2
def skymode_degen(params0, index):
    params = params0.copy()
    if not params.get('Lframe', False):
        raise ValueError('Input parameters are not in the L-frame.')
    if index[0]==-1:
        params['beta'] = -params['beta']
        params['inc'] = np.pi - params['inc']
        params['psi'] = np.pi - params['psi']
    elif index[0]==1:
        pass
    else:
        raise ValueError('Invalid sky mode index.')
    params['lambda'] = params['lambda'] + index[1]*np.pi/2
    params['psi'] = params['psi'] + index[1]*np.pi/2
    params['lambda'] = pytools.mod2pi(params['lambda'])
    params['psi'] = pytools.modpi(params['psi'])
    return params

################################################################################
# Derivatives and Jacobian in the SSB-frame L-frame parameter map
################################################################################

# Derivatives of L-frame parameters with respect to SSB-frame parameters
# tL
def funcdtLdtSSB(tSSB, lambd, beta):
    return 1 + R/c * np.cos(beta) * Omega*np.sin(Omega*tSSB - lambd)
def funcdtLdlambda(tSSB, lambd, beta):
    return -R/c * np.cos(beta) * np.sin(Omega*tSSB - lambd)
def funcdtLdbeta(tSSB, lambd, beta):
    return R/c * np.sin(beta) * np.cos(Omega*tSSB - lambd)
# phiL
# Old flare setup: phi was tied to other parameters via phi(fref) = phiref
# Option SetphiRefSSBAtfRef indicates the meaning of phase in the input SSB frame
# L-frame phase is always a pure observer phase phase shift, equivalent to having SetphiRefSSBAtfRef=False AND flipping the sign
# So phiL = -phiSSB(SetphiRefSSBAtfRef=False)
# def funcdphiLdm1(m1, m2, t, phi, SetphiRefSSBAtfRef=False):
#     if SetphiRefSSBAtfRef:
#         MfROMmax22 = 0.14
#         dfRefdm1 = -MfROMmax22/((m1 + m2)**2 * msols)
#         return np.pi*t*dfRefdm1
#     else:
#         return 0.
# def funcdphiLdm2(m1, m2, t, phi, SetphiRefSSBAtfRef=False):
#     if SetphiRefSSBAtfRef:
#         MfROMmax22 = 0.14
#         dfRefdm2 = -MfROMmax22/((m1 + m2)**2 * msols)
#         return np.pi*t*dfRefdm2
#     else:
#         return 0.
# def funcdphiLdt(m1, m2, t, phi, SetphiRefSSBAtfRef=False):
#     if SetphiRefSSBAtfRef:
#         MfROMmax22 = 0.14
#         fRef = MfROMmax22/((m1 + m2)*msols)
#         return np.pi*fRef
#     else:
#         return 0.
# def funcdphiLdphi(m1, m2, t, phi, SetphiRefSSBAtfRef=False):
#     return -1
# lambdaL
# NOTE: derivative identical for different L-frame conventions that differ by a constant shift in lambdaL
def funcdlambdaLdlambda(lambd, beta):
    return (np.cos(beta)*(np.cos(beta)*np.cos(np.pi/3) + np.cos(lambd)*np.sin(beta)*np.sin(np.pi/3))) / ((np.cos(beta)*np.cos(np.pi/3)*np.cos(lambd) + np.sin(beta)*np.sin(np.pi/3))**2 +
 np.cos(beta)**2 * np.sin(lambd)**2)
def funcdlambdaLdbeta(lambd, beta):
    return -((np.sin(np.pi/3)*np.sin(lambd))/((np.cos(beta)*np.cos(np.pi/3)*np.cos(lambd) + np.sin(beta)*np.sin(np.pi/3))**2 +
  np.cos(beta)**2*np.sin(lambd)**2))
# betaL
def funcdbetaLdlambda(lambd, beta):
    return np.cos(beta)*np.sin(np.pi/3)*np.sin(lambd) / np.sqrt(1 - (np.cos(np.pi/3)*np.sin(beta) - np.cos(beta)*np.cos(lambd)*np.sin(np.pi/3))**2)
def funcdbetaLdbeta(lambd, beta):
    return (np.cos(beta)*np.cos(np.pi/3) + np.cos(lambd)*np.sin(beta)*np.sin(np.pi/3)) / np.sqrt(1 - (np.cos(np.pi/3)*np.sin(beta) - np.cos(beta)*np.cos(lambd)*np.sin(np.pi/3))**2)
# psiL
def funcdpsiLdlambda(lambd, beta, psi):
    return -((np.sin(np.pi/3)*(np.cos(beta)*np.cos(np.pi/3)*np.cos(lambd) + np.sin(beta)*np.sin(np.pi/3))) / ((np.cos(beta)*np.cos(np.pi/3) + np.cos(lambd)*np.sin(beta)*np.sin(np.pi/3))**2 + np.sin(np.pi/3)**2 * np.sin(lambd)**2))
def funcdpsiLdbeta(lambd, beta, psi):
    return (np.sin(np.pi/3)*(-np.cos(np.pi/3)*np.sin(beta) + np.cos(beta)*np.cos(lambd)*np.sin(np.pi/3))*np.sin(lambd)) / ((np.cos(beta)*np.cos(np.pi/3) + np.cos(lambd)*np.sin(beta)*np.sin(np.pi/3))**2 +
 np.sin(np.pi/3)**2 * np.sin(lambd)**2)
def funcdpsiLdpsi(lambd, beta, psi):
    return 1.
# Jacobian matrix of the SSB-Lframe transformation
# Required to convert Fisher matrices computed with SSB params to L-frame params
# Parameters order : m1 m2 t D phi inc lambda beta psi
# Matrix Jij = \partial xLi / \partial xj
# In old setup for flare:
# Option SetphiRefSSBAtfRef indicates the meaning of phase in the input SSB frame
# L-frame phase is always a pure phase shift, like having SetphiRefSSBAtfRef=False
# In new setup for lisabeta, no need for option SetphiRefSSBAtfRef
# params is a dictionary, list_params gives the ordering of params
def funcJacobianSSBtoLframe(params, list_params, t0=0.):
    it = list_params.index('Deltat')
    ilambda = list_params.index('lambda')
    ibeta = list_params.index('beta')
    ipsi = list_params.index('psi')
    # SSB parameters
    if params.get('Lframe', False):
        raise ValueError('Input parameters are given in the L-frame.')
    t = params['Deltat'] + t0*pyconstants.YRSID_SI
    lambd = params['lambda']
    beta = params['beta']
    psi = params['psi']
    # Jacobian matrix
    n = len(list_params)
    J = np.identity(n, dtype=float)
    # tL
    J[it,it] = funcdtLdtSSB(t, lambd, beta)
    J[it,ilambda] = funcdtLdlambda(t, lambd, beta)
    J[it,ibeta] = funcdtLdbeta(t, lambd, beta)
    # phi - no interdependency in the lisabeta setup
    # lambdaL
    J[ilambda,ilambda] = funcdlambdaLdlambda(lambd, beta)
    J[ilambda,ibeta] = funcdlambdaLdbeta(lambd, beta)
    # betaL
    J[ibeta,ilambda] = funcdbetaLdlambda(lambd, beta)
    J[ibeta,ibeta] = funcdbetaLdbeta(lambd, beta)
    # psiL
    J[ipsi,ilambda] = funcdpsiLdlambda(lambd, beta, psi)
    J[ipsi,ibeta] = funcdpsiLdbeta(lambd, beta, psi)
    J[ipsi,ipsi] = funcdpsiLdpsi(lambd, beta, psi)
    return J

def funcJacobianMqtom1m2(params, list_params):
    iM = list_params.index('M')
    iq = list_params.index('q')
    # Input mass parameters
    m1 = params['m1']
    m2 = params['m2']
    M = m1 + m2
    q = m1 / m2
    # Jacobian matrix
    n = len(list_params)
    J = np.identity(n, dtype=float)
    J[iM,iM] = q/(1+q)
    J[iq,iM] = 1/(1+q)
    J[iM,iq] = M/(1+q)**2
    J[iq,iq] = -M/(1+q)**2
    list_par = copy.deepcopy(list_params)
    list_par[iM] = 'm1'
    list_par[iq] = 'm2'
    return J, list_par

################################################################################
# Converting Fisher matrices and covariance to different sets of parameters
################################################################################

# Fisher matrix and covariance conversions
# Fisher matrix : if J_{a',b} = \partial xL^a' / \partial x^b Jacobian matrix
# F' = tJ^-1 . F . J^-1
# C' = (F^-1)' = J . C . tJ
# Convert SSB-frame covariance to L-frame
# def convert_fisher_covariance_Lframe(fishcov, t0=0., frozenLISA=False,
#                                         LISAconst=pyresponse.LISAconstProposal):
#     if fishcov['Lframe']:
#         raise ValueError('Fisher/Covariance already in the L-frame.')
#     fishcov_Lframe = copy.deepcopy(fishcov)
#     J = funcJacobianSSBtoLframe(fishcov['params'],
#                                 fishcov['list_params'], t0=t0)
#     Jinv = np.linalg.inv(J)
#     fishcov_Lframe['params'] = convert_SSBframe_to_Lframe(fishcov['params'],
#                               t0=t0, frozenLISA=frozenLISA, LISAconst=LISAconst)
#     fishcov_Lframe['fisher'] = np.dot(Jinv.T, np.dot(fishcov['fisher'], Jinv))
#     fishcov_Lframe['cov'] = np.dot(J, np.dot(fishcov['cov'], J.T))
#     fishcov_Lframe['Lframe'] = True
#     return fishcov_Lframe
# Convert (M,q) to (m1,m2)
def convert_fisher_covariance_Mq_to_m1m2(fishcov):
    fishcov_m1m2 = copy.deepcopy(fishcov)
    J, list_par = funcJacobianMqtom1m2(fishcov['params'], fishcov['list_params'])
    Jinv = np.linalg.inv(J)
    fishcov_m1m2['fisher'] = np.dot(Jinv.T, np.dot(fishcov['fisher'], Jinv))
    fishcov_m1m2['cov'] = np.dot(J, np.dot(fishcov['cov'], J.T))
    fishcov_m1m2['list_params'] = list_par
    return fishcov_m1m2

################################################################################
# Sky area error computed from Fisher matrix
################################################################################

def sky_area_cov(fishercov, sq_deg=False, n_sigma=None, prob=0.90):
    """ Compute sky area error from FiM
    Keyword args:
        sq_deg: convert result to square degrees
        n_sigma: ellipse area corresponds to n_sigma*sigma Gaussian contour
        prob: ellipse area corresponds to the enclosed probability prob
        Note that n_sigma and prob are mutually exclusive.
    """
    beta = fishercov['params']['beta']
    ilambda = fishercov['list_params'].index('lambda')
    ibeta = fishercov['list_params'].index('beta')
    cov = fishercov['cov']
    sigmalambdalambda = cov[ilambda,ilambda]
    sigmabetabeta = cov[ibeta,ibeta]
    sigmalambdabeta = cov[ilambda,ibeta]
    # Determine the prefactor for the desired enclosed probability weight
    # Two options, nb of sigmas or probability, mutually exclusive
    prefactor = 1.
    if not n_sigma is None:
        if not prob is None:
            raise ValueError('n_sigma and prob mutually exclusive.')
        prefactor = n_sigma**2
    else:
        if prob is None:
            raise ValueError('Either one of n_sigma or prob must be set.')
        else:
            prefactor = -2*np.log(1. - prob)
    # In sr
    DeltaOmega = prefactor * np.pi * np.cos(beta) *\
                 np.sqrt(sigmalambdalambda*sigmabetabeta - sigmalambdabeta**2)
    # If asked, convert to sq. deg.
    if sq_deg:
        DeltaOmega = DeltaOmega * (180./np.pi)**2
    return DeltaOmega

################################################################################
# Simplified likelihood
################################################################################

# Pattern functions for A,E channels, see Eq. (43) in arXiv:2003.00357
def Faplus(lambd, beta):
    return 1./2 * (1 + np.sin(beta)**2) * np.cos(2*lambd - np.pi/3)
def Facross(lambd, beta):
    return np.sin(beta) * np.sin(2*lambd - np.pi/3)
def Feplus(lambd, beta):
    return 1./2 * (1 + np.sin(beta)**2) * np.cos(2*lambd + np.pi/6)
def Fecross(lambd, beta):
    return np.sin(beta) * np.sin(2*lambd + np.pi/6)

# From simple_likelihood.py, modified to follow notations of the paper
# Format : pars numpy array in the form [d, phiL, inc, lambdaL, betaL, psiL]
# Here d is DL/DLinj, distance scale relative to the injection
# The angles lambdaL, betaL, psiL have a LISA-frame meaning here - and phiL is the quasi-orbital phase shift defined for the L-frame as well (that is, assuming tL is common to all waveforms, not tSSB)
# factor = 4 int((6 pi f L/c)^(2) |h22|^2) - to be given as input, and gives the SNR scale
# sa, se will be precomputed for the injection
def func_sa_22(params):
    [d, phi, inc, lambd, beta, psi] = params
    Faplusval = Faplus(lambd, beta)
    Facrossval = Facross(lambd, beta)
    a22 = 1./4/d * np.sqrt(5/np.pi) * np.cos(inc/2)**4 * np.exp(2.*1j*(phi-psi)) * (Faplusval + 1j*Facrossval)
    a2m2 = 1./4/d * np.sqrt(5/np.pi) * np.sin(inc/2)**4 * np.exp(2.*1j*(phi+psi)) * (Faplusval - 1j*Facrossval)
    return a22 + a2m2
def func_se_22(params):
    [d, phi, inc, lambd, beta, psi] = params
    Feplusval = Feplus(lambd, beta)
    Fecrossval = Fecross(lambd, beta)
    e22 = 1./4/d * np.sqrt(5/np.pi) * np.cos(inc/2)**4 * np.exp(2.*1j*(phi-psi)) * (Feplusval + 1j*Fecrossval)
    e2m2 = 1./4/d * np.sqrt(5/np.pi) * np.sin(inc/2)**4 * np.exp(2.*1j*(phi+psi)) * (Feplusval - 1j*Fecrossval)
    return e22 + e2m2
def simple_likelihood_22(pars, factor, sainj, seinj):
    return -1./2 * factor * (abs(func_sa_22(pars) - sainj)**2 + abs(func_se_22(pars) - seinj)**2)

# Mode pattern functions, Eq. (44) in arXiv:2003.00357
def Fae_lm(params, modes=None):
    if not params.get('Lframe', False):
        raise ValueError('Need parameters in the L-frame.')
    if modes is None:
        modes = [(2,2)]
    inc = params['inc']
    phi = params['phi']
    lambd = params['lambda']
    beta = params['beta']
    psi = params['psi']
    # A bit slow, 4us each (2us / trigo function ?)
    Faplusval = Faplus(lambd, beta)
    Facrossval = Facross(lambd, beta)
    Feplusval = Feplus(lambd, beta)
    Fecrossval = Fecross(lambd, beta)
    sYlm = np.array([pytools.sYlm(lm[0], lm[1], inc, phi) for lm in modes])
    sYlminusmstar = np.array([np.conj(pytools.sYlm(lm[0], -lm[1], inc, phi)) for lm in modes])
    # (-1)^l -- np.power here is surprisingly slow
    minus1l = np.array([1. - 2*(lm[0] % 2) for lm in modes])
    # Fa_lm = 1./2 * (sYlm * np.exp(-1j*2*psi) * (Faplusval + 1j*Facrossval) + minus1l * sYlminusmstar * np.exp(1j*2*psi) * (Faplusval - 1j*Facrossval))
    # Fe_lm = 1./2 * (sYlm * np.exp(-1j*2*psi) * (Feplusval + 1j*Fecrossval) + minus1l * sYlminusmstar * np.exp(1j*2*psi) * (Feplusval - 1j*Fecrossval))
    # Optimize a bit -- should maybe use a C extension
    e2ipsi = np.exp(1j*2*psi)
    em2ipsi = np.conj(e2ipsi)
    # These simple multiplications are not negligible (should be in C)
    fac_a_mpsi = (1./2 * em2ipsi * (Faplusval + 1j*Facrossval))
    fac_a_ppsi = (1./2 * e2ipsi * (Faplusval - 1j*Facrossval))
    fac_e_mpsi = (1./2 * em2ipsi * (Feplusval + 1j*Fecrossval))
    fac_e_ppsi = (1./2 * e2ipsi * (Feplusval - 1j*Fecrossval))
    Fa_lm = fac_a_mpsi * sYlm + fac_a_ppsi * minus1l * sYlminusmstar
    Fe_lm = fac_e_mpsi * sYlm + fac_e_ppsi * minus1l * sYlminusmstar
    return Fa_lm, Fe_lm

# Simple likelihood with higher modes, see Eqs. (67)-(68) in arXiv:2003.00357
# Params must be in the L-frame, and need injection distance
# No check that the intrinsic params and time match injection values
def sae_lm(params, dist_inj, modes=None):
    dist = params['dist']
    d = dist / dist_inj
    Fa_lm, Fe_lm = Fae_lm(params, modes=modes)
    sa_lm = 1./d * Fa_lm
    se_lm = 1./d * Fe_lm
    return sa_lm, se_lm

# Degenerate sky positions - see PE paper
# Because we take the ratio sigma_+/sigma_-, d and phi are silent
def func_degen_sky(inc, lambd, beta, psi):
    phi_dummy = 0.
    d_dummy = 1.
    params = [d_dummy, phi_dummy, inc, lambd, beta, psi]
    sa = func_sa_22(params)
    se = func_se_22(params)
    sigma_plus = sa + 1j*se
    sigma_minus = sa - 1j*se
    r = sigma_plus / sigma_minus
    lambdaL_star = pytools.mod2pi(np.pi/6. - 1./4 * np.angle(r))
    betaL_star = np.pi/2. - 2*np.arctan((np.abs(r))**(1./4))
    # Resolve the pi/2 degeneracy in lambdaL_star by picking the closest to lambdaL
    lambdaL_star = lambdaL_star + ((lambd - lambdaL_star) - pytools.modsym(lambd - lambdaL_star, np.pi/2))
    return (lambdaL_star, betaL_star)
# Degenerate parameters - see PE paper
# Looks for building degenerate point at 0 inclination and 0 phase
# Gives value of all [d, phi, inc, lambd, beta, psi]
# NOTE: phi=0 might not work with previous waveform conventions where it did not
# have its purely extrinsic meaning
def func_degen_params_0inc_0phase(d, phi, inc, lambd, beta, psi):
    params = [d, phi, inc, lambd, beta, psi]
    sa = func_sa_22(params)
    se = func_se_22(params)
    sigma_plus = 1./2 * (sa + 1j*se)
    sigma_minus = 1./2 * (sa - 1j*se)
    r = sigma_plus / sigma_minus
    # Inclination, phase -- by convention
    inc_star = 0. # we choose to look for face-on point
    phi_star = 0. # we choose to look for zero-phase (exact degen with psi)
    # Sky
    lambdaL_star = pytools.mod2pi(np.pi/6. - 1./4 * np.angle(r))
    betaL_star = np.pi/2. - 2*np.arctan((np.abs(r))**(1./4))
    # Resolve the pi/2 degeneracy in lambdaL_star by picking the closest to lambdaL
    lambdaL_star = lambdaL_star + ((lambd - lambdaL_star) - pytools.modsym(lambd - lambdaL_star, np.pi/2))
    # Distance
    thetaL_star = np.pi/2. - betaL_star
    d_star = 1./4*np.sqrt(5./np.pi) * 1./(1. + np.tan(thetaL_star/2.)**2)**2 / np.abs(sigma_minus)
    # Polarization, for generic phi_star although here we have phi_star = 0
    psiL_star = pytools.modpi(-1./2 * np.angle(sigma_minus) + lambdaL_star - np.pi/6 + phi_star) # defined mod pi
    return (d_star, phi_star, inc_star, lambdaL_star, betaL_star, psiL_star)
# Looks for building degenerate point at 0 inclination
# Gives value of all [d, phi, inc, lambd, beta, psi]
# NOTE: phi=0 might not work with previous waveform conventions where it did not
# have its purely extrinsic meaning
# So if phi_star is None (by default), keep the original value
def func_degen_params_0inc(d, phi, inc, lambd, beta, psi, phi_star=None):
    params = [d, phi, inc, lambd, beta, psi]
    sa = func_sa_22(params)
    se = func_se_22(params)
    sigma_plus = 1./2 * (sa + 1j*se)
    sigma_minus = 1./2 * (sa - 1j*se)
    r = sigma_plus / sigma_minus
    # Inclination, phase -- by convention
    inc_star = 0. # we choose to look for face-on point
    if phi_star is None:
        phi_star = phi # by default, just keep the same value of the phase
    # Sky
    lambdaL_star = pytools.mod2pi(np.pi/6. - 1./4 * np.angle(r))
    betaL_star = np.pi/2. - 2*np.arctan((np.abs(r))**(1./4))
    # Resolve the pi/2 degeneracy in lambdaL_star by picking the closest to lambdaL
    lambdaL_star = lambdaL_star + ((lambd - lambdaL_star) - pytools.modsym(lambd - lambdaL_star, np.pi/2))
    # Distance
    thetaL_star = np.pi/2. - betaL_star
    d_star = 1./4*np.sqrt(5./np.pi) * 1./(1. + np.tan(thetaL_star/2.)**2)**2 / np.abs(sigma_minus)
    # Polarization, for generic phi_star although here we have phi_star = 0
    psiL_star = pytools.modpi(-1./2 * np.angle(sigma_minus) + lambdaL_star - np.pi/6 + phi_star) # defined mod pi
    return (d_star, phi_star, inc_star, lambdaL_star, betaL_star, psiL_star)

################################################################################
# Function to compute likelihood for the 8 sky modes
################################################################################

# From a likelihood class, compute the loglikelihood for the 8 sky modes
# Sky indices: (a,b) a=+-1 for a*betaL, b=0-3 for lambdaL + b*pi/2
def func_loglikelihood_skymodes(likelihoodClass):

    params = likelihoodClass.source_params_Lframe
    inc = params['inc']
    lambdaL = params['lambda']
    betaL = params['beta']
    psiL = params['psi']

    skymodes = [(1,0), (-1,0), (1,1), (-1,1), (1,2), (-1,2), (1,3), (-1,3)]

    lnL_skymodes = {}
    for skymode in skymodes:
        params_skymode = params.copy()
        skymode_index0, skymode_index1 = skymode
        params_skymode['inc'] = pytools.modpi(np.pi/2 - (skymode_index0 *(np.pi/2 - inc)))
        params_skymode['lambda'] = pytools.mod2pi(lambdaL + skymode_index1 * np.pi/2)
        params_skymode['beta'] = skymode_index0 * betaL
        params_skymode['psi'] = pytools.modpi(np.pi/2 - (skymode_index0 *(np.pi/2 - psiL)) + skymode_index1 * np.pi/2)
        lnL_skymodes[skymode] = likelihoodClass.lnL(params_skymode, **likelihoodClass.waveform_params)

    return lnL_skymodes
# Compute the loglikelihood for the 8 sky modes