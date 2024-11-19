import astropy.units as u
import numpy as np
import scipy

from data_handling import comp_spatial_frequency

def comp_VIS2_limbDarkDisk_overresolved(
    u_spatial_frequency: "Scalar or array (float)",
    v_spatial_frequency: "Scalar or array (float)",
    f_cse: float,
    stellar_diameter: float,
    lin_limb_dark_param: float,
) -> "Scalar or array (float)":
    """
    Compute the squared vis. for limb dark. disk and overresolved component.

    Compute the result as the square of the visibility amplitude. See for more
    information and Args the function comp_VISAMP_limbDarkDisk_overresolved.

    Returns:
      VIS2: The computed squared visibility as scalar or array.
    """

    VIS2 = comp_VISAMP_limbDarkDisk_overresolved(
        u_spatial_frequency, v_spatial_frequency, f_cse, stellar_diameter,
        lin_limb_dark_param
    )**2

    return VIS2

def comp_VISAMP_limbDarkDisk_overresolved(
    u_spatial_frequency: "Scalar or array (float)",
    v_spatial_frequency: "Scalar or array (float)",
    f_cse: float,
    stellar_diameter: float,
    lin_limb_dark_param: float
) -> "Scalar or array (float)":
    """
    Compute vis. amplitude for limb dark. disk and overresolved component.

    Reference: Di Folco et al. 2007, eq. 4.

    Args:
        u_spatial_frequency: Spatial frequency along u axis in units of 1/rad.
        v_spatial_frequency: Spatial frequency along v axis in units of 1/rad.
        f_cse: The ratio of the flux of the overresolved component and the
          total flux (star + overresolved component). CSE is short for
          circumstellar environment.
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter. Set 0.0
          for a uniform disk without limb-darkening.

    Returns:
      VISAMP: The computed visibility amplitude as scalar or array.
    """

    VISAMP = (
        (1.0-f_cse) * comp_VISAMP_limbDarkDisk(
            u_spatial_frequency,
            v_spatial_frequency,
            stellar_diameter,
            lin_limb_dark_param
            )
    )

    return VISAMP

def comp_VIS2_limbDarkDisk_gauss(
    u_spatial_frequency: "Scalar or array (float)",
    v_spatial_frequency: "Scalar or array (float)",
    f_cse: float,
    stellar_diameter: float,
    lin_limb_dark_param: float,
    FWHM: float
) -> "Scalar or array (float)":
    """
    Compute the squared vis. for limb darkened disk and surrounding Gaussian.

    Compute the result as the square of the visibility amplitude. See for more
    information and Args the function comp_VISAMP_limbDarkDisk_gauss.

    Returns:
        VIS2: The computed squared visibility as scalar or array.
    """

    VIS2 = comp_VISAMP_limbDarkDisk_gauss(
        u_spatial_frequency, v_spatial_frequency, f_cse, stellar_diameter,
        lin_limb_dark_param, FWHM
    )**2

    return VIS2

def comp_VISAMP_limbDarkDisk_gauss(
    u_spatial_frequency: "Scalar or array (float)",
    v_spatial_frequency: "Scalar or array (float)",
    f_cse: float,
    stellar_diameter: float,
    lin_limb_dark_param: float,
    FWHM: float
) -> "Scalar or array (float)":
    """
    Compute (squared) visibility for limb darkened disk and environment.

    Reference: Di Folco et al. 2007, eq. 3.
    Both the star and the Gaussian are located in the center of the
    field-of-view.

    Args:
        u_spatial_frequency: Spatial frequency along u axis in units of 1/rad.
        v_spatial_frequency: Spatial frequency along v axis in units of 1/rad.
        f_cse: The ratio of the flux of the Gaussian and the total flux (star +
          Gaussian). CSE is short for circumstellar environment.
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter. Set 0.0
          for a uniform disk without limb-darkening.
        FWHM: The FWHM of the circular Gaussian in units of mas.

    Returns:
      VISAMP: The computed visibility amplitude as scalar or array.
    """

    VISAMP = (
        f_cse * comp_VISAMP_circGauss(u_spatial_frequency,
                                  v_spatial_frequency,
                                  FWHM)
        + (1.0-f_cse) * comp_VISAMP_limbDarkDisk(u_spatial_frequency,
                                             v_spatial_frequency,
                                             stellar_diameter,
                                             lin_limb_dark_param)
    )

    return VISAMP

def comp_VIS2_limbDarkDisk_gauss_ptSrc(
    u_spatial_frequency: "Scalar or array (float)",
    v_spatial_frequency: "Scalar or array (float)",
    stellar_diameter: float,
    lin_limb_dark_param: float,
    f_cse: float,
    FWHM: float,
    f_ptsrc: float,
    alpha_ptsrc: float,
    beta_ptsrc: float
) -> "Scalar or array (float)":
    """
    Compute the squ. vis. for for limb. dark star, Gaussian, and point source.

    Compute the result as the square of the visibility amplitude. See for more
    information and Args the function def comp_VISAMP_limbDarkDisk_gauss_ptSrc.

    Returns:
        VIS2: The computed squared visibility as scalar or array.
    """

    VIS2 = comp_VISAMP_limbDarkDisk_gauss_ptSrc(
        u_spatial_frequency,
        v_spatial_frequency,
        stellar_diameter,
        lin_limb_dark_param,
        f_cse,
        FWHM,
        f_ptsrc,
        alpha_ptsrc,
        beta_ptsrc
    )**2

    return VIS2


def comp_VISAMP_limbDarkDisk_gauss_ptSrc(
    u_spatial_frequency: "Scalar or array (float)",
    v_spatial_frequency: "Scalar or array (float)",
    stellar_diameter: float,
    lin_limb_dark_param: float,
    f_cse: float,
    FWHM: float,
    f_ptsrc: float,
    alpha_ptsrc: float,
    beta_ptsrc: float
) -> "Scalar or array (float)":

    """
    Compute vis. amplitude for limb. dark star, Gaussian, and point source.

    The flux of the star is set to 1.
    The computation is based on Berger & Segansan 2007, eq. 28. That is the
    general form for the visibility of a multicomponent model. The different
    components here are the visibility of a limb darkened star, a Gaussian, and
    a point source. Note that the visibility of a point source is 1 for all
    spatial frequencies. From this eq. 28 which gives a complex quantity the
    absolute has been computed to get the visibility amplitude.

    Instead of the direction independent spatial frequency B/wavelength, the
    spatial frequencies along the u and v axes, i.e., u/wavelength and
    v/wavelength, have to be used. This is because the point source can be
    located away from the center of the field-of-view.

    Args:
        u_spatial_frequency: Spatial frequency along u axis in units of 1/rad.
        v_spatial_frequency: Spatial frequency along v axis in units of 1/rad.
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter. Set 0.0
          for a uniform disk without limb-darkening.
        f_cse: The ratio of the flux of the Gaussian and the total flux
          (star + Gaussian + point source). CSE is short for circumstellar
          environment.
        FWHM: The FWHM of the circular Gaussian in units of mas.
        f_ptsrc: The ratio of the flux of the point source and the total flux
          (star + Gaussian + point source).
        alpha_ptsrc: The position of the point source on-sky along the u axis
          (North-South) in units of mas.
        beta_ptsrc: The position of the point source on-sky along the v axis
          (West-East) in units of mas.

    Returns:
        VISAMP: The visibility amplitude.
    """

    # Convert angular coordinates of the point source to rad.
    alpha_ptsrc = alpha_ptsrc * u.mas.to(u.rad)
    beta_ptsrc = beta_ptsrc * u.mas.to(u.rad)

    f_star = 1.0
    f_tot = f_star + f_cse + f_ptsrc

    V_star = comp_VISAMP_limbDarkDisk(
        u_spatial_frequency,
        v_spatial_frequency,
        stellar_diameter,
        lin_limb_dark_param
    )

    V_gauss = comp_VISAMP_circGauss(
        u_spatial_frequency,
        v_spatial_frequency,
        FWHM
    )

    V_ptsrc = 1.0

    A = 2.0 * np.pi * (u_spatial_frequency * alpha_ptsrc
                       + v_spatial_frequency * beta_ptsrc)

    B = (
        (f_star*V_star)**2 + (f_cse*V_gauss)**2 + (f_ptsrc*V_ptsrc)**2
        + 2*f_star*V_star*f_cse*V_gauss
        + 2*f_star*V_star*f_ptsrc*V_ptsrc*np.cos(A)
        + 2*f_cse*V_gauss*f_ptsrc*V_ptsrc*np.cos(A)
    )**0.5

    VISAMP = B / f_tot

    return VISAMP

def comp_VISAMP_limbDarkDisk(
    u_spatial_frequency: "Scalar or array (float)",
    v_spatial_frequency: "Scalar or array (float)",
    stellar_diameter: float,
    lin_limb_dark_param: float = 0.0
) -> "Scalar or array (float)":
    """
    Compute vis. amplitude for disk with or without linear limb-darkening.

    In case the linear limb-darkening parameter is chosen to be zero
    (default case) the equation simplifies to the visibility amplitude for
    a uniform disk.
    Reference: Hanbury Brown et al. 1974, eqs. 2 and 4. Implemented form of
        the equation is from Kirchschlager et al. 2020, eq. 2.

    Args:
        u_spatial_frequency: Spatial frequency along u axis in units of 1/rad.
        v_spatial_frequency: Spatial frequency along v axis in units of 1/rad.
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter.

    Returns:
        VISAMP: The visibility amplitude.
    """

    # In case of zero diameter, the disk is a point source which has a
    # visibility of 1.0 for all spatial frequencies.
    if stellar_diameter == 0.0:
        VISAMP = np.ones(u_spatial_frequency.shape)
        return VISAMP

    spatial_frequency = comp_spatial_frequency(
        u_spatial_frequency, v_spatial_frequency
    )

    spatial_frequency *= u.rad
    stellar_diameter *= u.mas

    x = (np.pi * stellar_diameter.to(u.rad) * spatial_frequency).value

    f = 6.0 / (3.0 - lin_limb_dark_param)
    a = (1.0-lin_limb_dark_param) * scipy.special.jv(1, x) / x

    # The case selection is not strictly necessary as b = 0 for a zero
    # limb darkened parameter, but this way the computation of the costly
    # Bessel function is avoided.
    match lin_limb_dark_param:
        case 0.0:
            b = 0.0
        case _:
            b = (lin_limb_dark_param * np.sqrt(np.pi/2.0)
                    * scipy.special.jv(1.5, x) / x**1.5)

    VISAMP = np.abs(f * (a+b))

    return VISAMP

def comp_VISAMP_circGauss(
    u_spatial_frequency: "Scalar or array (float)",
    v_spatial_frequency: "Scalar or array (float)",
    FWHM: float
) -> "Scalar or array (float)":
    """
    Compute visibility amplitude for Gaussian from FWHM.

    Args:
        u_spatial_frequency: Spatial frequency along u axis in units of 1/rad.
        v_spatial_frequency: Spatial frequency along v axis in units of 1/rad.
        FWHM: The FWHM of a circular Gaussian in units of mas.

    Returns:
        VISAMP: The visibility amplitude.
    """

    spatial_frequency = comp_spatial_frequency(
        u_spatial_frequency, v_spatial_frequency
    )

    spatial_frequency *= u.rad
    FWHM *= u.mas

    x = (np.pi * FWHM.to(u.rad) * spatial_frequency).value

    VISAMP = np.exp(-x**2 / (4*np.log(2)))

    return VISAMP