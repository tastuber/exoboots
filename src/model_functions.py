import astropy.units as u
import numpy as np
import scipy

def comp_VIS2_limb_dark_disk_plus_overresolved(
    spatial_frequency: "Scalar or array (float)",
    f: float,
    stellar_diameter: float,
    lin_limb_dark_param: float,
) -> "Scalar or array (float)":
    """
    Compute the squared vis. for limb dark. disk and overresolved component.

    Compute the result as the square of the visibility amplitude.

    Args:
        spatial_frequency: The spatial frequency in units of 1/rad.
        f: The ratio of the flux of the overresolved component and the total
          flux (overresolved component plus star).
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter. Set 0.0
          for a uniform disk without limb-darkening.

    Returns:
      VIS2: The computed squared visibility as scalar or array.
    """

    VIS2 = comp_VISAMP_limb_dark_disk_plus_overresolved(
        spatial_frequency, f, stellar_diameter, lin_limb_dark_param
    )**2

    return VIS2

def comp_VISAMP_limb_dark_disk_plus_overresolved(
    spatial_frequency: "Scalar or array (float)",
    f: float,
    stellar_diameter: float,
    lin_limb_dark_param: float
) -> "Scalar or array (float)":
    """
    Compute vis. amplitude for limb dark. disk and overresolved component.

    Reference: Di Folco et al. 2007, eq. 4.

    Args:
        spatial_frequency: The spatial frequency in units of 1/rad.
        f: The ratio of the flux of the overresolved component and the total
          flux (overresolved component plus star).
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter. Set 0.0
          for a uniform disk without limb-darkening.

    Returns:
      VISAMP: The computed visibility amplitude as scalar or array.
    """

    VISAMP = (
        (1.0-f) * comp_VISAMP_limb_dark_disk(spatial_frequency,
                                             stellar_diameter,
                                             lin_limb_dark_param)
    )

    return VISAMP

def comp_VIS2_limb_dark_disk_plus_uniform_CSE(
    spatial_frequency: "Scalar or array (float)",
    f: float,
    stellar_diameter: float,
    lin_limb_dark_param: float,
    FOV: float
) -> "Scalar or array (float)":
    """
    Compute the squared vis. for limb darkened disk and environment.

    Compute the result as the square of the visibility amplitude.

    Args:
        spatial_frequency: The spatial frequency in units of 1/rad.
        f: The ratio of the flux of the uniform circumstellar environment (CSE)
          and the total flux (CSE plus star).
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter. Set 0.0
          for a uniform disk without limb-darkening.
        FOV: The field-of-view of the instrument in units of mas.

    Returns:
      VIS2: The computed squared visibility as scalar or array.
    """

    VIS2 = comp_VISAMP_limb_dark_disk_plus_uniform_CSE(
        spatial_frequency, f, stellar_diameter, lin_limb_dark_param, FOV
    )**2

    return VIS2

def comp_VISAMP_limb_dark_disk_plus_uniform_CSE(
    spatial_frequency: "Scalar or array (float)",
    f: float,
    stellar_diameter: float,
    lin_limb_dark_param: float,
    FOV: float
) -> "Scalar or array (float)":
    """
    Compute (squared) visibility for limb darkened disk and environment.

    Reference: Di Folco et al. 2007, eq. 3.

    Args:
        spatial_frequency: The spatial frequency in units of 1/rad.
        f: The ratio of the flux of the uniform circumstellar environment (CSE)
          and the total flux (CSE plus star).
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter. Set 0.0
          for a uniform disk without limb-darkening.
        FOV: The field-of-view of the instrument in units of mas.

    Returns:
      VISAMP: The computed visibility amplitude as scalar or array.
    """

    VISAMP = (
        f * comp_VISAMP_uniform_gauss_sens(spatial_frequency, FOV)
        + (1.0-f) * comp_VISAMP_limb_dark_disk(spatial_frequency,
                                               stellar_diameter,
                                               lin_limb_dark_param)
    )

    return VISAMP

def comp_VISAMP_limb_dark_disk(
    spatial_frequency: "Scalar or array (float)",
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
        spatial_frequency: The spatial frequency in units of 1/rad.
        stellar_diameter: The stellar diameter in units of mas.
        lin_limb_dark_parameter: The linear limb-darkened parameter.

    Returns:
        VISAMP: The visibility amplitude.
    """

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

def comp_VISAMP_uniform_gauss_sens(
    spatial_frequency: "Scalar or array (float)",
    FOV: float
) -> "Scalar or array (float)":
    """
    Compute vis. ampltfor Gaussian with FWHM=field-of-view.

    Args:
        spatial_frequency: The spatial frequency in units of 1/rad.
        FOV: The field-of-view of the instrument in units of mas.

    Returns:
        VISAMP: The visibility amplitude.
    """

    spatial_frequency *= u.rad
    FOV *= u.mas

    x = (np.pi * FOV.to(u.rad) * spatial_frequency).value

    VISAMP = np.exp(-x**2 / (4*np.log(2)))

    return VISAMP