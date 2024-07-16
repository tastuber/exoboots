import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(
        data: np.array,
        param_descriptor: str,
        sample_descriptor: str,
        fit_function_descriptor: str,
        wavelength_str: str = "",
        bins: int = 20,
        save_fig: bool = True,
        save_fig_path: str = "../figures/"
):
    """
    Plot the bootstrap histogram for a given parameter.

    Args:
        data: Numpy array of the various best-fit parameter values retrieved
          through bootstrapping.
        param_descriptor: Descriptor of the fitted parameter.
        wavelength_str: String representation of the wavelength. Use is
          intended for the case of fitting to each wavelength separately. The
          default is an empty string.
        bins: Number of histogram bins. The default is bins = 20.
        save_fig: Decides whether the figure is saved. True saves the figure,
          False does not.
        save_fig_path: Path where to save the figure. The default is
          "../figures/".
    """

    fig, ax = plt.subplots()
    ax.hist(x=data, bins=bins)

    # Obtain various strings for title, axes labels, and legend.
    short_param_str = get_short_param_str(param_descriptor)
    long_param_str = get_long_param_str(param_descriptor)
    param_unit_str = get_param_unit_str(param_descriptor)

    # Compute the bootstrap statistics: the median, the 0.16 quantile, and the
    # 0.84 quantile. The median is the best-fit parameter, the difference from
    # the median and the quantiles are the uncertainties.
    median = np.median(data)
    quantiles = (np.quantile(data, q=0.16), np.quantile(data, q=0.84))
    minus_uncertainty = median - quantiles[0]
    plus_uncertainty = quantiles[1] - median

    # Indicate the median and the quantiles as vertical lines. Labels are
    # defined anew for each vertical line.
    color_median = "tab:red"
    color_quant = "tab:orange"
    linewidth = 3
    ylim = ax.get_ylim()

    label=(
        f"0.16 quantile: {short_param_str} = {quantiles[0]:.2} "
        + f"{param_unit_str}"
    )
    ax.vlines(
        quantiles[0], ylim[0], ylim[1], color=color_quant, linestyle="--",
        linewidth=linewidth, label=label
    )

    label = (
        f"Median: {short_param_str} = {median:.2} "
        + f"{param_unit_str}"
    )
    ax.vlines(
        median, ylim[0], ylim[1], color=color_median, linestyle="--",
        linewidth=linewidth, label=label
    )

    label=(
        f"0.84 quantile: {short_param_str} = {quantiles[1]:.2} "
        + f"{param_unit_str}"
    )
    ax.vlines(
        quantiles[1], ylim[0], ylim[1], color=color_quant, linestyle="--",
        linewidth=linewidth, label=label
    )
    ax.set_ylim(ylim)

    ax.legend()

    # Print fit results as the title.
    title = (
        f"{wavelength_str}{": " if wavelength_str!="" else ""}"
        f"{long_param_str} = {"(" if param_unit_str!="" else ""}{median:.2}"
        + f" + {plus_uncertainty:.1} - {minus_uncertainty:.1}"
        + f"{")" if param_unit_str!="" else ""}{param_unit_str}"
    )
    ax.set_title(title)

    # Set axes labels.
    ax.set_xlabel(get_hist_xlabel(param_descriptor))
    ax.set_ylabel("counts")

    if save_fig:
        fig_format = "pdf"
        fig_name = get_hist_fig_name(
            param_descriptor, sample_descriptor, fit_function_descriptor,
            wavelength_str=wavelength_str, fig_format=fig_format
        )
        fig.savefig(save_fig_path+fig_name, format=fig_format)

def get_short_param_str(param_descriptor: str) -> str:
    """
    Define a short str for visualization for a given fit parameter.

    Args:
        param_descriptor: The string representation of the fit parameter.

    Returns:
        short_param_str: The string to be used for the parameter.
    """

    if param_descriptor == "f":
        short_param_str = "f"
    elif param_descriptor == "stellar_diameter":
        short_param_str = "stellar diameter"
    elif param_descriptor == "lin_limb_dark_param":
        short_param_str = "a"
    elif param_descriptor == "FOV":
        short_param_str = "FOV"

    return short_param_str

def get_long_param_str(param_descriptor: str) -> str:
    """
    Define a long str for visualization for a given fit parameter.

    Args:
        param_descriptor: The string representation of the fit parameter.

    Returns:
        long_param_str: The string to be used for the parameter.
    """

    if param_descriptor == "f":
        long_param_str = param_descriptor
    elif param_descriptor == "stellar_diameter":
        long_param_str = "stellar diameter"
    elif param_descriptor == "lin_limb_dark_param":
        long_param_str = "linear limb-darkened parameter a"
    elif param_descriptor == "FOV":
        long_param_str = "field-of-view (FOV)"

    return long_param_str

def get_param_unit_str(param_descriptor: str) -> str:
    """
    Define the unit str for a given fit parameter.

    Args:
        param_descriptor: The string representation of the fit parameter.

    Returns:
        unit_str: The string to be used for the parameter unit.
    """

    if param_descriptor == "f":
        unit_str = ""
    elif param_descriptor == "stellar_diameter":
        unit_str = "mas"
    elif param_descriptor == "lin_limb_dark_param":
        unit_str = ""
    elif param_descriptor == "FOV":
        unit_str = "mas"

    return unit_str

def get_hist_xlabel(param_descriptor: str):
    """
    Define the x-axis label of the histograms for a given fit parameter.

    Args:
        param_descriptor: The string representation of the fit parameter.

    Returns:
        x_label: The string to be used as x-axis label.
    """

    param_unit_str = get_param_unit_str(param_descriptor)
    x_label = (
        f"{get_long_param_str(param_descriptor)}"
        + f"{" /" if param_unit_str!="" else ""}"
        + f"{param_unit_str}"
    )

    return x_label

def get_hist_fig_name(
        param_descriptor: str, sample_descriptor: str,
        fit_function_descriptor: str, wavelength_str: str, fig_format: str
):
    """
    Returns the figure file name for given fit parameter and settings.

    Args:
        param_descriptor: Descriptor of the fittet paramter.
        sample_descriptor: Descriptor of the chosen sampling during
          bootstrapping, such as data points, baselines, observations or
          data points per wavelength.
        fit_function_descriptor: Descriptor of the chosen fit function.
        wavelength_str: String representation of the wavelength. Use is
          intended for the case of fitting to each wavelength separately.
        fig_format: The file format.

    Returns:
        fig_name: The figure file name.
    """

    fig_name = (
        f"{"_".join([param_descriptor, "hist", sample_descriptor,
                     fit_function_descriptor])}"
        f"{f"_{wavelength_str}" if wavelength_str!="" else ""}.{fig_format}"
    )

    return fig_name
