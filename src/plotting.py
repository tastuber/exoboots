import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import data_handling

def call_plot_histogram(bs, bins, figsize, save_fig, save_fig_path):

    match bs.bootstrap_selector:
        case 1 | 2 | 3:

            wavelength_descr = "all_waves"

            for sampling_results, model_param_name in zip(
                    bs.sampling_results, bs.varied_param_ls
                ):

                plot_histogram(
                    data=sampling_results,
                    param_descr=model_param_name,
                    sample_descr=bs.sample_descr,
                    fit_func_descr=bs.fit_func_descr,
                    wavelength_str=wavelength_descr,
                    bins=bins,
                    figsize=figsize,
                    save_fig=save_fig,
                    save_fig_path=save_fig_path
                )

        case 4:

            wavelength_descr = "for_single_waves"

            for i_varied_param, model_param_name in enumerate(bs.varied_param_ls):

                sampling_results_per_param = (
                    bs.sampling_results[:, i_varied_param, :]
                )

                pdf_name = get_hist_fig_name(
                    model_param_name, bs.sample_descr, bs.fit_func_descr,
                    wavelength_str=wavelength_descr, file_format="pdf"
                )


                fig_ls = [] #  Create list of figures for saving
                for i_wave, sampling_results in enumerate(sampling_results_per_param):

                    wavelength = bs.wavelength_ls[i_wave]
                    wavelength_str = (
                        f"{wavelength*1e6:.4f} micron"
                    )

                    fig = plot_histogram(
                        data=sampling_results,
                        param_descr=model_param_name,
                        sample_descr=bs.sample_descr,
                        fit_func_descr=bs.fit_func_descr,
                        wavelength_str=wavelength_str,
                        bins=bins,
                        figsize=figsize,
                        save_fig=False,
                        save_fig_path=save_fig_path
                    )
                    fig_ls.append(fig)

                if save_fig:

                    # Make one pdf with each figure on one page.
                    with PdfPages(save_fig_path+pdf_name) as pdf:

                        for fig in fig_ls:

                            pdf.savefig(fig)

def plot_histogram(
        data: np.array,
        param_descr: str,
        sample_descr: str,
        fit_func_descr: str,
        wavelength_str: str,
        bins: int,
        figsize: tuple[float, float],
        save_fig: bool,
        save_fig_path: str
):
    """
    Plot the bootstrap histogram for a given parameter.

    Args:
        data: Numpy array of the various best-fit parameter values retrieved
          through bootstrapping.
        param_descr: Descriptor of the fitted parameter.
        wavelength_str: String representation of the wavelength. Use is
          intended for the case of fitting to each wavelength separately. The
          default is an empty string.
        bins: Number of histogram bins. The default is bins = 20.
        save_fig: Decides whether the figure is saved. True saves the figure,
          False does not.
        save_fig_path: Path where to save the figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(x=data, bins=bins)

    # Obtain various strings for title, axes labels, and legend.
    short_param_str = get_short_param_str(param_descr)
    long_param_str = get_long_param_str(param_descr)
    param_unit_str = get_var_unit_str(param_descr)

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
    ax.set_xlabel(get_hist_xlabel(param_descr))
    ax.set_ylabel("counts")

    if save_fig:
        file_format = "pdf"
        fig_name = get_hist_fig_name(
            param_descr, sample_descr, fit_func_descr,
            wavelength_str=wavelength_str, file_format=file_format
        )
        fig.savefig(save_fig_path+fig_name, format=file_format)

    return fig

def plot_vis(bs, plot_data_uncertainty, figsize, save_fig, save_fig_path,
             title):

    match bs.bootstrap_selector:
        case 1 | 2 | 3:
            plot_vis_all_wavelengths(bs, plot_data_uncertainty, figsize,
                                     save_fig, save_fig_path, title)
        case 4:
            plot_vis_for_fixed_wavelengths(bs, plot_data_uncertainty, figsize,
                                           save_fig, save_fig_path, title)

def plot_vis_all_wavelengths(
        bs,
        plot_data_uncertainty,
        figsize,
        save_fig,
        save_fig_path,
        title
):

    wavelength_descr = "all_waves"

    ##### Precalculations.
    (data,
     data_error,
     _,
     u_spatial_frequency,
     v_spatial_frequency,
     _) = bs.full_data_set.get_all_data_flattened()

    spatial_frequency = data_handling.comp_spatial_frequency(
        u_spatial_frequency, v_spatial_frequency
    )

    if not plot_data_uncertainty:
        data_error = None

    # Compute the data of the model if it is already set up. If it is set
    # up, but the bootstrapping has not been performed, plot the model with
    # the initial values. If the bootstrapping has been performed, plot the
    # model with the best fit parameters.

    # Derive input values for the analytic function to produce data
    # for plotting.
    u_spatial_frequency_func = np.linspace(
        u_spatial_frequency.min(), u_spatial_frequency.max(), 100
    )
    v_spatial_frequency_func = np.linspace(
        v_spatial_frequency.min(), v_spatial_frequency.max(), 100
    )

    spatial_frequency_func = data_handling.comp_spatial_frequency(
        u_spatial_frequency_func, v_spatial_frequency_func
    )

    # This is executed if the bootstrapping has been performed.
    try:
        # Make dict containing only the varied params and its result to feed
        # the fit function.
        fitted_param = {
            param: bs.results[param] for param in bs.varied_param_ls
        }
        # Obtain data for the best fit model.
        func_data = bs.fit_func(
            u_spatial_frequency_func, v_spatial_frequency_func,
            **bs.fixed_param, **fitted_param
        )
        func_label = "result"

        # Create string for title with the varied parameter and their
        # fit results.
        title_varied_param_str = []
        for param in bs.varied_param_ls:

            title_varied_param_str.append(
                f"{get_short_param_str(param)} = ("
                f"{bs.results[param]:.2}"
                f" + {bs.results[f"+Delta {param}"]:.2}"
                f" - {bs.results[f"-Delta {param}"]:.2}"
                f") {get_var_unit_str(param)}".rstrip()
            )
        title_varied_param_str = ", ".join(title_varied_param_str)

        # Create string for title with the fixed parameter and their
        # values.
        title_fixed_param_str = []
        for param in bs.fixed_param:
            title_fixed_param_str.append(
                f"{get_short_param_str(param)} ="
                f" {bs.fixed_param[param]}"
                f" {get_var_unit_str(param)}"
            )
        title_fixed_param_str = ", ".join(title_fixed_param_str)

        title = (
            f"Fitted parameters: {title_varied_param_str}\n"
            f"Fixed parameters: {title_fixed_param_str}"
        )

    # This is executed if the bootstrapping has not been performed.
    except AttributeError:

        # This is executed if the model has been set up.
        try:
            func_data = bs.fit_func(
                u_spatial_frequency_func, v_spatial_frequency_func,
                **bs.param_init_value
            )
            func_label = "initial guess"

            # Create string for title with the initial parameter values
            # before fitting.
            title_init_param_str = []
            for param in bs.model.param_names:
                title_init_param_str.append(
                    f"{get_short_param_str(param)} = "
                    f"{bs.param_init_value[param]} "
                    f"{get_var_unit_str(param)}"
                )
            title_init_param_str = ", ".join(title_init_param_str)

            title = f"Initial parameters: {title_init_param_str}\n"

        # This is executed if the model has not been set up.
        except AttributeError:
            func_data = None
            func_label = None

    ##### Actual plotting.
    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(spatial_frequency, data, yerr=data_error, fmt="x")

    if np.any(func_data):
        ax.plot(spatial_frequency_func, func_data, label=func_label)
        ax.legend()

    ax.set_xlabel("spatial frequency")
    ax.set_ylabel(
        f"{"squared " if bs.fit_vis_or_vis2=="VIS2" else ""}visibility"
    )

    ax.set_title(title, loc="left")

    if save_fig:
        file_format = "pdf"
        fig_name = get_vis_fig_name(
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr,
            file_format=file_format
        )
        fig.savefig(save_fig_path+fig_name, format=file_format)

def plot_vis_for_fixed_wavelengths(
        bs,
        plot_data_uncertainty,
        figsize,
        save_fig,
        save_fig_path,
        title
):

    wavelength_descr = "for_single_waves"
    pdf_name = get_vis_fig_name(
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr,
            file_format="pdf"
    )

    # Make one pdf with each figure on one page.
    with PdfPages(save_fig_path+pdf_name) as pdf:

        for i_wave in range(bs.N_wavelength):

            ##### Precalculations.
            wavelength = bs.wavelength_ls[i_wave]
            wavelength_str = (
                f"{wavelength*1e6:.4f} micron"
            )

            u_spatial_frequency = (
                bs.data_per_wavelength[i_wave].u_spatial_frequency
            )
            v_spatial_frequency = (
                bs.data_per_wavelength[i_wave].v_spatial_frequency
            )
            spatial_frequency = (
                bs.data_per_wavelength[i_wave].spatial_frequency
            )

            data = bs.data_per_wavelength[i_wave].data
            data_error = bs.data_per_wavelength[i_wave].data_error
            if not plot_data_uncertainty:
                data_error = None

            # Compute the data of the model if it is already set up. If it is
            # set up, but the bootstrapping has not been performed, plot the
            # model with the initial values. If the bootstrapping has been
            # performed, plot the model with the best fit parameters.

            # Derive input values for the analytic function to produce data
            # for plotting.
            u_spatial_frequency_func = np.linspace(
                u_spatial_frequency.min(), u_spatial_frequency.max(), 100
            )
            v_spatial_frequency_func = np.linspace(
                v_spatial_frequency.min(), v_spatial_frequency.max(), 100
            )

            spatial_frequency_func = data_handling.comp_spatial_frequency(
                u_spatial_frequency_func, v_spatial_frequency_func
            )

            # This is executed if the bootstrapping has been performed.
            try:
                # Make dict containing only the varied params and its result to
                # feed the fit function.
                fitted_param = {
                    param: bs.results[param][i_wave] \
                    for param in bs.varied_param_ls
                }
                # Obtain data for the best fit model.
                func_data = bs.fit_func(
                    u_spatial_frequency_func, v_spatial_frequency_func,
                    **bs.fixed_param, **fitted_param
                )
                func_label = "result"

                # Create string for title with the varied parameter and their
                # fit results.
                title_varied_param_str = []
                for param in bs.varied_param_ls:

                    title_varied_param_str.append(
                        f"{get_short_param_str(param)} = ("
                        f"{bs.results[param][i_wave]:.2}"
                        f" + {bs.results[f"+Delta {param}"][i_wave]:.2}"
                        f" - {bs.results[f"-Delta {param}"][i_wave]:.2}"
                        f") {get_var_unit_str(param)}".rstrip()
                    )
                title_varied_param_str = ", ".join(title_varied_param_str)

                # Create string for title with the fixed parameter and their
                # values.
                title_fixed_param_str = []
                for param in bs.fixed_param:
                    title_fixed_param_str.append(
                        f"{get_short_param_str(param)} ="
                        f" {bs.fixed_param[param]}"
                        f" {get_var_unit_str(param)}"
                    )
                title_fixed_param_str = ", ".join(title_fixed_param_str)

                title = (
                    f"Wavelength = {wavelength_str}\n"
                    f"Fitted parameters: {title_varied_param_str}\n"
                    f"Fixed parameters: {title_fixed_param_str}"
                )

            # This is executed if the bootstrapping has not been performed.
            except AttributeError:

                # This is executed if the model has been set up.
                try:
                    func_data = bs.fit_func(
                        u_spatial_frequency_func, v_spatial_frequency_func,
                        **bs.param_init_value
                    )
                    func_label = "initial guess"

                    # Create string for title with the initial parameter values
                    # before fitting.
                    title_init_param_str = []
                    for param in bs.model.param_names:
                        title_init_param_str.append(
                            f"{get_short_param_str(param)} = "
                            f"{bs.param_init_value[param]} "
                            f"{get_var_unit_str(param)}"
                        )
                    title_init_param_str = ", ".join(title_init_param_str)

                    title = (
                        f"Wavelength = {wavelength_str}\n"
                        f"Initial parameters: {title_init_param_str}\n"
                    )

                # This is executed if the model has not been set up.
                except AttributeError:
                    func_data = None
                    func_label = None
                    title = f"Wavelength = {wavelength_str}"

            ##### Actual plotting.
            fig, ax = plt.subplots(figsize=figsize)

            ax.errorbar(spatial_frequency, data, yerr=data_error, fmt="x")

            if np.any(func_data):
                ax.plot(spatial_frequency_func, func_data, label=func_label)
                ax.legend()

            ax.set_xlabel("spatial frequency")
            ax.set_ylabel(
                f"{"squared " if bs.fit_vis_or_vis2=="VIS2" else ""}visibility"
            )

            ax.set_title(title, loc="left")

            pdf.savefig(fig)

def plot_relative_sed(
        bs,
        plot_data_uncertainty,
        figsize,
        save_fig,
        save_fig_path,
        wavelength_descr,
        title = "Relative SED"
):

    fig, ax = plt.subplots(figsize=figsize)

    if plot_data_uncertainty:

        # Specific format for plt.errorbar to work with one and multiple data
        # points at the same time.
        if type(bs.relative_sed["dust to star flux ratio"]) == np.float64:
            yerr = [
                [bs.relative_sed["+Delta dust to star flux ratio"]],
                [bs.relative_sed["-Delta dust to star flux ratio"]]
            ]
        else:
            yerr = (bs.relative_sed["+Delta dust to star flux ratio"],
                    bs.relative_sed["-Delta dust to star flux ratio"])

        ax.errorbar(
            x=bs.relative_sed["wavelength /m"],
            y=bs.relative_sed["dust to star flux ratio"],
            yerr=yerr,
            marker="x"
        )

    else:

        # Plot single data point. Otherwise in the special case on only one
        # data point and not to be plotted errorbars, nothing is displayed.
        if type(bs.relative_sed["wavelength /m"]) == np.float64:
            marker = "x"
        else:
            marker = None

        ax.plot(
             bs.relative_sed["wavelength /m"],
             bs.relative_sed["dust to star flux ratio"],
             marker=marker
        )

    ax.set_title(title, loc="right")
    ax.set_xlabel("wavelength /m")
    ax.set_ylabel("dust to star flux ratio")

    if save_fig:
        file_format = "pdf"
        fig_name = get_sed_file_name(
            sed_descr="relative_SED",
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr,
            file_format=file_format
        )
        fig.savefig(save_fig_path+fig_name, format=file_format)

def plot_dust_sed(
        bs,
        plot_data_uncertainty,
        figsize,
        save_fig,
        save_fig_path,
        wavelength_descr,
        title = "Dust SED"
):

    fig, ax = plt.subplots(figsize=figsize)

    if plot_data_uncertainty:

        # Specific format for plt.errorbar to work with one and multiple data
        # points at the same time.
        if type(bs.sed["dust flux /Jy"]) == np.float64:
            yerr = [
                [bs.sed["+Delta dust flux /Jy"]],
                [bs.sed["-Delta dust flux /Jy"]]
            ]
        else:
            yerr = (bs.sed["+Delta dust flux /Jy"],
                    bs.sed["-Delta dust flux /Jy"])

        ax.errorbar(
            x=bs.sed["wavelength /m"],
            y=bs.sed["dust flux /Jy"],
            yerr=yerr,
            marker="x"
        )

    else:

        # Plot single data point. Otherwise in the special case on only one
        # data point and not to be plotted errorbars, nothing is displayed.
        if type(bs.sed["wavelength /m"]) == np.float64:
            marker = "x"
        else:
            marker = None

        ax.plot(
            bs.sed["wavelength /m"], bs.sed["dust flux /Jy"], marker=marker
        )

    ax.set_title(title, loc="right")
    ax.set_xlabel("wavelength /m")
    ax.set_ylabel("dust flux /Jy")

    if save_fig:
        file_format = "pdf"
        fig_name = get_sed_file_name(
            sed_descr="SED",
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr,
            file_format=file_format
        )
        fig.savefig(save_fig_path+fig_name, format=file_format)

def get_short_param_str(param_descr: str) -> str:
    """
    Define a short str for visualization for a given fit parameter.

    Args:
        param_descr: The string representation of the fit parameter.

    Returns:
        short_param_str: The string to be used for the parameter.
    """

    if param_descr == "f":
        short_param_str = "f"
    elif param_descr == "stellar_diameter":
        short_param_str = "stellar diameter"
    elif param_descr == "lin_limb_dark_param":
        short_param_str = "a"
    elif param_descr == "FWHM":
        short_param_str = "FWHM"

    return short_param_str

def get_long_param_str(param_descr: str) -> str:
    """
    Define a long str for visualization for a given fit parameter.

    Args:
        param_descr: The string representation of the fit parameter.

    Returns:
        long_param_str: The string to be used for the parameter.
    """

    if param_descr == "f":
        long_param_str = param_descr
    elif param_descr == "stellar_diameter":
        long_param_str = "stellar diameter"
    elif param_descr == "lin_limb_dark_param":
        long_param_str = "linear limb-darkened parameter a"
    elif param_descr == "FWHM":
        long_param_str = "Gaussian FWHM"

    return long_param_str

def get_var_unit_str(var_descr: str) -> str:
    """
    Return unit str for a given variable like wavelength or a fit parameter.

    Args:
        var_descr: The string representation of the fit parameter.

    Returns:
        unit_str: The string to be used for the parameter unit.
    """

    if var_descr == "wavelength":
        unit_str = "m"
    elif var_descr == "f":
        unit_str = ""
    elif var_descr == "stellar_diameter":
        unit_str = "mas"
    elif var_descr == "lin_limb_dark_param":
        unit_str = ""
    elif var_descr == "FWHM":
        unit_str = "mas"
    else:
        unit_str = ""

    return unit_str

def get_hist_xlabel(param_descr: str):
    """
    Define the x-axis label of the histograms for a given fit parameter.

    Args:
        param_descr: The string representation of the fit parameter.

    Returns:
        x_label: The string to be used as x-axis label.
    """

    param_unit_str = get_var_unit_str(param_descr)
    x_label = (
        f"{get_long_param_str(param_descr)}"
        + f"{" /" if param_unit_str!="" else ""}"
        + f"{param_unit_str}"
    )

    return x_label

def get_hist_fig_name(
        param_descr: str, sample_descr: str, fit_func_descr: str,
        wavelength_str: str, file_format: str
):
    """
    Returns the figure file name for given fit parameter and settings.

    Args:
        param_descr: Descriptor of the fittet paramter.
        sample_descr: Descriptor of the chosen sampling during
          bootstrapping, such as data points, baselines, observations or
          data points per wavelength.
        fit_func_descr: Descriptor of the chosen fit function.
        wavelength_str: String representation of the wavelength. Use is
          intended for the case of fitting to each wavelength separately.
        file_format: The file format.

    Returns:
        fig_name: The figure file name.
    """

    fig_name = (
        f"{"_".join([param_descr, "hist", sample_descr, fit_func_descr])}"
        f"{f"_{wavelength_str}" if wavelength_str!="" else ""}.{file_format}"
    )

    return fig_name

def get_vis_fig_name(
        fit_vis_or_vis2: str, sample_descr: str,
        fit_func_descr: str, wavelength_descr: str, file_format: str
):
    """
    Returns the figure file name for given fit parameter and settings.

    Args:
        sample_descr: Descriptor of the chosen sampling during
          bootstrapping, such as data points, baselines, observations or
          data points per wavelength.
        fit_func_descr: Descriptor of the chosen fit function.
        wavelength_descr: Contains information about the wavelengths.
          Typically either all wavelengths are fitted together or separately.
        file_format: The file format.

    Returns:
        fig_name: The figure file name.
    """

    # Remove the "VISAMP_" or "VIS2_" from fit_func_desc as it is present in
    # fit_vis_or_vis2 anyway.
    fit_func_descr = "_".join(fit_func_descr.split("_")[1:])

    fig_name = (
        f"{"_".join([fit_vis_or_vis2, sample_descr, fit_func_descr,
                     wavelength_descr])}"
        f".{file_format}"
    )

    return fig_name

def get_sed_file_name(
        sed_descr: str, fit_vis_or_vis2: str, sample_descr: str,
        fit_func_descr: str, wavelength_descr: str, file_format: str
):
    """
    Returns the file name for a SED for given fit parameter and settings.

    Args:
        sed_descr: Descriptor of the table.
        sample_descr: Descriptor of the chosen sampling during
          bootstrapping, such as data points, baselines, observations or
          data points per wavelength.
        fit_func_descr: Descriptor of the chosen fit function.
        wavelength_descr: Contains information about the wavelengths.
          Typically either all wavelengths are fitted together or separately.
        file_format: The file format.

    Returns:
        sed_name: The file name.
    """

    # Remove the "VISAMP_" or "VIS2_" from fit_func_desc as it is present in
    # fit_vis_or_vis2 anyway.
    fit_func_descr = "_".join(fit_func_descr.split("_")[1:])

    sed_name = (
        f"{"_".join([sed_descr, fit_vis_or_vis2, sample_descr,
                     fit_func_descr, wavelength_descr])}"
        f".{file_format}"
    )

    return sed_name