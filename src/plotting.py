import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

def call_plot_histogram(bs, bins, figsize, save_fig, save_fig_path):

    match bs.bootstrap_selector:
        case 1 | 2 | 3:

            wavelength_str = "all_waves"

            for sampling_results, model_param_name in zip(
                    bs.sampling_results, bs.varied_param_ls
                ):

                plot_histogram(
                    data=sampling_results,
                    param_descr=model_param_name,
                    sample_descr=bs.sample_descr,
                    fit_func_descr=bs.fit_func_descr,
                    wavelength_str=wavelength_str,
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
                    wavelength_str=wavelength_descr, fig_format="pdf"
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
        save_fig_path: Path where to save the figure. The default is
          "../figures/".
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(x=data, bins=bins)

    # Obtain various strings for title, axes labels, and legend.
    short_param_str = get_short_param_str(param_descr)
    long_param_str = get_long_param_str(param_descr)
    param_unit_str = get_param_unit_str(param_descr)

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
        fig_format = "pdf"
        fig_name = get_hist_fig_name(
            param_descr, sample_descr, fit_func_descr,
            wavelength_str=wavelength_str, fig_format=fig_format
        )
        fig.savefig(save_fig_path+fig_name, format=fig_format)

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
    (data, data_error,
     _, spatial_frequency, _) = bs.full_data_set.get_all_data_flattened()

    if not plot_data_uncertainty:
        data_error = None

    # Compute the data of the model if it is already set up. If it is set
    # up, but the bootstrapping has not been performed, plot the model with
    # the initial values. If the bootstrapping has been performed, plot the
    # model with the best fit parameters.

    # Derive different data intervals for the measured data, the analytic
    # function, and the axis. Each interval border is 5% smaller/larger
    # than the former.
    spatial_frequency_min = spatial_frequency.min()
    spatial_frequency_max = spatial_frequency.max()
    func_min = 0.95 * spatial_frequency_min
    func_max = 1.05 * spatial_frequency_max

    spatial_frequency_func = np.geomspace(func_min, func_max, 100)

    # This is executed if the bootstrapping has been performed.
    try:
        fitted_param = {
            key: bs.results[key] for key in bs.varied_param_ls
        }
        func_data = bs.fit_func(
            spatial_frequency_func, **bs.fixed_param, **fitted_param
        )
        func_label = "result"

        # Create string for title with the varied parameter and their
        # fit results.
        title_varied_param_str = []
        for param in bs.varied_param_ls:

            title_varied_param_str.append(
                f"{get_short_param_str(param)} ="
                f" {fitted_param[param]:.2}"
                f" {get_param_unit_str(param)}"
            )
        title_varied_param_str = ", ".join(title_varied_param_str)

        # Create string for title with the fixed parameter and their
        # values.
        title_fixed_param_str = []
        for param in bs.fixed_param:
            title_fixed_param_str.append(
                f"{get_short_param_str(param)} ="
                f" {bs.fixed_param[param]}"
                f" {get_param_unit_str(param)}"
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
                spatial_frequency_func, *bs.param_init_value_ls
            )
            func_label = "initial guess"

            # Create string for title with the initial parameter values
            # before fitting.
            title_init_param_str = []
            for i_param, param in enumerate(bs.model.param_names):
                title_init_param_str.append(
                    f"{get_short_param_str(param)} = "
                    f"{bs.param_init_value_ls[i_param]} "
                    f"{get_param_unit_str(param)}"
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
        fig_format = "pdf"
        fig_name = get_vis_fig_name(
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr,
            fig_format=fig_format
        )
        fig.savefig(save_fig_path+fig_name, format=fig_format)

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
            fig_format="pdf"
    )

    # Make one pdf with each figure on one page.
    with PdfPages(save_fig_path+pdf_name) as pdf:

        for i_wave in range(bs.N_wavelength):

            ##### Precalculations.
            wavelength = bs.wavelength_ls[i_wave]
            wavelength_str = (
                f"{wavelength*1e6:.4f} micron"
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

            # Derive different data intervals for the measured data, the
            # analytic function, and the axis. Each interval border is 5%
            # smaller/larger than the former.
            spatial_frequency_min = spatial_frequency.min()
            spatial_frequency_max = spatial_frequency.max()
            func_min = 0.95 * spatial_frequency_min
            func_max = 1.05 * spatial_frequency_max

            spatial_frequency_func = np.geomspace(func_min, func_max, 100)

            # This is executed if the bootstrapping has been performed.
            try:
                fitted_param = {
                    param: bs.results[f"{wavelength_str}, {param}"] \
                    for param in bs.varied_param_ls
                }
                func_data = bs.fit_func(
                    spatial_frequency_func, **bs.fixed_param, **fitted_param
                )
                func_label = "result"

                # Create string for title with the varied parameter and their
                # fit results.
                title_varied_param_str = []
                for param in bs.varied_param_ls:

                    title_varied_param_str.append(
                        f"{get_short_param_str(param)} ="
                        f" {fitted_param[param]:.2}"
                        f" {get_param_unit_str(param)}"
                    )
                title_varied_param_str = ", ".join(title_varied_param_str)

                # Create string for title with the fixed parameter and their
                # values.
                title_fixed_param_str = []
                for param in bs.fixed_param:
                    title_fixed_param_str.append(
                        f"{get_short_param_str(param)} ="
                        f" {bs.fixed_param[param]}"
                        f" {get_param_unit_str(param)}"
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
                        spatial_frequency_func, *bs.param_init_value_ls
                    )
                    func_label = "initial guess"

                    # Create string for title with the initial parameter values
                    # before fitting.
                    title_init_param_str = []
                    for i_param, param in enumerate(bs.model.param_names):
                        title_init_param_str.append(
                            f"{get_short_param_str(param)} = "
                            f"{bs.param_init_value_ls[i_param]} "
                            f"{get_param_unit_str(param)}"
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
    elif param_descr == "FOV":
        short_param_str = "FOV"

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
    elif param_descr == "FOV":
        long_param_str = "field-of-view (FOV)"

    return long_param_str

def get_param_unit_str(param_descr: str) -> str:
    """
    Define the unit str for a given fit parameter.

    Args:
        param_descr: The string representation of the fit parameter.

    Returns:
        unit_str: The string to be used for the parameter unit.
    """

    if param_descr == "f":
        unit_str = ""
    elif param_descr == "stellar_diameter":
        unit_str = "mas"
    elif param_descr == "lin_limb_dark_param":
        unit_str = ""
    elif param_descr == "FOV":
        unit_str = "mas"

    return unit_str

def get_hist_xlabel(param_descr: str):
    """
    Define the x-axis label of the histograms for a given fit parameter.

    Args:
        param_descr: The string representation of the fit parameter.

    Returns:
        x_label: The string to be used as x-axis label.
    """

    param_unit_str = get_param_unit_str(param_descr)
    x_label = (
        f"{get_long_param_str(param_descr)}"
        + f"{" /" if param_unit_str!="" else ""}"
        + f"{param_unit_str}"
    )

    return x_label

def get_hist_fig_name(
        param_descr: str, sample_descr: str, fit_func_descr: str,
        wavelength_str: str, fig_format: str
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
        fig_format: The file format.

    Returns:
        fig_name: The figure file name.
    """

    fig_name = (
        f"{"_".join([param_descr, "hist", sample_descr, fit_func_descr])}"
        f"{f"_{wavelength_str}" if wavelength_str!="" else ""}.{fig_format}"
    )

    return fig_name

def get_vis_fig_name(
        fit_vis_or_vis2: str, sample_descr: str,
        fit_func_descr: str, wavelength_descr: str, fig_format: str
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
        fig_format: The file format.

    Returns:
        fig_name: The figure file name.
    """

    fig_name = (
        f"{"_".join([fit_vis_or_vis2, sample_descr, fit_func_descr,
                     wavelength_descr])}"
        f".{fig_format}"
    )

    return fig_name