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
        f"{short_param_str} = {"(" if param_unit_str!="" else ""}{median:.2}"
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
             set_title):

    match bs.bootstrap_selector:
        case 1 | 2 | 3:
            plot_vis_all_wavelengths(bs, plot_data_uncertainty, figsize,
                                     save_fig, save_fig_path, set_title)
        case 4:
            plot_vis_for_fixed_wavelengths(bs, plot_data_uncertainty, figsize,
                                           save_fig, save_fig_path, set_title)

def plot_vis_all_wavelengths(
        bs,
        plot_data_uncertainty,
        figsize,
        save_fig,
        save_fig_path,
        set_title
):

    wavelength_descr = "all_waves"

    # Get data for each individual baseline.
    data_ls = []
    data_error_ls = []
    spatial_frequency_data_ls = []
    baseline_id_ls = []

    baseline_ls = bs.full_data_set.get_all_baselines()

    # Sort the lists after baselines with increasing baseline length and hence
    # increasing spatial frequencies. This is to set the later order along
    # which the data is plotted. This ensures that baselines with similar
    # spatial frequencies and hence closely located data points in the plot
    # get colors that are easy to distinguish from each other.
    # The lambda function as key ensures that only the entries of B_ls are
    # used for sorting. This is necessary to use two times the same Oifits,
    # e.g., to select different wavelength intervals or give it more weight in
    # the analysis. In this case the same baseline length appears twice in B_ls
    # and the sorted function goes on to sort the Baseline objects in
    # baseline_ls, which cannot be sorted.
    B_ls = [baseline.B for baseline in baseline_ls]
    B_tuple, baseline_tuple = zip(
        *sorted(zip(B_ls, baseline_ls), key=lambda x: x[0])
    )
    baseline_ls = list(baseline_tuple)

    for baseline in baseline_ls:
        data_ls.append(baseline.data)
        spatial_frequency_data_ls.append(
            data_handling.comp_spatial_frequency(
                baseline.u_spatial_frequency, baseline.v_spatial_frequency
            )
        )
        if plot_data_uncertainty:
            data_error_ls.append(baseline.data_error)
        else:
            data_error_ls.append(None)
        baseline_id_ls.append(baseline.baseline_id)

    spatial_frequency_data_min = np.concatenate(spatial_frequency_data_ls).min()
    spatial_frequency_data_max = np.concatenate(spatial_frequency_data_ls).max()

    # Derive spatial frequencies for the analytic function to produce data
    # for plotting. This is only needed if a model is there.
    # A distinction is made between
    # 1) polar symmetric models for which the differentiation between telescope
    #    pairs and hence u/v spatial frequencies does not matter
    # 2) non polar symmetric models.
    # For this purpose the spatial frequencies used to plot the analytic
    # function are put in a list. In case 1) this list has only one item, an
    # array of all spatial frequencies. In case 2) this list has one item for
    # each telescope pair (i.e., baseline).
    if hasattr(bs, "results") or hasattr(bs, "param_init_value"):

        alpha = 0.7 #  transparent data to see better the analytic solution

        if bs.model_is_polar_symmetric:

            # Use the u coordinate to store the baseline spatial frequencies
            # and set the v coordinate to zero. By doing this the following is
            # possible: Extend the spatial frequencies by 5% into each
            # direction for nicer plots.
            # Automatically, the spatial frequencies appear in increasing
            # order.
            u_spatial_frequency_func_ls = [
                np.linspace(spatial_frequency_data_min * 0.95,
                            spatial_frequency_data_max * 1.05,
                            1000
                )
            ]
            v_spatial_frequency_func_ls = [
                np.zeros(len(u_spatial_frequency_func_ls[0]))
            ]
            func_color = "black"

        elif not bs.model_is_polar_symmetric:

            u_spatial_frequency_func_ls = []
            v_spatial_frequency_func_ls = []
            for baseline in baseline_ls:

                u_spatial_frequency = baseline.u_spatial_frequency
                v_spatial_frequency = baseline.v_spatial_frequency
                # Increase the sampling for smoother plots. Do not use min/max,
                # as for negative spatial frequencies only the absolute
                # increases.
                u_spatial_frequency_func = np.linspace(
                    u_spatial_frequency[0],
                    u_spatial_frequency[-1],
                    100
                )
                v_spatial_frequency_func = np.linspace(
                    v_spatial_frequency[0],
                    v_spatial_frequency[-1],
                    100
                )

                u_spatial_frequency_func_ls.append(u_spatial_frequency_func)
                v_spatial_frequency_func_ls.append(v_spatial_frequency_func)

                func_color = None

    # Compute the data of the model if it is already set up. If it is set
    # up, but the bootstrapping has not been performed, plot the model with
    # the initial values. If the bootstrapping has been performed, plot the
    # model with the best fit parameters.

    # This is executed if the bootstrapping has been performed.
    if hasattr(bs, "results"):

        # Make dict containing only the varied params and its result to feed
        # the fit function.
        fitted_param = {
            param: bs.results[param] for param in bs.varied_param_ls
        }

        data_func_ls = []
        label_ls = []
        alpha = 0.7 #  transparent data to see better the analytic solution
        data_label_ls = [None for i in range(len(baseline_id_ls))]

        for (u_spatial_frequency_func,
             v_spatial_frequency_func,
             baseline_id) in zip(
            u_spatial_frequency_func_ls, v_spatial_frequency_func_ls,
            baseline_id_ls
        ):
            # Obtain data for the best fit model.
            data_func_ls.append(bs.fit_func(
                u_spatial_frequency_func, v_spatial_frequency_func,
                **bs.fixed_param, **fitted_param
                )
            )
            if baseline_id is None:
                label_ls.append("result")
            else:
                label_ls.append(f"result: {baseline_id}")

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
        title_varied_param_str = "\n    ".join(title_varied_param_str)

        # Create string for title with the fixed parameter and their
        # values.
        title_fixed_param_str = []
        for param in bs.fixed_param:
            title_fixed_param_str.append(
                f"{get_short_param_str(param)} ="
                f" {bs.fixed_param[param]}"
                f" {get_var_unit_str(param)}"
            )
        title_fixed_param_str = "\n    ".join(title_fixed_param_str)

        title = (
            f"Fitted parameters:\n    {title_varied_param_str}\n"
            f"Fixed parameters:\n    {title_fixed_param_str}"
        )

    # This is executed if the bootstrapping has not been performed, but the
    # model has been set up.
    elif hasattr(bs, "param_init_value"):

        data_func_ls = []
        label_ls = []
        alpha = 0.7 #  transparent data to see better the analytic solution
        data_label_ls = [None for i in range(len(baseline_id_ls))]

        for (u_spatial_frequency_func,
             v_spatial_frequency_func,
             baseline_id) in zip(
            u_spatial_frequency_func_ls, v_spatial_frequency_func_ls,
            baseline_id_ls
        ):

            # Obtain data for the initial model.
            data_func_ls.append(bs.fit_func(
                u_spatial_frequency_func, v_spatial_frequency_func,
                **bs.param_init_value
                )
            )
            if baseline_id is None:
                label_ls.append("initial guess")
            else:
                label_ls.append(f"initial guess: {baseline_id}")

        # Create string for title with the initial parameter values
        # before fitting.
        title_init_param_str = []
        for param in bs.model.param_names:
            title_init_param_str.append(
                f"{get_short_param_str(param)} = "
                f"{bs.param_init_value[param]} "
                f"{get_var_unit_str(param)}"
            )
        title_init_param_str = "\n    ".join(title_init_param_str)

        title = f"Initial parameters:\n    {title_init_param_str}\n"

    # This is executed if the model has not been set up.
    else:

        data_func_ls = None
        alpha = 1.0 #  non transparent data
        data_label_ls = baseline_id_ls
        title = ""

    ##### Actual plotting.
    fig, ax = plt.subplots(figsize=figsize)

    for spatial_frequency_data, data, data_error, data_label in zip(
        spatial_frequency_data_ls, data_ls, data_error_ls, data_label_ls
    ):

        ax.errorbar(
            spatial_frequency_data, data, yerr=data_error, fmt="x",
            markersize=10, markeredgewidth=2, alpha=alpha, label=data_label
        )

    # Reset color cycle before plotting the analytical solutions to match the
    # colors to the previous plots of data.
    ax.set_prop_cycle(None)

    # Plot analytical function if available.
    if np.any(data_func_ls):

        for (u_spatial_frequency_func,
            v_spatial_frequency_func,
            data_func,
            label) in zip(
            u_spatial_frequency_func_ls,
            v_spatial_frequency_func_ls,
            data_func_ls,
            label_ls
        ):

            spatial_frequency_func = data_handling.comp_spatial_frequency(
                u_spatial_frequency_func, v_spatial_frequency_func
            )
            ax.plot(
                spatial_frequency_func, data_func, label=label, linewidth=2.5,
                color=func_color
            )

    fontsize_L = 20
    fontsize_S = 16

    ax.legend(fontsize=fontsize_S)

    ax.set_xlabel("spatial frequency /rad\u207B\u00b9", fontsize=fontsize_L)
    ax.set_ylabel(
        f"{"squared " if bs.fit_vis_or_vis2=="VIS2" else ""}visibility",
        fontsize=fontsize_L
    )

    ax.tick_params(axis='x', labelsize=fontsize_L)
    ax.tick_params(axis='y', labelsize=fontsize_L)

    # Set fontsize factor with the power of 10 from the scientific notation.
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(fontsize_L)

    if set_title:
        ax.set_title(title, loc="left", fontsize=fontsize_L)

    if save_fig:
        file_format = "pdf"
        fig_name = get_vis_fig_name(
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr,
            file_format=file_format
        )
        fig.savefig(save_fig_path+fig_name, format=file_format,
                    bbox_inches='tight')

def plot_vis_for_fixed_wavelengths(
        bs,
        plot_data_uncertainty,
        figsize,
        save_fig,
        save_fig_path,
        set_title
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

            u_spatial_frequency_data = (
                bs.data_per_wavelength[i_wave].u_spatial_frequency
            )
            v_spatial_frequency_data = (
                bs.data_per_wavelength[i_wave].v_spatial_frequency
            )
            spatial_frequency_data = (
                bs.data_per_wavelength[i_wave].spatial_frequency
            )

            data = bs.data_per_wavelength[i_wave].data
            data_error = bs.data_per_wavelength[i_wave].data_error
            if not plot_data_uncertainty:
                data_error = None

            # Derive spatial frequencies for the analytic function to produce
            # data for plotting. This is only needed if a model is there.
            # A distinction is made between
            # 1) polar symmetric models for which the differentiation between
            #    telescope pairs and hence u/v spatial frequencies does not
            #    matter.
            # 2) non polar symmetric models.
            # In case 1) one can plot one analytic curve for all baselines. In
            # this case increase the sampling of the spatial frequencies for a
            # smooth and continuous plot.
            # In case 2) this is not possible as every baseline has a different
            # analytical curve because the model has two independent variables
            # the spatial frequency in both u and v direction).
            # In this case 2) plot the fit results only as points.
            if hasattr(bs, "results") or hasattr(bs, "param_init_value"):

                if bs.model_is_polar_symmetric:

                    # Use the u coordinate to store the baseline spatial
                    # frequencies and set the v coordinate to zero.
                    # Extend the spatial frequencies by 5% into each direction
                    # for nicer plots.
                    # Automatically, the spatial frequencies appear in
                    # increasing order.
                    u_spatial_frequency_func = (
                        np.linspace(spatial_frequency_data.min() * 0.95,
                                    spatial_frequency_data.max() * 1.05,
                                    1000
                        )
                    )
                    v_spatial_frequency_func = (
                        np.zeros(len(u_spatial_frequency_func))
                    )
                    spatial_frequency_func = u_spatial_frequency_func

                elif not bs.model_is_polar_symmetric:

                    pass

            # Compute the data of the model if it is already set up. If it is
            # set up, but the bootstrapping has not been performed, plot the
            # model with the initial values. If the bootstrapping has been
            # performed, plot the model with the best fit parameters.

            # This is executed if the bootstrapping has been performed.
            if hasattr(bs, "results"):

                # Make dict containing only the varied params and its result to
                # feed the fit function.
                fitted_param = {
                    param: bs.results[param][i_wave] \
                    for param in bs.varied_param_ls
                }
                # Obtain data for the best fit model.
                if bs.model_is_polar_symmetric:
                    data_func = bs.fit_func(
                        u_spatial_frequency_func, v_spatial_frequency_func,
                        **bs.fixed_param, **fitted_param
                    )

                elif not bs.model_is_polar_symmetric:
                    data_func = bs.fit_func(
                        u_spatial_frequency_data, v_spatial_frequency_data,
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

            # This is executed if the bootstrapping has not been performed, but
            # the model has been set up.
            elif hasattr(bs, "param_init_value"):

                if bs.model_is_polar_symmetric:
                    data_func = bs.fit_func(
                        u_spatial_frequency_func, v_spatial_frequency_func,
                        **bs.param_init_value
                )

                elif not bs.model_is_polar_symmetric:

                    data_func = bs.fit_func(
                        u_spatial_frequency_data, v_spatial_frequency_data,
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
            else:

                data_func = None
                title = f"Wavelength = {wavelength_str}"

            ##### Actual plotting.
            fig, ax = plt.subplots(figsize=figsize)

            ax.errorbar(spatial_frequency_data, data, yerr=data_error, fmt="x")

            if bs.model_is_polar_symmetric:

                if np.any(data_func):

                    ax.plot(spatial_frequency_func, data_func, label=func_label)
                    ax.legend()

            elif not bs.model_is_polar_symmetric:

                if np.any(data_func):

                    ax.scatter(
                        spatial_frequency_data,
                        data_func,
                        marker="X",
                        s=106, #  markersize
                        color="red",
                        label=func_label)
                    ax.legend()

            ax.set_xlabel("spatial frequency /rad\u207B\u00b9")
            ax.set_ylabel(
                f"{"squared " if bs.fit_vis_or_vis2=="VIS2" else ""}visibility"
            )

            if set_title:
                ax.set_title(title, loc="left")

            pdf.savefig(fig)

def plot_relative_dust_sed(
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

    # Parameters listed alphabetically. If parameter is not listed, the short
    # representation defaults to the parameter name.
    if param_descr == "diameter_UD":
        short_param_str = "diameter UD"
    elif param_descr == "lin_limb_dark_param":
        short_param_str = "a"
    elif param_descr == "stellar_diameter":
        short_param_str = "stellar diameter"
    else:
        short_param_str = param_descr

    return short_param_str

def get_long_param_str(param_descr: str) -> str:
    """
    Define a long str for visualization for a given fit parameter.

    Args:
        param_descr: The string representation of the fit parameter.

    Returns:
        long_param_str: The string to be used for the parameter.
    """

    # Parameters listed alphabetically.
    if param_descr == "alpha_ptsrc":
        long_param_str = "alpha point source"
    elif param_descr == "alpha_UD":
        long_param_str = "alpha uniform disk"
    elif param_descr == "beta_ptsrc":
        long_param_str = "beta point source"
    elif param_descr == "beta_UD":
        long_param_str = "beta uniform disk"
    elif param_descr == "diameter_UD":
        long_param_str = "diameter uniform disk"
    elif param_descr == "f_cse":
        long_param_str = "f circumstellar environment"
    elif param_descr == "f_ptsrc":
        long_param_str = "f point source"
    elif param_descr == "f_ring":
        long_param_str = "f ring"
    elif param_descr == "f_UD":
        long_param_str = "f uniform disk"
    elif param_descr == "FWHM":
        long_param_str = "Gaussian FWHM"
    elif param_descr == "lin_limb_dark_param":
        long_param_str = "linear limb-darkened parameter a"
    elif param_descr == "R_in":
        long_param_str = "Inner ring radius"
    elif param_descr == "stellar_diameter":
        long_param_str = "stellar diameter"
    elif param_descr == "width_scaling":
        long_param_str = "width scaling"
    else:
        raise ValueError(
            f"Parameter {param_descr} has no corresponding long string. Please"
            " implement here."
        )

    return long_param_str

def get_var_unit_str(var_descr: str) -> str:
    """
    Return unit str for a given variable like wavelength or a fit parameter.

    Args:
        var_descr: The string representation of the fit parameter.

    Returns:
        unit_str: The string to be used for the parameter unit.
    """

    # Parameter names in lists sorted alphabetically.
    if var_descr in [
        "alpha_ptsrc", "alpha_UD", "beta_ptsrc", "beta_UD", "diameter", "FWHM",
        "R_in", "stellar_diameter"
    ]:
        unit_str = "mas"

    elif var_descr in ["wavelength"]:
        unit_str = "m"

    elif var_descr in ["lin_limb_dark_param"]:
        unit_str = ""

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