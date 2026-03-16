import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

from exoboots import data_handling

def call_plot_histogram(bs, bins, figsize, save, save_path):

    match bs.bootstrap_selector:
        case 1 | 2 | 3:

            wavelength_descr = "all_waves"

            for sampling_results, model_param_name in zip(
                    bs.sampling_results, bs.varied_params
                ):

                plot_histogram(
                    data=sampling_results,
                    param_descr=model_param_name,
                    sample_descr=bs.sample_descr,
                    fit_func_descr=bs.fit_func_descr,
                    wavelength_str=wavelength_descr,
                    bins=bins,
                    figsize=figsize,
                    save=save,
                    save_path=save_path
                )

        case 4:

            wavelength_descr = "for_single_waves"

            for i_varied_param, model_param_name in enumerate(bs.varied_params):

                sampling_results_per_param = (
                    bs.sampling_results[:, i_varied_param, :]
                )

                pdf_name = get_hist_fig_name(
                    model_param_name, bs.sample_descr, bs.fit_func_descr,
                    wavelength_str=wavelength_descr, file_format="pdf"
                )


                figs = [] #  Create list of figures for saving
                for i_wave, sampling_results in enumerate(sampling_results_per_param):

                    wavelength = bs.wavelengths[i_wave]
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
                        save=False,
                        save_path=save_path
                    )
                    figs.append(fig)

                if save:

                    # Make one pdf with each figure on one page.
                    with PdfPages(save_path+pdf_name) as pdf:

                        for fig in figs:

                            pdf.savefig(fig)

def plot_histogram(
        data: np.array,
        param_descr: str,
        sample_descr: str,
        fit_func_descr: str,
        wavelength_str: str,
        bins: int,
        figsize: tuple[float, float],
        save: bool,
        save_path: str
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
        save: Decides whether the figure is saved. True saves the figure,
          False does not.
        save_path: Path where to save the figure.
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

    if save:
        file_format = "pdf"
        fig_name = get_hist_fig_name(
            param_descr, sample_descr, fit_func_descr,
            wavelength_str=wavelength_str, file_format=file_format
        )
        fig.savefig(save_path+fig_name, format=file_format)

    return fig

def plot_vis(bs, plot_data_uncertainty, figsize, save, save_path,
             set_title, show_baseline_legend):

    match bs.bootstrap_selector:
        case 1 | 2 | 3:
            plot_vis_all_wavelengths(bs, plot_data_uncertainty, figsize,
                                     save, save_path, set_title,
                                     show_baseline_legend)
        case 4:
            plot_vis_for_fixed_wavelengths(bs, plot_data_uncertainty, figsize,
                                           save, save_path, set_title,
                                           show_baseline_legend)

def plot_vis_all_wavelengths(
        bs,
        plot_data_uncertainty,
        figsize,
        save,
        save_path,
        set_title,
        show_baseline_legend
):

    wavelength_descr = "all_waves"

    # Get data for each individual baseline.
    data_per_baseline = []
    data_error_per_baseline = []
    spfrq_data_per_baseline = []
    baseline_ids = []

    baselines = bs.full_data_set.get_all_baselines()

    # Sort the lists after baselines with increasing baseline length (B) and
    # hence increasing spatial frequencies. This is to set the later order
    # along which the data is plotted. This ensures that baselines with similar
    # spatial frequencies and hence closely located data points in the plot
    # get colors that are easy to distinguish from each other.
    # The lambda function as key ensures that only the entries of Bs are
    # used for sorting. This is necessary to use two times the same Oifits,
    # e.g., to select different wavelength intervals or give it more weight in
    # the analysis. In this case the same baseline length appears twice in Bs
    # and the sorted function goes on to sort the Baseline objects in
    # baselines, which cannot be sorted.
    Bs = [baseline.B for baseline in baselines]
    _, tmp_baselines = zip(
        *sorted(zip(Bs, baselines), key=lambda x: x[0])
    )
    baselines = list(tmp_baselines)

    for baseline in baselines:
        data_per_baseline.append(baseline.data)
        spfrq_data_per_baseline.append(
            data_handling.comp_spfrq(
                baseline.u_spfrq, baseline.v_spfrq
            )
        )
        if plot_data_uncertainty:
            data_error_per_baseline.append(baseline.data_error)
        else:
            data_error_per_baseline.append(None)
        baseline_ids.append(baseline.baseline_id)

    min_spfrq_data = np.concatenate(spfrq_data_per_baseline).min()
    max_spfrq_data = np.concatenate(spfrq_data_per_baseline).max()

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
            u_spfrq_per_baseline_func = [
                np.linspace(min_spfrq_data * 0.95,
                            max_spfrq_data * 1.05,
                            1000
                )
            ]
            v_spfrq_per_baseline_func = [
                np.zeros(len(u_spfrq_per_baseline_func[0]))
            ]
            func_color = "black"
            data_labels = baseline_ids

        elif not bs.model_is_polar_symmetric:

            u_spfrq_per_baseline_func = []
            v_spfrq_per_baseline_func = []
            for baseline in baselines:

                u_spfrq = baseline.u_spfrq
                v_spfrq = baseline.v_spfrq
                # Increase the sampling for smoother plots. Do not use min/max,
                # as for negative spatial frequencies only the absolute
                # increases.
                u_spfrq_func = np.linspace(
                    u_spfrq[0],
                    u_spfrq[-1],
                    100
                )
                v_spfrq_func = np.linspace(
                    v_spfrq[0],
                    v_spfrq[-1],
                    100
                )

                u_spfrq_per_baseline_func.append(u_spfrq_func)
                v_spfrq_per_baseline_func.append(v_spfrq_func)

                func_color = None
                data_labels = [None for i in range(len(baseline_ids))]

    # Compute the data of the model if it is already set up. If it is set
    # up, but the bootstrapping has not been performed, plot the model with
    # the initial values. If the bootstrapping has been performed, plot the
    # model with the best fit parameters.

    # This is executed if the bootstrapping has been performed.
    if hasattr(bs, "results"):

        # Make dict containing only the varied params and its result to feed
        # the fit function.
        fitted_param = {
            param: bs.results[param] for param in bs.varied_params
        }

        data_per_baseline_func = []
        labels = []
        alpha = 0.7 #  transparent data to see better the analytic solution

        for (u_spfrq_func,
             v_spfrq_func,
             baseline_id) in zip(
            u_spfrq_per_baseline_func, v_spfrq_per_baseline_func,
            baseline_ids
        ):
            # Obtain data for the best fit model.
            data_per_baseline_func.append(bs.fit_func(
                u_spfrq_func, v_spfrq_func,
                **bs.fixed_param, **fitted_param
                )
            )
            if bs.model_is_polar_symmetric:
                labels.append("result")
            else:
                labels.append(f"result: {baseline_id}")

        # Create string for title with the varied parameter and their
        # fit results.
        title_varied_param_str = []
        for param in bs.varied_params:

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

        chi2 = bs.results["chi2"]
        red_chi2 = bs.results["red_chi2"]
        ndof = bs.results["ndof"]
        title_chi2_str = (
            f"weighted chi2 = {chi2:.2}, red. chi2 = {red_chi2:.2}, "
            f"ndof = {ndof}"
        )

        title = (
            f"Fitted parameters:\n    {title_varied_param_str}\n"
            f"Fixed parameters:\n    {title_fixed_param_str}\n"
            f"{title_chi2_str}"
        )

        fig_name = get_vis_fig_name(
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr
        )

    # This is executed if the bootstrapping has not been performed, but the
    # model has been set up.
    elif hasattr(bs, "param_init_value"):

        data_per_baseline_func = []
        labels = []
        alpha = 0.7 #  transparent data to see better the analytic solution

        for (u_spfrq_func,
             v_spfrq_func,
             baseline_id) in zip(
            u_spfrq_per_baseline_func, v_spfrq_per_baseline_func,
            baseline_ids
        ):

            # Obtain data for the initial model.
            data_per_baseline_func.append(bs.fit_func(
                u_spfrq_func, v_spfrq_func,
                **bs.param_init_value
                )
            )
            if bs.model_is_polar_symmetric:
                labels.append("initial guess")
            else:
                labels.append(f"initial guess: {baseline_id}")

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

        fig_name = get_vis_fig_name(
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr
        )

    # This is executed if the model has not been set up.
    else:

        data_per_baseline_func = None
        alpha = 1.0 #  non transparent data
        data_labels = baseline_ids #  label the data for any model
        title = ""
        fig_name = get_vis_fig_name(
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=None,
            fit_func_descr=None,
            wavelength_descr=wavelength_descr
        )

    ##### Actual plotting.
    fig, ax = plt.subplots(figsize=figsize)

    for spfrq_data, data, data_error, data_label in zip(
        spfrq_data_per_baseline, data_per_baseline, data_error_per_baseline, data_labels
    ):

        ax.errorbar(
            spfrq_data, data, yerr=data_error, fmt="x",
            markersize=10, markeredgewidth=2, alpha=alpha, label=data_label
        )

    # Reset color cycle before plotting the analytical solutions to match the
    # colors to the previous plots of data.
    ax.set_prop_cycle(None)

    # Plot analytical function if available.
    if np.any(data_per_baseline_func):

        for (u_spfrq_func,
            v_spfrq_func,
            data_func,
            label) in zip(
            u_spfrq_per_baseline_func,
            v_spfrq_per_baseline_func,
            data_per_baseline_func,
            labels
        ):

            spfrq_func = data_handling.comp_spfrq(
                u_spfrq_func, v_spfrq_func
            )
            ax.plot(
                spfrq_func, data_func, label=label, linewidth=2.5,
                color=func_color
            )

    fontsize_L = 20
    fontsize_S = 16

    if show_baseline_legend:
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

    if save:

        file_format = "pdf"
        fig_name = fig_name
        fig.savefig(save_path+fig_name+"."+file_format, format=file_format,
                    bbox_inches='tight')

def plot_vis_for_fixed_wavelengths(
        bs,
        plot_data_uncertainty,
        figsize,
        save,
        save_path,
        set_title,
        show_baseline_legend
):

    # Make one pdf with each figure on one page.
    if save:

        file_format = "pdf"
        wavelength_descr = "for_single_waves"

        if hasattr(bs, "param_init_value"):
            sample_descr = bs.sample_descr
            fit_func_descr = bs.fit_func_descr
        else:
            sample_descr = None
            fit_func_descr = None

        pdf_name = get_vis_fig_name(
                fit_vis_or_vis2=bs.fit_vis_or_vis2,
                sample_descr=sample_descr,
                fit_func_descr=fit_func_descr,
                wavelength_descr=wavelength_descr
        )
        pdf = PdfPages(save_path+pdf_name+"."+file_format)

    for i_wave in range(bs.N_wavelength):

        ##### Precalculations.
        wavelength = bs.wavelengths[i_wave]
        wavelength_str = (
            f"{wavelength*1e6:.4f} micron"
        )

        u_spfrq_data = (
            bs.data_per_wavelength[i_wave].u_spfrq
        )
        v_spfrq_data = (
            bs.data_per_wavelength[i_wave].v_spfrq
        )
        spfrq_data = (
            bs.data_per_wavelength[i_wave].spfrq
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
                u_spfrq_func = (
                    np.linspace(spfrq_data.min() * 0.95,
                                spfrq_data.max() * 1.05,
                                1000
                    )
                )
                v_spfrq_func = (
                    np.zeros(len(u_spfrq_func))
                )
                spfrq_func = u_spfrq_func

            elif not bs.model_is_polar_symmetric:

                pass

        # Compute the data of the model if it is already set up. If it is
        # set up, but the bootstrapping has not been performed, plot the
        # model with the initial values. If the bootstrapping has been
        # performed, plot the model with the best fit parameters.

        # This is executed if the bootstrapping has been performed.
        if hasattr(bs, "results"):

            # Make dict containing only the varied params and their results to
            # feed the fit function.
            fitted_param = {
                param: bs.results[param][i_wave] \
                for param in bs.varied_params
            }
            # Obtain data for the best fit model.
            if bs.model_is_polar_symmetric:
                data_func = bs.fit_func(
                    u_spfrq_func, v_spfrq_func,
                    **bs.fixed_param, **fitted_param
                )

            elif not bs.model_is_polar_symmetric:
                data_func = bs.fit_func(
                    u_spfrq_data, v_spfrq_data,
                    **bs.fixed_param, **fitted_param
                )

            func_label = "result"

            # Create string for title with the varied parameter and their
            # fit results.
            title_varied_param_str = []
            for param in bs.varied_params:

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

            chi2 = bs.results["chi2"][i_wave]
            red_chi2 = bs.results["red_chi2"][i_wave]
            ndof = bs.results["ndof"][i_wave]
            title_chi2_str = (
                f"weighted chi2 = {chi2:.2}, red. chi2 = {red_chi2:.2}, "
                f"ndof = {ndof}"
            )

            title = (
                f"Wavelength = {wavelength_str}\n"
                f"Fitted parameters: {title_varied_param_str}\n"
                f"Fixed parameters: {title_fixed_param_str}\n"
                f"{title_chi2_str}"
            )

        # This is executed if the bootstrapping has not been performed, but
        # the model has been set up.
        elif hasattr(bs, "param_init_value"):

            if bs.model_is_polar_symmetric:
                data_func = bs.fit_func(
                    u_spfrq_func, v_spfrq_func,
                    **bs.param_init_value
            )

            elif not bs.model_is_polar_symmetric:

                data_func = bs.fit_func(
                    u_spfrq_data, v_spfrq_data,
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

        ax.errorbar(spfrq_data, data, yerr=data_error, fmt="x")

        if np.any(data_func):

            if bs.model_is_polar_symmetric:

                ax.plot(spfrq_func, data_func, label=func_label)
                if show_baseline_legend:
                    ax.legend()

            elif not bs.model_is_polar_symmetric:

                ax.scatter(
                    spfrq_data,
                    data_func,
                    marker="X",
                    s=106, #  markersize
                    color="red",
                    label=func_label)
                if show_baseline_legend:
                    ax.legend()

        ax.set_xlabel("spatial frequency /rad\u207B\u00b9")
        ax.set_ylabel(
            f"{"squared " if bs.fit_vis_or_vis2=="VIS2" else ""}visibility"
        )

        if set_title:
            ax.set_title(title, loc="left")

        if save:
            pdf.savefig(fig)

    if save:
        pdf.close()

def plot_relative_sed(
        bs,
        plot_data_uncertainty,
        figsize,
        save,
        save_path,
        wavelength_descr,
        title = "Relative SED"
):

    fig, ax = plt.subplots(figsize=figsize)

    if plot_data_uncertainty:

        # Specific format for plt.errorbar to work with one and multiple data
        # points at the same time.
        if type(bs.relative_sed["dust to star flux ratio"]) == np.float64:
            yerr = [
                [bs.relative_sed["-Delta dust to star flux ratio"]],
                [bs.relative_sed["+Delta dust to star flux ratio"]]
            ]
        else:
            yerr = (bs.relative_sed["-Delta dust to star flux ratio"],
                    bs.relative_sed["+Delta dust to star flux ratio"])

        ax.errorbar(
            x=bs.relative_sed["wavelength /m"],
            y=bs.relative_sed["dust to star flux ratio"],
            yerr=yerr, #  IMPORTANT: first lower error, second upper error.
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

    if save:
        file_format = "pdf"
        fig_name = get_results_file_name(
            suffix="relative_sed",
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr,
            file_format=file_format
        )
        fig.savefig(save_path+fig_name, format=file_format)

def plot_dust_sed(
        bs,
        plot_data_uncertainty,
        figsize,
        save,
        save_path,
        wavelength_descr,
        title = "Dust SED"
):

    fig, ax = plt.subplots(figsize=figsize)

    if plot_data_uncertainty:

        # Specific format for plt.errorbar to work with one and multiple data
        # points at the same time.
        if type(bs.sed["dust flux /Jy"]) == np.float64:
            yerr = [
                [bs.sed["-Delta dust flux /Jy"]],
                [bs.sed["+Delta dust flux /Jy"]]
            ]
        else:
            yerr = (bs.sed["-Delta dust flux /Jy"],
                    bs.sed["+Delta dust flux /Jy"])

        ax.errorbar(
            x=bs.sed["wavelength /m"],
            y=bs.sed["dust flux /Jy"],
            yerr=yerr, #  IMPORTANT: first lower error, second upper error.
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

    if save:
        file_format = "pdf"
        fig_name = get_results_file_name(
            suffix="dust_SED",
            fit_vis_or_vis2=bs.fit_vis_or_vis2,
            sample_descr=bs.sample_descr,
            fit_func_descr=bs.fit_func_descr,
            wavelength_descr=wavelength_descr,
            file_format=file_format
        )
        fig.savefig(save_path+fig_name, format=file_format)

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
        fit_func_descr: str, wavelength_descr: str
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

    Returns:
        fig_name: The figure file name.
    """

    # Remove the "VISAMP_" or "VIS2_" from fit_func_desc as it is present in
    # fit_vis_or_vis2 anyway.
    if fit_func_descr is not None:
        fit_func_descr = "_".join(fit_func_descr.split("_")[1:])

    fig_name = (
        f"{"_".join(filter(None, [fit_vis_or_vis2, sample_descr,
                                  fit_func_descr, wavelength_descr]))}"
    )

    return fig_name

def get_results_file_name(
        suffix: str, fit_vis_or_vis2: str, sample_descr: str,
        fit_func_descr: str, wavelength_descr: str, file_format: str
):
    """
    Returns the file name for a SED for given fit parameter and settings.

    Args:
        suffix: Suffix of the file name.
        sample_descr: Descriptor of the chosen sampling during
          bootstrapping, such as data points, baselines, observations or
          data points per wavelength.
        fit_func_descr: Descriptor of the chosen fit function.
        wavelength_descr: Contains information about the wavelengths.
          Typically either all wavelengths are fitted together or separately.
        file_format: The file format.

    Returns:
        file_name: The file name.
    """

    # Remove the "VISAMP_" or "VIS2_" from fit_func_desc as it is present in
    # fit_vis_or_vis2 anyway.
    fit_func_descr = "_".join(fit_func_descr.split("_")[1:])

    file_name = (
        f"{"_".join([suffix, fit_vis_or_vis2, sample_descr,
                     fit_func_descr, wavelength_descr])}"
        f".{file_format}"
    )

    return file_name