import itertools

import lmfit
import numpy as np

import data_handling
import model_functions
import plotting

class Bootstrapper():
    """

    Any object is meant to be initialized once and no changes to input
    variables afterwards. If any attributes changed after initialization,
    nothing is reinitialized. If one wants to choose different settings, the
    object has to be reinitalized.
    """

    def __init__(
        self, N_sample: int, model_selector: int, bootstrap_selector: int,
        fit_vis_or_vis2: str,
        full_data_set: "Full_data_set",
        rng_seed: int
    ):

        self.N_sample = N_sample
        self.model_selector = model_selector
        self.bootstrap_selector = bootstrap_selector
        self.fit_vis_or_vis2 = fit_vis_or_vis2
        self.full_data_set = full_data_set
        self.data_per_wavelength = self.full_data_set.get_data_per_wavelength()
        self.N_wavelength = len(self.data_per_wavelength)
        self.wavelength_ls = [
            self.data_per_wavelength[i].wavelength
            for i in range(self.N_wavelength)
        ]

        # Set up random number generator.
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)

        # Select the analytic function for fitting.
        match self.model_selector:
            case 1:
                if fit_vis_or_vis2 == "VISAMP":
                    self.fit_function = \
                        model_functions.comp_VISAMP_limb_dark_disk_plus_overresolved
                    self.fit_function_descriptor = \
                        "VISAMP_limb_dark_disk_plus_overresolved"
                elif fit_vis_or_vis2 == "VIS2":
                    self.fit_function = \
                        model_functions.comp_VIS2_limb_dark_disk_plus_overresolved
                    self.fit_function_descriptor = \
                        "VIS2_limb_dark_disk_plus_overresolved"
            case 2:
                if fit_vis_or_vis2 == "VISAMP":
                    self.fit_function = \
                        model_functions.comp_VISAMP_limb_dark_disk_plus_uniform_CSE
                    self.fit_function_descriptor = \
                        "VISAMP_limb_dark_disk_plus_uniform_CSE"
                elif fit_vis_or_vis2 == "VIS2":
                    self.fit_function = \
                        model_functions.comp_VIS2_limb_dark_disk_plus_uniform_CSE
                    self.fit_function_descriptor = \
                        "VIS2_limb_dark_disk_plus_uniform_CSE"

        # Select how the data is bootstrapped.
        match self.bootstrap_selector:
            case 1:
                # Sample all vis. measurements, so sample the full data set
                self.sample = self.sample_data_points
                self.sample_descriptor = "sampled_data_points"
            case 2:
                # Sample the baselines, so treat all data for one baseline
                # fully correlated.
                self.sample = self.sample_baselines
                self.sample_descriptor = "sampled_baselines"
            case 3:
                # Sample complete observations.
                self.sample = self.sample_observations
                self.sample_descriptor = "sampled_observations"
            case 4:
                # Fit for selected wavelengths the data points, i.e., sample
                # for given wavelengths the baselines.
                self.sample_descriptor = (
                    "sampled_baselines_for_fixed_wavelength"
                )

    def setup_model(
            self, vary_param_ls: list[bool], value_param_ls: list[float],
            low_bound_param_ls: list[float], up_bound_param_ls: list[float]
    ):
        """
        Prepare the model for lmfit.

        Arguments are lists of the length equaling the number of fit function's
        parameters. The first item in the lists, respectively, belongs to the
        first paramter, the second item to the second parameter etc.

        Args:
            vary_param_ls: Defines whether the parameter shall be varied and
              optimized or remains fixed.
            value_param_ls: Gives the inital value of the parameters. For fixed
              parameters, this is the fixed value.
            low_bound_param_ls: Lower bound for the fitting process. Only has
              effect on varied parameters.
            up_bound_param_ls: Lower bound for the fitting process. Only has
              effect on varied parameters.
        """

        self.vary_param_ls = vary_param_ls
        self.N_varied_params = vary_param_ls.count(True)

        self.model = lmfit.Model(self.fit_function)

        # Add the parameters to the model.
        # Note that lmfit creates parameters also for fixed parameters.
        self.model_params = lmfit.Parameters()
        for (param_name, vary_param, value_param, low_bound_param,
             up_bound_param) in zip(
                self.model.param_names, self.vary_param_ls, value_param_ls,
                low_bound_param_ls, up_bound_param_ls
            ):
            self.model_params.add(
                param_name, vary=vary_param, value=value_param,
                min=low_bound_param, max=up_bound_param
            )

        # Create list of only the varied parameters.
        self.varied_param_ls = list(
            itertools.compress(self.model.param_names, self.vary_param_ls)
        )

    def do_bootstrapping(self):
        """Perform the bootstrapping with the chosen settings."""

        match self.bootstrap_selector:
            case 1 | 2 | 3:
                self.do_bootstrapping_all_wavelengths()

            case 4:
                self.do_bootstrapping_for_fixed_wavelengths()

    def do_bootstrapping_all_wavelengths(self):
        """Do one bootstrap fit for the full data set."""

        # Store the full results from the bootstrapping, thus the fit results
        # for every sample.
        self.sampling_results = np.zeros([self.N_varied_params,
                                               self.N_sample])

        # Store the high level result of the bootstrapping. That is, the
        # median, the 0.16 quantile, and the 0.84 quantile of the distribution
        # of best fit parameter values.
        self.param_results_median = np.zeros(self.N_varied_params)
        self.param_results_16percentile = np.zeros(self.N_varied_params)
        self.param_results_84percentile = np.zeros(self.N_varied_params)
        self.param_results_error_minus = np.zeros(self.N_varied_params)
        self.param_results_error_plus = np.zeros(self.N_varied_params)

        for i_sample in range(self.N_sample):

            (data, data_error, wavelength,
             spatial_frequency, weight) = self.sample()

            result = self.model.fit(
                data=data, params=self.model_params, weights=weight,
                spatial_frequency=spatial_frequency
            )

            # Lmfit gives results also for fixed parameters. Exclude them in
            # the result array.
            i_varied_param = 0
            for i_fit_param, best_value in enumerate(
                result.best_values.values()
            ):

                if self.vary_param_ls[i_fit_param]:

                    self.sampling_results[i_varied_param,
                                               i_sample] = best_value
                    i_varied_param += 1

        self.param_results_median = np.median(self.sampling_results, axis=1)
        self.param_results_16percentile = np.percentile(
            self.sampling_results, 16, axis=1
        )
        self.param_results_84percentile = np.percentile(
            self.sampling_results, 84, axis=1
        )
        self.param_results_error_minus = (
            self.param_results_median - self.param_results_16percentile
        )
        self.param_results_error_plus = (
            self.param_results_84percentile - self.param_results_median
        )

        # Store final results in dictionary.
        self.results = {}
        for i_varied_param, varied_param in enumerate(self.varied_param_ls):

            self.results[varied_param] = (
                self.param_results_median[i_varied_param]
            )
            self.results[f"+Delta {varied_param}"] = (
                self.param_results_error_plus[i_varied_param]
            )
            self.results[f"-Delta {varied_param}"] = (
                self.param_results_error_minus[i_varied_param]
            )

    def do_bootstrapping_for_fixed_wavelengths(self):
        """Do a bootstrap fit each wavelength."""

        self.results = {}

        # Store the full results from the bootstrapping, thus the fit results
        # for every sample.
        self.sampling_results = np.zeros(
            [self.N_wavelength, self.N_varied_params, self.N_sample]
        )

        # Store the high level result of the bootstrapping. That is, the
        # median, the 0.16 quantile, and the 0.84 quantile of the distribution
        # of best fit parameter values.
        param_results_median = np.zeros(
            [self.N_wavelength, self.N_varied_params]
        )
        param_results_error_minus = np.zeros(
            [self.N_wavelength, self.N_varied_params]
        )
        param_results_error_plus = np.zeros(
            [self.N_wavelength, self.N_varied_params]
        )

        for i_wavelength in range(self.N_wavelength):

            for i_sample in range(self.N_sample):

                (data, spatial_frequency, weight) = (
                    self.sample_baselines_for_fixed_wavelength(i_wavelength)
                )

                result = self.model.fit(
                    data=data, params=self.model_params, weights=weight,
                    spatial_frequency=spatial_frequency
                )

                # Lmfit gives results also for fixed parameters. Exclude them
                # in the result array.
                i_varied_param = 0
                for i_fit_param, best_value in enumerate(
                    result.best_values.values()
                ):

                    if self.vary_param_ls[i_fit_param]:

                        self.sampling_results[
                            i_wavelength, i_varied_param, i_sample
                        ] = best_value
                        i_varied_param += 1

            param_results_median[i_wavelength] = np.median(
                self.sampling_results[i_wavelength], axis=1
            )
            param_results_error_minus[i_wavelength] = (
                param_results_median[i_wavelength]
                - np.percentile(self.sampling_results[i_wavelength], 16, axis=1)
            )
            param_results_error_plus[i_wavelength] = (
                np.percentile(self.sampling_results[i_wavelength], 84, axis=1)
                - param_results_median[i_wavelength]
            )

            # Store final results in dictionary.
            wavelength_str = (
                f"{self.wavelength_ls[i_wavelength]*1e6:.4f} micron"
            )
            for i_varied_param, varied_param in enumerate(
                self.varied_param_ls
            ):

                self.results[f"{wavelength_str}: {varied_param}"] = (
                    param_results_median[i_wavelength, i_varied_param]
                )
                self.results[f"{wavelength_str}: +Delta {varied_param}"] = (
                    param_results_error_plus[i_wavelength, i_varied_param]
                )
                self.results[f"{wavelength_str}: -Delta {varied_param}"] = (
                    param_results_error_minus[i_wavelength, i_varied_param]
                )

    def sample_data_points(self):

        (data,
         data_error,
         wavelength,
         spatial_frequency,
         weight) = self.resample_parallel(
            self.full_data_set.get_all_data_flattened()
        )

        return data, data_error, wavelength, spatial_frequency, weight

    def sample_observations(self):

        full_data_set_tmp = data_handling.Full_data_set_from_list(
            self.resample(self.full_data_set.file_data_set_ls)
        )

        data_tuple = full_data_set_tmp.get_all_data_flattened()

        return data_tuple

    def sample_baselines(self):

        baseline_ls = self.resample(self.full_data_set.get_all_baselines())

        # Utilize class All_Baselines_per_File to create a temporary
        # Full_data_set to flatten all data arrays.
        full_data_set_tmp = data_handling.Full_data_set_from_list(
            [data_handling.All_Baselines_per_File(file="all chosen files",
                                   baseline_ls=baseline_ls)]
        )
        data_tuple = full_data_set_tmp.get_all_data_flattened()

        return data_tuple

    def sample_baselines_for_fixed_wavelength(self, i_wavelength: int):
        """
        Sample for a selected wavelength the different baselines.

        Args:
            i_wavelength: The index of the wavelength to be sampled. This picks
              a Data_per_wavelength object from self.data_per_wavelength.

        Returns:
        """

        (data, spatial_frequency, weight) = (
            self.resample_parallel(
                (self.data_per_wavelength[i_wavelength].data,
                 self.data_per_wavelength[i_wavelength].spatial_frequency,
                 self.data_per_wavelength[i_wavelength].weight)
            )
        )

        return data, spatial_frequency, weight

    def resample(self, data: "iterable", sample_size=None) -> "iterable":
        """
        Resample the input with replacement.

        Args:
            data: An iterable that will be resampled.
            sample_size: The length of the created sample. The default is the
              size of the input data.

        Returns:
            sample: The resampled input.
        """

        if not sample_size:
            sample_size = len(data)

        indices = self.rng.integers(
            0, high=sample_size-1, size=sample_size, endpoint=True
        )

        sample = np.take(data, indices)

        return sample

    def resample_parallel(self, data_tuple: tuple, sample_size=None) -> tuple:
        """
        Resample the input with replacement.

        Args:
            data_tuple: A tuple of iterables all with the same length. All
              iterables will be resampled individually, but in the same way.
            sample_size: The length of the created sample for each iterable.
              The default is the size of the input data.

        Returns:
            sample_tuple: The resampled input.
        """

        tuple_len = len(data_tuple)
        data_len = len(data_tuple[0])
        for i in range(1, tuple_len):
            if len(data_tuple[i]) != data_len:
                raise ValueError(
                    f"Tuple items do not have the same length. Found length "
                    f"of {data_len} for tupel item 0 and {len(data_tuple[i])} "
                    f"for tupel item {i}."
                )

        if not sample_size:
            sample_size = data_len

        indices = self.rng.integers(0, high=data_len-1, size=sample_size,
                            endpoint=True)

        data_ar = np.zeros((tuple_len, sample_size))

        for i in range(tuple_len):
            data_ar[i] = np.take(data_tuple[i], indices)

        sample_tuple = tuple(data_ar)

        return sample_tuple

    def plot_bootstrap_histograms(
        self,
        bins: int = 20,
        save_fig: bool = True,
        save_fig_path: str = "../figures/"
    ):

        match self.bootstrap_selector:

            # Plot one histogram per fitted parameter.
            case 1 | 2 | 3:

                for sampling_results, model_param_name in zip(
                    self.sampling_results, self.varied_param_ls
                ):

                    plotting.plot_histogram(
                        data=sampling_results,
                        param_descriptor=model_param_name,
                        sample_descriptor=self.sample_descriptor,
                        fit_function_descriptor=self.fit_function_descriptor,
                        wavelength_str="",
                        bins=bins,
                        save_fig=save_fig,
                        save_fig_path=save_fig_path
                    )

            # Plot for each wavelength one histogram per fitted parameter.
            case 4:

                for i_wavelength in range(self.N_wavelength):

                    wavelength = self.wavelength_ls[i_wavelength]
                    wavelength_str = (
                        f"{wavelength*1e6:.4f} micron"
                    )

                    for sampling_results, model_param_name in zip(
                        self.sampling_results[i_wavelength],
                        self.varied_param_ls
                    ):
                        # Missing: wavelength
                        plotting.plot_histogram(
                            data=sampling_results,
                            param_descriptor=model_param_name,
                            sample_descriptor=self.sample_descriptor,
                            fit_function_descriptor=self.fit_function_descriptor,
                            wavelength_str=wavelength_str,
                            bins=bins,
                            save_fig=save_fig,
                            save_fig_path=save_fig_path
                        )
