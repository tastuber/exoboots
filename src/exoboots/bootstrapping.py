import inspect
import itertools

import astropy.constants as c
import astropy.units as u
import lmfit
import numpy as np

from exoboots import data_handling
from exoboots import exceptions
from exoboots import model_functions
from exoboots import plotting

class Bootstrapper():
    """

    Any object is meant to be initialized once and no changes to input
    variables afterwards. If any attributes changed after initialization,
    nothing is reinitialized. If one wants to choose different settings, the
    object has to be reinitalized.
    """

    def __init__(
        self,
        N_sample: int,
        bootstrap_selector: int,
        full_data_set: "Full_data_set",
        weight_mode: str,
        rng_seed: int
    ):
        """
        Args:
            weight_mode: Defines how weights are computed. Options are
              "no weights", "error", "points per baseline", or "both".

              Options explained:
              "no weights": All weights are the same and equal to one.
              "error": The errors of the data define the weights as 1/error^2.
              "points per baseline": The weight is set as the inverse of the
                number of data points per baseline. This is motivated by the
                fact that for optical interferometry usually the different
                spectral data points of one baseline are highly correlated.
                Thus, it is reasonably to assume that only the entirety of data
                points per baseline give one independent data point. This has
                to be considered if data with different numbers of data points
                per baseline are analyzed jointly, e.g., when combining
                observations of multiple instruments or of the same instrument
                but with different spectral dispersions.
              "both": Both "error" and "points per baseline" weights combined
                via multiplication.
        """

        self.N_sample = N_sample
        self.bootstrap_selector = bootstrap_selector
        self.vis_or_vis2 = full_data_set.vis_or_vis2
        self.weight_mode = weight_mode
        self.full_data_set = full_data_set
        self.full_data_set.set_weight(weight_mode=self.weight_mode)
        self.model_is_setup = False

        self.default_save_path = "./"


        # Set up random number generator.
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)

        # Register analytical model functions.
        all_funcs = inspect.getmembers(model_functions, inspect.isfunction)
        prefix = "comp_VISAMP_"
        # Filter functions starting with the prefix and create a dictionary.
        # Remove the prefix from the keys.
        self.model_funcs = {
            name.removeprefix(prefix): func
            for name, func in all_funcs
            if name.startswith(prefix)
        }

        # Select how the data is bootstrapped.
        match self.bootstrap_selector:
            case 1:
                # Sample all vis. measurements, so sample the full data set
                self.sample = self.sample_data_points
                self.sample_descr = "sampled_data_points"
                self.wavelength_descr = "all_waves"
                self.do_bootstrapping = self.do_bootstrapping_all_wavelengths
            case 2:
                # Sample the baselines, so treat all data for one baseline
                # fully correlated.
                self.sample = self.sample_baselines
                self.sample_descr = "sampled_baselines"
                self.wavelength_descr = "all_waves"
                self.do_bootstrapping = self.do_bootstrapping_all_wavelengths
            case 3:
                # Sample complete observations.
                self.sample = self.sample_observations
                self.sample_descr = "sampled_observations"
                self.wavelength_descr = "all_waves"
                self.do_bootstrapping = self.do_bootstrapping_all_wavelengths
            case 4:
                # Fit for selected wavelengths the data points, i.e., sample
                # for given wavelengths the baselines.
                self.data_per_wavelength = (
                    self.full_data_set.get_data_per_wavelength()
                )
                self.N_wavelength = len(self.data_per_wavelength)
                self.wavelengths = [
                    self.data_per_wavelength[i].wavelength
                    for i in range(self.N_wavelength)
                ]
                self.sample_descr = (
                    "sampled_baselines_for_fixed_wavelength"
                )
                self.wavelength_descr = "for_single_waves"
                self.do_bootstrapping = \
                    self.do_bootstrapping_for_fixed_wavelengths

    def list_model_keys(self, sorted_by_symmetry: bool=True):
        """
        List analytical models. split into polar non polar symmetric ones.

        Args:
            sorted_by_symmetry: Split into polar symmetric and non polar
              symmetric models.
        """

        if not sorted_by_symmetry:
            for key in self.model_funcs.keys():
                print(key)

        else:

            print("Polar symmetric models:")
            for key in self.model_funcs.keys():
                if self.model_funcs[key].model_category == "polar_symmetric":
                    print(f"    {key}")
            print("")
            print("Non polar symmetric models:")
            for key in self.model_funcs.keys():
                if self.model_funcs[key].model_category \
                    == "non_polar_symmetric":
                    print(f"    {key}")

    def setup_model(
            self,
            model_key: str,
            vary_param: dict[bool],
            param_init_value: dict[float],
            param_bounds: dict[tuple[float]] | dict = {}
    ):
        """
        Prepare the model for lmfit.

        Arguments are dicts with the parameters of the fit function as keys.
        Those can always accessed via self.fit_func.__annotations__.

        Args:
            model_key: Dictionary key that selects the analytical model.
            vary_param: Defines whether the parameter shall be varied and
              optimized or remains fixed.
            param_init_value: Gives the inital value of the parameters. For
              fixed parameters, this is the fixed value.
            param_bounds: Optional; parameter bounds for the fit. Contains
              tuple with the first value being the lower bound and the second
              value being the upper bound. The bounds only have and effect on
              varied parameters. If no bounds are given for a parameter,
              +/- infinity are set as bounds.

        Raises:
            KeyError: In case the input arguments have keys not matching the
              parameters of the fit function.
        """

        fit_func = self.model_funcs[model_key]

        if fit_func.model_category == "polar_symmetric":
            self.model_is_polar_symmetric = True
        elif fit_func.model_category == "non_polar_symmetric":
            self.model_is_polar_symmetric = False

        # To yield the squared visibility, square the visibility amplitude.
        if self.vis_or_vis2 == "VIS2":

            self.fit_func = model_functions.square_func(fit_func)
            self.fit_func_descr = f"VIS2_{model_key}"

        elif self.vis_or_vis2 == "VISAMP":

            self.fit_func = fit_func
            self.fit_func_descr = f"VISAMP_{model_key}"

        self.model = lmfit.Model(
            self.fit_func,
            independent_vars=["u_spfrq", "v_spfrq"]
        )

        # If no parameter bounds are given for a particular parameter, set
        # it to +/- infinity.
        for param_name in self.model.param_names:

            if param_name not in param_bounds:

                param_bounds[param_name] = (-np.inf, np.inf)

        # Check whether the input dicts have the right keys.
        if set(vary_param) != set(self.model.param_names):
            raise KeyError(
                "The argument vary_param of setup_model() has the wrong"
                f" keys. It has {list(vary_param.keys())} but needs "
                f"{self.model.param_names}."
            )
        if set(param_init_value) != set(self.model.param_names):
            raise KeyError(
                "The argument param_init_value of setup_model() has the wrong"
                f" keys. It has {list(param_init_value.keys())} but needs "
                f"{self.model.param_names}."
            )
        if set(param_bounds) != set(self.model.param_names):
            raise KeyError(
                "The argument param_bounds of setup_model() has the wrong"
                f" keys. It has {list(param_bounds.keys())} but needs "
                f"{self.model.param_names}."
            )

        self.vary_param = vary_param
        self.N_varied_params = sum(
            1 for condition in vary_param.values() if condition
        )
        self.param_init_value = param_init_value

        # Add the parameters to the model.
        # Note that lmfit creates parameters also for fixed parameters.
        self.model_params = lmfit.Parameters()
        for param_name in self.model.param_names:
            self.model_params.add(
                param_name,
                vary=vary_param[param_name],
                value=param_init_value[param_name],
                min=param_bounds[param_name][0],
                max=param_bounds[param_name][1]
            )

        # Create list of the varied parameters.
        self.varied_params = [
            param_name for param_name in self.model.param_names
            if vary_param[param_name]
        ]

        # Create dictionary of the fixed parameters. This is mainly for easier
        # plotting.
        self.fixed_param = {
            param_name: param_init_value[param_name]
            for param_name in self.model_params
            if not vary_param[param_name]
        }

        self.model_is_setup = True

    def do_bootstrapping_all_wavelengths(self):
        """
        Do one bootstrap fit for the full data set.

        Raises:
            NoModelError: Bootstrapping is started before a model is set up by
              executing exoboots.setup_model().
        """

        if not self.model_is_setup:
            raise exceptions.NoModelError()

        # Store the full results from the bootstrapping, thus the fit results
        # for every sample.
        self.sampling_results = np.zeros([self.N_varied_params,
                                          self.N_sample])

        # Compute the high level result of the bootstrapping. That is, the
        # median as the best fit value and the difference of the median and the
        # 0.16 (0.84) quantile as the lower (upper) uncertainty.
        param_results_median = np.zeros(self.N_varied_params)
        param_results_error_minus = np.zeros(self.N_varied_params)
        param_results_error_plus = np.zeros(self.N_varied_params)

        for i_sample in range(self.N_sample):

            (data, data_error, wavelength,
             u_spfrq, v_spfrq, weight) = self.sample()

            result = self.model.fit(
                data=data, params=self.model_params, weights=weight,
                u_spfrq=u_spfrq,
                v_spfrq=v_spfrq
            )

            # Store the sampling results for the varied parameters.
            for i_varied_param, param_name in enumerate(self.varied_params):

                self.sampling_results[i_varied_param, i_sample] = (
                    result.best_values[param_name]
                )

        param_results_median = np.median(self.sampling_results, axis=1)
        param_results_error_minus = (
            param_results_median
            - np.quantile(self.sampling_results, 0.16, axis=1)
        )
        param_results_error_plus = (
            np.quantile(self.sampling_results, 0.84, axis=1)
            - param_results_median
        )

        # Store final results in dictionary.
        # If the parameter is the relative flux of the dust ('f'), write
        # this in an additional dictionary 'relative_sed' for easy plotting and
        # saving.
        self.results = {}
        relative_sed = np.zeros(4)
        _, _, tmp_wave, _, _, _ = self.full_data_set.get_all_data_flattened()
        mean_wavelength = np.mean(tmp_wave)

        for i_varied_param, varied_param in enumerate(self.varied_params):

            param_median = param_results_median[i_varied_param]
            param_error_plus = param_results_error_plus[i_varied_param]
            param_error_minus = param_results_error_minus[i_varied_param]

            self.results[varied_param] = param_median
            self.results[f"+Delta {varied_param}"] = param_error_plus
            self.results[f"-Delta {varied_param}"] = param_error_minus

            if varied_param == "f_cse":

                relative_sed[0] = mean_wavelength
                relative_sed[1] = param_median
                relative_sed[2] = param_error_plus
                relative_sed[3] = param_error_minus

        if np.any(relative_sed):

            self.relative_sed = {
                "wavelength /m": relative_sed[0],
                "dust to star flux ratio": relative_sed[1],
                "+Delta dust to star flux ratio": relative_sed[2],
                "-Delta dust to star flux ratio": relative_sed[3]
            }

        # Compute chi^2, reduced chi^2, and degreed of freedom and append it to
        # results dictionary.
        chi2, red_chi2, ndof = self.compute_chi2(ndof=None)
        self.results["chi2"] = chi2
        self.results["red_chi2"] = red_chi2
        self.results["ndof"] = ndof

    def do_bootstrapping_for_fixed_wavelengths(self):
        """
        Do a bootstrap fit each wavelength.

        Raises:
            NoModelError: Bootstrapping is started before a model is set up by
              executing exoboots.setup_model().
        """

        if not self.model_is_setup:
            raise exceptions.NoModelError()

        self.results = {}
        relative_sed = np.zeros([4, self.N_wavelength])

        # Store the full results from the bootstrapping, thus the fit results
        # for every sample.
        self.sampling_results = np.zeros(
            [self.N_wavelength, self.N_varied_params, self.N_sample]
        )

        # Compute the high level result of the bootstrapping. That is, the
        # median as the best fit value and the difference of the median and the
        # 0.16 (0.84) quantile as the lower (upper) uncertainty.
        param_results_median = np.zeros(
            [self.N_wavelength, self.N_varied_params]
        )
        param_results_error_minus = np.zeros(
            [self.N_wavelength, self.N_varied_params]
        )
        param_results_error_plus = np.zeros(
            [self.N_wavelength, self.N_varied_params]
        )

        for i_wave in range(self.N_wavelength):

            wavelength = self.wavelengths[i_wave]

            for i_sample in range(self.N_sample):

                (data, u_spfrq, v_spfrq, weight) = (
                    self.sample_baselines_for_fixed_wavelength(i_wave)
                )

                result = self.model.fit(
                    data=data, params=self.model_params, weights=weight,
                    u_spfrq=u_spfrq,
                    v_spfrq=v_spfrq
                )

                # Store the sampling results for the varied parameters.
                for i_varied_param, param_name in enumerate(self.varied_params):

                    self.sampling_results[i_wave, i_varied_param, i_sample] = (
                        result.best_values[param_name]
                    )

            param_results_median[i_wave] = np.median(
                self.sampling_results[i_wave], axis=1
            )
            param_results_error_minus[i_wave] = (
                param_results_median[i_wave]
                - np.quantile(self.sampling_results[i_wave], 0.16, axis=1)
            )
            param_results_error_plus[i_wave] = (
                np.quantile(self.sampling_results[i_wave], 0.84, axis=1)
                - param_results_median[i_wave]
            )

            # If the parameter is the relative flux of the dust ('f'), write
            # this in a dictionary 'relative_sed' for easy plotting and saving.
            wavelength_str = (
                f"{wavelength*1e6:.4f} micron"
            )
            for i_varied_param, varied_param in enumerate(
                self.varied_params
            ):

                param_median = param_results_median[i_wave,
                                                    i_varied_param]
                param_error_plus = param_results_error_plus[i_wave,
                                                            i_varied_param]
                param_error_minus = param_results_error_minus[i_wave,
                                                              i_varied_param]

                if varied_param == "f_cse":

                    relative_sed[0][i_wave] = wavelength
                    relative_sed[1][i_wave] = param_median
                    relative_sed[2][i_wave] = param_error_plus
                    relative_sed[3][i_wave] = param_error_minus

            if np.any(relative_sed):

                self.relative_sed = {
                    "wavelength /m": relative_sed[0],
                    "dust to star flux ratio": relative_sed[1],
                    "+Delta dust to star flux ratio": relative_sed[2],
                    "-Delta dust to star flux ratio": relative_sed[3]
                }

        # Store final results in dictionary.
        self.results["wavelength"] = self.wavelengths
        for i_varied_param, varied_param in enumerate(
                self.varied_params
            ):

                self.results[varied_param] = (
                    param_results_median[:, i_varied_param]
                )
                self.results[f"+Delta {varied_param}"] = (
                    param_results_error_plus[:, i_varied_param]
                )
                self.results[f"-Delta {varied_param}"] = (
                    param_results_error_minus[:, i_varied_param]
                )

        # Compute chi^2, reduced chi^2, and degreed of freedom and append it to
        # results dictionary.
        chi2, red_chi2, ndof = self.compute_chi2(ndof=None)
        self.results["chi2"] = chi2
        self.results["red_chi2"] = red_chi2
        self.results["ndof"] = ndof

    def compute_chi2(self, ndof: int | None = None) \
        -> tuple[float, float, int] \
            | tuple[list[float], list[float], list[int]]:
        """
        Compute weighted chi^2, the reduced chi2 and the degrees of freedom.

        Compute the weighted chi2 as the sum of the ratio of the squared
        difference of the model and the data, and the squared uncertainties.
        The reduced chi2 is computed as the chi2 divided by the degrees of
        freedom (ndof). If no specific ndof are given, compute it as the
        number of data points minus the number of varied parameters.

        If one fit to all wavelengths was done: Return tuple with the
        weighted chi2, the reduced chi2, and the number of degrees of freedom.
          If the data was fitted for each wavelength separately: Return tuple
        with three lists with the weighted chi2, the reduced chi2, and the
        number of degrees of freedom for each wavelength, respectively.

        Args:
            The number of degrees of freedom.

        Returns:
            chi2: The weighted chi2.
            red_chi2: The reduced chi2.
            ndof: The number of degrees of freedom. Either computed
              from the data or given by the input argument.
        """

        if self.do_bootstrapping == self.do_bootstrapping_all_wavelengths:

            fitted_param = {
                param: self.results[param] for param in self.varied_params
            }
            (data, data_error, _,
             u_spfrq, v_spfrq, _) = \
                self.full_data_set.get_all_data_flattened()

            data_func = self.fit_func(
                u_spfrq=u_spfrq,
                v_spfrq=v_spfrq,
                **self.fixed_param,
                **fitted_param
            )
            chi2 = np.sum(((data-data_func)/data_error)**2)

            if not ndof:
                ndof = len(data) - len(self.varied_params)

            red_chi2 = chi2 / ndof

            return chi2, red_chi2, ndof

        elif self.do_bootstrapping_for_fixed_wavelengths:

            chi2_per_wavelength = []
            red_chi2_per_wavelength = []
            ndof_per_wavelength = []
            for i_wave in range(self.N_wavelength):
                fitted_param = {
                    param: self.results[param][i_wave] \
                    for param in self.varied_params
                }
                data = self.data_per_wavelength[i_wave].data
                data_error = self.data_per_wavelength[i_wave].data_error
                u_spfrq = (
                    self.data_per_wavelength[i_wave].u_spfrq
                )
                v_spfrq = (
                    self.data_per_wavelength[i_wave].v_spfrq
                )
                data_func = self.fit_func(
                    u_spfrq=u_spfrq,
                    v_spfrq=v_spfrq,
                    **self.fixed_param,
                    **fitted_param
                )

                chi2 = np.sum(((data-data_func)/data_error)**2)
                chi2_per_wavelength.append(chi2)

                if not ndof:
                    ndof = len(data) - len(self.varied_params)

                red_chi2_per_wavelength.append(chi2 / ndof)
                ndof_per_wavelength.append(ndof)


            return (chi2_per_wavelength,
                    red_chi2_per_wavelength,
                    ndof_per_wavelength)

    def compute_dust_sed(
            self, T_star: float, R_star: float, dist_star: float
        ) -> tuple:
        """
        Compute the flux from the dust based on a Planck curve for the star.

        Args:
            T_star: Effective temperature of the central star in units of K.
            R_star: Radius of the star in units of solar radii.
            dist_star: Distance to the star in units of parsec.

        Returns:
            sed: Tuple of five numpy arrays in the form: (wavelength [m], dust
              SED [Jy], its plus uncertainty [Jy], its minus uncertainty [Jy],
              stellar SED [Jy]).
        """

        T_star = T_star * u.K
        R_star = R_star * u.Rsun
        dist_star = dist_star * u.parsec
        wavelength = self.relative_sed["wavelength /m"] * u.m

        hd1 = (2.0*c.h*c.c**2) / wavelength**5
        hd2 = ((c.h*c.c) / (c.k_B*T_star*wavelength)).decompose()
        planck = hd1 / (np.exp(hd2) - 1.0)
        self.hd1 = hd1
        self.hd2 = hd2
        self.planck  = planck

        # Compute the stellar flux in units of Jy.
        # Explanation:
        # 4*Pi*R_star is the surface of the star.
        # The second Pi comes from the integration of the half sphere for
        # every emitting point (Lambert Cosine Law).
        # The division by 4*Pi*dist_star is the geometrical dilution (the
        # more distant the star, the less flux we receive).
        # The factor wavelength**2/c converts into units of Jy.
        F_star = (
            4.0 * np.pi * R_star**2 * np.pi * planck
            / (4.0*np.pi*dist_star**2) * (wavelength**2/c.c)
        ).to(u.Jy)

        sed = self.relative_sed["dust to star flux ratio"] * F_star
        sed_error_plus = (
            self.relative_sed["+Delta dust to star flux ratio"] * F_star
        )
        sed_error_minus = (
            self.relative_sed["-Delta dust to star flux ratio"] * F_star
        )

        self.sed = {
            "wavelength /m": wavelength.value,
            "dust flux /Jy": sed.value,
            "+Delta dust flux /Jy": sed_error_plus.value,
            "-Delta dust flux /Jy": sed_error_minus.value,
            "star flux /Jy": F_star.value
        }

        return self.sed

    def save_fit_results(self, save_path: str | None = None):

        if not save_path:
            save_path = self.default_save_path

        file_name = plotting.get_results_file_name(
            suffix="fit_results",
            vis_or_vis2=self.vis_or_vis2,
            sample_descr=self.sample_descr,
            fit_func_descr=self.fit_func_descr,
            wavelength_descr=self.wavelength_descr,
            file_format="txt"
        )

        # TODO: Write header with information about the fit settings.
        header = None
        data_handling.write_dict_to_txt(
            d=self.results, file=file_name, path=save_path, header=header
        )

    def save_relative_sed(self, save_path: str | None = None):

        if not save_path:
            save_path = self.default_save_path

        file_name = plotting.get_results_file_name(
            suffix="relative_sed",
            vis_or_vis2=self.vis_or_vis2,
            sample_descr=self.sample_descr,
            fit_func_descr=self.fit_func_descr,
            wavelength_descr=self.wavelength_descr,
            file_format="txt"
        )

        # TODO: Write header with information about the fit settings.
        header = None
        data_handling.write_dict_to_txt(
            d=self.relative_sed, file=file_name, path=save_path, header=header
        )

    def save_dust_sed(self, save_path: str | None = None):

        if not save_path:
            save_path = self.default_save_path

        file_name = plotting.get_results_file_name(
            suffix="dust_SED",
            vis_or_vis2=self.vis_or_vis2,
            sample_descr=self.sample_descr,
            fit_func_descr=self.fit_func_descr,
            wavelength_descr=self.wavelength_descr,
            file_format="txt"
        )

        # TODO: Write header with information about the fit settings.
        header = None

        try:
            data_handling.write_dict_to_txt(
                d=self.sed, file=file_name, path=save_path, header=header
            )
        except AttributeError:
            print("No dust SED has been computed yet.\n"
                  "Execute: Bootstrapper.compute_dust_sed()")
            raise

    def sample_data_points(self):

        (data,
         data_error,
         wavelength,
         u_spfrq,
         v_spfrq,
         weight) = self.resample_parallel(
            self.full_data_set.get_all_data_flattened()
        )

        return (
            data, data_error, wavelength, u_spfrq,
            v_spfrq, weight
        )

    def sample_observations(self):

        full_data_set_tmp = data_handling.Full_data_set_from_list(
            self.resample(self.full_data_set.file_data_sets)
        )

        data = full_data_set_tmp.get_all_data_flattened()

        return data

    def sample_baselines(self):

        baselines = self.resample(self.full_data_set.get_all_baselines())

        # Utilize class All_Baselines_per_File to create a temporary
        # Full_data_set to flatten all data arrays.
        full_data_set_tmp = data_handling.Full_data_set_from_list(
            [data_handling.All_Baselines_per_File(file="all chosen files",
                                                  baselines=baselines)]
        )
        data = full_data_set_tmp.get_all_data_flattened()

        return data

    def sample_baselines_for_fixed_wavelength(self, i_wave: int):
        """
        Sample for a selected wavelength the different baselines.

        Args:
            i_wave: The index of the wavelength to be sampled. This picks
              a Data_for_one_wavelength object from self.data_per_wavelength.

        Returns:
        """

        (data, u_spfrq, v_spfrq, weight) = (
            self.resample_parallel(
                (self.data_per_wavelength[i_wave].data,
                 self.data_per_wavelength[i_wave].u_spfrq,
                 self.data_per_wavelength[i_wave].v_spfrq,
                 self.data_per_wavelength[i_wave].weight)
            )
        )

        return data, u_spfrq, v_spfrq, weight

    def resample(self, data: "iterable", sample_size=None) -> "iterable":
        """
        Resample the input with replacement.

        Args:
            data: An iterable that will be resampled.
            sample_size: The length of the created sample. The default is the
              size of the input data.

        Returns:
            resampled_data: The resampled input.
        """

        if sample_size is None:
            sample_size = len(data)

        indices = self.rng.integers(
            0, high=sample_size-1, size=sample_size, endpoint=True
        )

        resampled_data = np.take(data, indices)

        return resampled_data

    def resample_parallel(self, data: tuple, sample_size=None) -> tuple:
        """
        Resample the input with replacement.

        Args:
            data: A tuple of iterables all with the same length. All
              iterables will be sampled individually, but in the same way.
            sample_size: The length of the created sample for each iterable.
              The default is the size of the input data.

        Returns:
            resampled_data: The resampled input.
        """

        len_data = len(data)
        len_element = len(data[0])
        for i in range(1, len_data):
            if len(data[i]) != len_element:
                raise ValueError(
                    f"Tuple items do not have the same length. Found length "
                    f"of {len_element} for tupel item 0 and {len(data[i])} "
                    f"for tupel item {i}."
                )

        if sample_size is None:
            sample_size = len_element

        indices = self.rng.integers(0, high=len_element-1, size=sample_size,
                            endpoint=True)

        resampled_data = np.zeros((len_data, sample_size))

        for i in range(len_data):
            resampled_data[i] = np.take(data[i], indices)

        resampled_data = tuple(resampled_data)

        return resampled_data

    def plot_bootstrap_histograms(
        self,
        bins: int = 20,
        figsize: tuple[float, float] = (10, 8),
        save: bool = True,
        save_path: str | None = None
    ):

        if not save_path:
            save_path = self.default_save_path

        plotting.call_plot_histogram(
            bs=self,
            bins=bins,
            figsize=figsize,
            save=save,
            save_path=save_path
        )

    def plot_data(
        self,
        plot_data_uncertainty: bool = True,
        figsize: tuple[float, float] = (20, 12),
        save: bool = True,
        save_path: str | None = None,
        set_title: bool = True,
        show_baseline_legend: bool = True
    ):

        if not save_path:
            save_path = self.default_save_path

        plotting.plot_vis(
            bs=self,
            plot_data_uncertainty=plot_data_uncertainty,
            figsize=figsize,
            save=save,
            save_path=save_path,
            set_title=set_title,
            show_baseline_legend=show_baseline_legend
        )

    def plot_dust_sed(
        self,
        plot_data_uncertainty: bool = True,
        figsize: tuple[float, float] = (16, 8),
        save: bool = True,
        save_path: str | None = None,
        title: str = ""
    ):

        if not save_path:
            save_path = self.default_save_path

        try:
            plotting.plot_dust_sed(
                bs=self,
                plot_data_uncertainty=plot_data_uncertainty,
                figsize=figsize,
                save=save,
                save_path=save_path,
                wavelength_descr=self.wavelength_descr,
                title=title
            )
        except AttributeError:
            print("No dust SED has been computed yet.\n"
                  "Execute: Bootstrapper.compute_dust_sed()")
            raise

    def plot_relative_sed(
        self,
        plot_data_uncertainty: bool = True,
        figsize: tuple[float, float] = (16, 8),
        save: bool = True,
        save_path: str | None = None,
        title: str = ""
    ):

        if not save_path:
            save_path = self.default_save_path

        plotting.plot_relative_sed(
            bs=self,
            plot_data_uncertainty=plot_data_uncertainty,
            figsize=figsize,
            save=save,
            save_path=save_path,
            wavelength_descr=self.wavelength_descr,
            title=title
        )
