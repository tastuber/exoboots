import numpy as np

import data_handling
import model_functions

class Bootstrapper():

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

        # Set up random number generator.
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)

        # Select the analytic function for fitting.
        match self.model_selector:
            case 1:
                if fit_vis_or_vis2 == "VISAMP":
                    self.fit_function = \
                        model_functions.comp_VISAMP_limb_dark_disk_plus_overresolved
                elif fit_vis_or_vis2 == "VIS2":
                    self.fit_function = \
                        model_functions.comp_VIS2_limb_dark_disk_plus_overresolved
            case 2:
                if fit_vis_or_vis2 == "VISAMP":
                    self.fit_function = \
                        model_functions.comp_VISAMP_limb_dark_disk_plus_uniform_CSE
                elif fit_vis_or_vis2 == "VIS2":
                    self.fit_function = \
                        model_functions.comp_VIS2_limb_dark_disk_plus_uniform_CSE

        # Select how the data is bootstrapped.
        match self.bootstrap_selector:
            case 1:
                # Sample all vis. measurements, so sample the full data set
                self.sample = self.sample_data_points
            case 2:
                # Sample the baselines, so treat all data for one baseline fully
                # correlated.
                self.sample = self.sample_baselines
            case 3:
                # Sample complete observations.
                self.sample = self.sample_observations
            case 4:
                # Fit for selected wavelengths the data points, i.e., sample for
                # given wavelengths the baselines.
                pass

    def do_bootstrapping(self):
        """Perform the bootstrapping with the chosen settings."""

        for i in range(self.N_sample):

            data, data_error, wavelength, spatial_frequency, weight = \
                self.sample()


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

        full_data_set_tmp = data_handling.Full_data_set(
            self.resample(self.full_data_set.file_data_set_ls)
        )

        data_tuple = full_data_set_tmp.get_all_data_flattened()

        return data_tuple

    def sample_baselines(self):

        baseline_ls = self.resample(self.full_data_set.get_all_baselines())

        # Utilize class All_Baselines_per_File to create a temporary
        # Full_data_set to flatten all data arrays.
        full_data_set_tmp = data_handling.Full_data_set(
            [data_handling.All_Baselines_per_File(file="all chosen files",
                                   baseline_ls=baseline_ls)]
        )
        data_tuple = full_data_set_tmp.get_all_data_flattened()

        return data_tuple

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
            data_tuple: A tuple of iterables all with the same length. All iterables
              will be resampled individually, but in the same way.
            sample_size: The length of the created sample for each iterable. The
              default is the size of the input data.

        Returns:
            sample_tuple: The resampled input.
        """

        tuple_len = len(data_tuple)
        data_len = len(data_tuple[0])
        for i in range(1, tuple_len):
            if len(data_tuple[i]) != data_len:
                raise ValueError(
                    f"Tuple items do not have the same length. Found length of "
                    f"{data_len} for tupel item 0 and {len(data_tuple[i])} for "
                    f"tupel item {i}."
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
