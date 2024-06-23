import numpy as np

def resample(data: "iterable", sample_size=None) -> "iterable":
    """
    Resample the input with replacement.

    Args:
      data: An iterable that will be resampled.
      sample_size: The length of the created sample. The default is the size of
        the input data.

    Returns:
      sample: The resampled input.
    """

    if not sample_size:
        sample_size = len(data)

    indices = rng.integers(0, high=sample_size-1, size=sample_size,
                           endpoint=True)

    sample = np.take(data, indices)

    return sample

def resample_parallel(data_tuple: tuple, sample_size=None) -> tuple:
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

    indices = rng.integers(0, high=data_len-1, size=sample_size,
                           endpoint=True)

    data_ar = np.zeros((tuple_len, sample_size))

    for i in range(tuple_len):
        data_ar[i] = np.take(data_tuple[i], indices)

    sample_tuple = tuple(data_ar)

    return sample_tuple

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

        # Set up random number generator.
        self.rng = np.random.default_rng(rng_seed)

        # Select the analytic function for fitting.
        match self.model_selector:
            case 1:
                self.fit_function = comp_vis_limb_dark_disk_plus_overresolved
            case 2:
                self.fit_function = comp_vis_limb_dark_disk_plus_uniform_CSE

        # Select how the data is bootstrapped.
        match self.bootstrap_selector:
            case 1:
                # Sample all vis. measurements, so sample the full data set
                pass
            case 2:
                # Sample the baselines, so treat all data for one baseline fully
                # correlated.
                pass
            case 3:
                # Sample complete observations.
                pass
            case 4:
                # Fit for selected wavelengths the data points, i.e., sample for
                # given wavelengths the baselines.
                pass

    def sample_observations(self):

        full_data_set_tmp = Full_data_set(
            resample(self.full_data_set.file_data_set_ls)
        )

        data_tuple = full_data_set_tmp.get_all_data_flattened()

        return data_tuple

    def sample_data_points(self):

        (data,
         data_error,
         wavelength,
         spatial_frequency,
         weight) = resample_parallel(
            self.full_data_set.get_all_data_flattened()
        )

        return data, data_error, wavelength, spatial_frequency, weight

    def sample_baselines(self):

        baseline_ls = resample(self.full_data_set.get_all_baselines())

        # Utilize class All_Baselines_per_File to create a temporary
        # Full_data_set to flatten all data arrays.
        full_data_set_tmp(
            [All_Baselines_per_File(file="all chosen files",
                                   baseline_ls=baseline_ls)]
        )
        data_tuple = full_data_set_tmp.get_all_data_flattened()

        return data_tuple
