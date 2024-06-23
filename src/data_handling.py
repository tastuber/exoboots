import oifits

import numpy as np

def read_settings(settings_file: str) -> (list, list, list, list):
    """
    Read the settings.txt file and put the data in lists.

    Obtain the oifits filenames, the wavelength selection and the excluded
    baselines.

    Args:
      settings_file: Filename of the input file as str.

    Returns:
      oifits_file_ls: List with the filenames of the Oifits files.
      wave_min_ls: List containing for each Oifits file the minimum wavelengths
        to be considered for analysis.
      wave_max_ls: List containing for each Oifits file the maximum wavelengths
        to be considered for analysis.
      exclude_baselines_ls_ls: List containing for each Oifits file the
        baselines to be excluded from analysis.
    """

    oifits_file_ls = []
    wave_min_ls = []
    wave_max_ls = []
    exclude_baselines_ls_ls = []

    with open(settings_file) as inp:
        lines = inp.readlines()
    for line in lines:
        if line[0] == '#':
            continue
        else:
            split = line.split()
            n_col = len(split)
            oifits_file_ls.append(split[0])
            # Convert the input of micron to meter as in the Oifits.
            wave_min_ls.append(float(split[1])*1e-6)
            wave_max_ls.append(float(split[2])*1e-6)
            if n_col > 3:
                exclude_baselines_ls_ls.append(split[3:])
            else:
                exclude_baselines_ls_ls.append([])

    return oifits_file_ls, wave_min_ls, wave_max_ls, exclude_baselines_ls_ls

def mask_wavelengths(oifits_obj: oifits.oifits, wave_min: float,
                     wave_max:float, fit_vis_or_vis2: str) -> oifits.oifits:
    """
    Masks all measurements of wavelength outside the chosen range.

    Takes the oifits object and returns a modified copy of it. Modified are
    only the masks of the masked arrays containing the data.

    Args:
        oifits_obj: The oifits object as instance of oifits.oifits.
        wave_min: The minimum wavelength in units of meter to be considered in
          the analysis. All data corresponding to smaller wavelengths are
          masked.
        wave_max: The maximum wavelength in units of meter to be considered in
          the analysis. All data corresponding to larger wavelengths are
          masked.
        fit_vis_or_vis2: String of either "VISAMP" or "VIS2" to select
          treatment of visibilities (VISAMP) or squared visibilities (VIS2).

    Returns:
        oifits_obj: Modified copy of the input oifits with masked data.
    """

    if fit_vis_or_vis2 == "VISAMP":

        for vis in oifits_obj.vis:

            wave_mask = vis.wavelength.eff_wave < wave_min
            vis.flag[wave_mask] = True
            wave_mask = vis.wavelength.eff_wave > wave_max
            vis.flag[wave_mask] = True

    elif fit_vis_or_vis2 == "VIS2":

        for vis2 in oifits_obj.vis2:

            wave_mask = vis2.wavelength.eff_wave < wave_min
            vis2.flag[wave_mask] = True
            wave_mask = vis2.wavelength.eff_wave > wave_max
            vis2.flag[wave_mask] = True

    return oifits_obj

class Baseline():
    """Data set of one baseline that can contain different types of data.

    Can be used for visibilities, squared visibilities, or closure phases.

    Attributes:
      baseline_id: Alphabetically ordered, concatenated station names. E.g.,
        A0B1 for the baseline between station A0 and B1 of the VLTI.
      data: The actual values of the measured quantity, e.g., values of
        visibility, squared visibility, or closure phase.
      data_error: The uncertainty of the values of the data attribute.
      wavelength: The wavelengths of the measurements.
      ucoord: The u-coordinate in units of meter [m].
      vcoord: The v-coordinate in units of meter [m].
      B: The projected baseline length in units of meter [m].
      spatial_frequency: The measured spatial frequencies. They are computed
        via spatial_frequency = baseline/wavelength and have the unit [1/rad].
        To get the common representation (e.g., used by the JMMC tool
        Oifitsexplorer) as 'Mega lambda' or '1e6/rad', one has to multiply with
        1e-6.
      weight: Weight of the data points for least-squares fitting.
    """

    def __init__(
        self,
        #instrument: str,
        baseline_id: str,
        data: np.array, data_error: np.array,
        wavelength: np.array,
        ucoord: float,
        vcoord: float,
        weight_mode: str="no weights",
    ):

        # Check whether all input data arrays are of the same length.
        if not len(set([len(data), len(data_error), len(wavelength)])) == 1:
            raise ValueError(
                "In construction of Baseline object, the array length of "
                "data, data error, and wavelength do not match."
            )

        #self.instrument = instrument
        self.baseline_id = baseline_id
        self.data = data
        self.data_error = data_error
        self.wavelength = wavelength
        self.ucoord = ucoord
        self.vcoord = vcoord

        # Compute projected baseline length from u-v-coordinates.
        self.B = np.sqrt(ucoord**2 + vcoord**2)

        # Compute spatial frequencies.
        self.spatial_frequency = self.B / self.wavelength

        # Set standard weights.
        self.set_weight(mode=weight_mode)

    def set_weight(self, mode: str):
        """
        Set the weights for each data point that can be used for fitting.

        Different types of weights of the data can be chosen that can be used
        in least-squares fitting.
        The options are:
          "no weights": All weights are the same and equal to one.
          "error": The errors of the data define the weights as
            1/error.
          "points per baseline": The weight is set as the inverse of the number
            of data points per baseline. This is motivated by the fact that for
            optical interferometry usually the different spectral data points
            of one baseline are highly correlated. Thus, it is reasonably to
            assume that only the entirety of data points per baseline give one
            independent data point. This has to be considered if data with
            different numbers of data points per baseline are analyzed jointly,
            e.g., when combining observations of multiple instruments or
            of the same instrument but with different spectral dispersions.
          "both": Both 'error' and 'points per baseline' weights combined via
            multiplication.

        Args:
          mode: String defining the type of weights. Choose from 'no weights',
            'error', 'points per baseline', or 'both'.
        """

        num_data_points = len(self.wavelength)

        if mode == "no weights":

            self.weight = np.array([1.0 for i in range(num_data_points)])

        elif mode == "error":

            self.weight = 1.0 / self.data_error

        elif mode == "points per baseline":

            self.weight = np.array(
                [1.0/num_data_points for i in range(num_data_points)]
            )

        elif mode == "both":

            weight_per_baseline = 1.0 / num_data_points
            self.weight = weight_per_baseline * (1.0/self.data_error)

        else:

            raise ValueError(
                "Wrong 'mode' to set weights. Choose from either 'no weights',"
                "'error', 'points per baseline', or 'both'."
            )

class All_Baselines_per_File():
    """Contains all Baseline objects of one Oifits file."""

    def __init__(self, file: str, baseline_ls: list):

        self.file = file
        self.baseline_ls = baseline_ls

    def __str__(self):

        return f"{len(self.baseline_ls)} baselines from {self.file}."

class Full_data_set():
    """
    Contains the full data set for the fit procedure.

    Attributes:
      file_data_set_ls: List of All_Baselines_per_File objects.
    """

    def __init__(self, file_data_set_ls: list):

        self.file_data_set_ls = file_data_set_ls

    def get_all_data_flattened(self) -> tuple:
        """
        Return the data of all baselines as a tuple of flattened arrays.

        Returns: Returns object attribute arrays in a tuple of the form
          (data, data_error, wavelength, spatial_frequency, weight).
        """

        data_ls = []
        data_error_ls = []
        wavelength_ls = []
        spatial_frequency_ls = []
        weight_ls = []

        for file_data_set in self.file_data_set_ls:

            for baseline in file_data_set.baseline_ls:

                data_ls.extend(baseline.data)
                data_error_ls.extend(baseline.data_error)
                wavelength_ls.extend(baseline.wavelength)
                spatial_frequency_ls.extend(baseline.spatial_frequency)
                weight_ls.extend(baseline.weight)

        data = np.asarray(data_ls).flatten()
        data_error = np.asarray(data_error_ls).flatten()
        wavelength = np.asarray(wavelength_ls).flatten()
        spatial_frequency = np.asarray(spatial_frequency_ls).flatten()
        weight = np.asarray(weight_ls).flatten()

        return data, data_error, wavelength, spatial_frequency, weight

    def get_all_baselines(self) -> list:
        """
        Returns all baselines of the data set as a list.

        Returns:
          baseline_ls: List of all baselines of all loaded Oifits files that
            have not be excluded from the process.
        """

        all_baseline_ls = []
        for file_data_set in self.file_data_set_ls:

            for baseline in file_data_set.baseline_ls:

                all_baseline_ls.append(baseline)

        return all_baseline_ls

    def get_data_per_wavelength(self) -> list:
        """
        Returns a list of Data_per_wavelength objects.

        Use this function only when analyzing Oifits with the exact same
        wavelength grid or a single file.
        Collects for each individual wavlength all associated data and puts it
        into a Data_per_wavelength object. All Data_per_wavelength objects are
        put into a list which is returned.
        The length of the output list is equal to the amount of individual
        wavelengths.

        Returns:
          data_per_wavelength_ls: List of Data_per_wavelength objects.
        """

        # Check whether all Baseline objects contain the exact same wavelength
        # grid. If not, raise an exception.
        error_message = (
            "The wavelength grids of two baselines do not match. This is "
            "required for the function get_data_per_wavelength() to be "
            "reasonable.\nProbably the data of two Oifits files with "
            "different wavelength grids are loaded."
        )
        wavelength = self.file_data_set_ls[0].baseline_ls[0].wavelength
        for file_data_set in self.file_data_set_ls:

            for baseline in file_data_set.baseline_ls:

                if len(wavelength) != len(baseline.wavelength):
                    raise ValueError(error_message)

                elif any (baseline.wavelength != wavelength):
                    raise ValueError(error_message)

        # Sort the data after wavelength.
        Data_per_wavelength_ls = []
        all_baselines = self.get_all_baselines()
        N_baselines = len(all_baselines)

        for i_wave, single_wavelength in enumerate(wavelength):

            data = np.zeros(N_baselines)
            data_error = np.zeros(N_baselines)
            spatial_frequency = np.zeros(N_baselines)
            weight = np.zeros(N_baselines)
            ucoord = np.zeros(N_baselines)
            vcoord = np.zeros(N_baselines)
            B = np.zeros(N_baselines)
            baseline_id_ls = []

            for i_baseline, baseline in enumerate(all_baselines):

                data[i_baseline] = baseline.data[i_wave]
                data_error[i_baseline] = baseline.data_error[i_wave]
                spatial_frequency[i_baseline] = \
                    baseline.spatial_frequency[i_wave]
                weight[i_baseline] = baseline.weight[i_wave]
                ucoord[i_baseline] = baseline.ucoord
                vcoord[i_baseline] = baseline.vcoord
                B[i_baseline] = baseline.B
                baseline_id_ls.append(baseline.baseline_id)

            Data_per_wavelength_ls.append(
                Data_per_wavelength(
                    single_wavelength, data, data_error, spatial_frequency,
                    weight, ucoord, vcoord, B, baseline_id_ls
                )
            )

        return Data_per_wavelength_ls

class Data_per_wavelength():
    """
    Contains for a one wavelength the associated data of various baselines.

    Can be used for visibilities, squared visibilities, or closure phases. It
    is very similar to the Baseline class, but instead of grouping data
    according to baseline, it groups data according to wavelength. A typical
    usage is to fit the data for specific wavelengths separately.

    Attributes:
      wavelength: The value of the wavelength in unit of meters [m].
      data: The data associated with the wavelengths from all baselines.
      data_error: Same as data, but for the data uncertainties.
      weight: Weight of the data points for least-squares fitting.
      ucoord: The u-coordinate in units of meter [m].
      vcoord: The v-coordinate in units of meter [m].
      B: The projected baseline length in units of meter [m].
      spatial_frequency: The measured spatial frequencies. They are computed
        via spatial_frequency = baseline/wavelength and have the unit [1/rad].
        To get the common representation (e.g., used by the JMMC tool
        Oifitsexplorer) as 'Mega lambda' or '1e6/rad', one has to multiply with
        1e-6.
    """

    def __init__(
        self, wavelength, data, data_error, spatial_frequency, weight, ucoord,
        vcoord, B, baseline_id_ls
    ):

        self.wavelength = wavelength
        self.data = data
        self.data_error = data_error
        self.spatial_frequency = spatial_frequency
        self.weight = weight
        self.ucoord = ucoord
        self.vcoord = vcoord
        self.B = B
        self.baseline_id_ls = baseline_id_ls

    def __str__(self):

        return (f"{len(self.data)} data points for the wavelength "
                f"{np.round(self.wavelength*1e6, 4)} micron")