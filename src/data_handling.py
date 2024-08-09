import oifits

import numpy as np
from scipy.stats import binned_statistic

import exceptions
import plotting

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

def unflag_all_wavelengths(oifits_obj: oifits.oifits,
                           fit_vis_or_vis2: str) -> oifits.oifits:
    """
    Unflag all measurements.

    Takes the oifits object and returns a modified copy of it. Modified are
    only the flags of the data. The oifits module creates the actual masked
    arrays with data on the fly making use of the flag attribute.

    Args:
        oifits_obj: The oifits object as instance of oifits.oifits.
        fit_vis_or_vis2: String of either "VISAMP" or "VIS2" to select
          treatment of visibilities (VISAMP) or squared visibilities (VIS2).

    Returns:
        oifits_obj: Modified copy of the input oifits with flagged data.
    """

    if fit_vis_or_vis2 == "VISAMP":

        for vis in oifits_obj.vis:

            vis.flag = vis.flag * False
            vis.flag = vis.flag * False

    elif fit_vis_or_vis2 == "VIS2":

        for vis2 in oifits_obj.vis2:

            vis2.flag = vis2.flag * False
            vis2.flag = vis2.flag * False

    return oifits_obj

def flag_wavelengths(oifits_obj: oifits.oifits, wave_min: float,
                     wave_max:float, fit_vis_or_vis2: str) -> oifits.oifits:
    """
    Masks all measurements outside the chosen wavelength range.

    Takes the oifits object and returns a modified copy of it. Modified are
    only the flags of the data. The oifits module creates the actual masked
    arrays with data on the fly making use of the flag attribute.

    Args:
        oifits_obj: The oifits object as instance of oifits.oifits.
        wave_min: The minimum wavelength in units of meter to be considered in
          the analysis. All data corresponding to smaller wavelengths are
          flagged.
        wave_max: The maximum wavelength in units of meter to be considered in
          the analysis. All data corresponding to larger wavelengths are
          flagged.
        fit_vis_or_vis2: String of either "VISAMP" or "VIS2" to select
          treatment of visibilities (VISAMP) or squared visibilities (VIS2).

    Returns:
        oifits_obj: Modified copy of the input oifits with flagged data.
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

# TODO:
# Write function mask_baselines()

def write_dict_to_txt(
        d: dict, file: str, path: str, header: str | None = None
):
    """Write a dictionary into a txt table with keys as column headers."""

    with open(path+file, "w") as f:

        # Write header for the file.
        if header is not None:
            f.write("# "+header.replace("\n", "\n# ")+"\n\n")

        # Write column headers.
        for key in d:

            # Get key unit.
            if key[1:6] == "Delta":
                unit_str = plotting.get_var_unit_str(var_descr=key[7:])
            else:
                unit_str = plotting.get_var_unit_str(var_descr=key)
            if unit_str != "":
                unit_str = f" /{unit_str}"

            f.write(f"{key+unit_str:<34}")
            if type(d[key]) == np.float64:
                len_col = 1
            else:
                len_col = len(d[key])
        f.write("\n")

        # Write rows.
        if len_col == 1:
            for key in d:
                f.write(f"{d[key]:<34}")

        else:
            for i_row in range(len_col):
                for key in d:
                    f.write(f"{d[key][i_row]:<34}")
                f.write("\n")

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
        if len(set([len(data), len(data_error), len(wavelength)])) != 1:
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
        self.weight_mode = weight_mode

        # Compute projected baseline length from u-v-coordinates.
        self.B = np.sqrt(ucoord**2 + vcoord**2)

        self.compute_spatial_frequency()

        self.set_weight()

    def compute_spatial_frequency(self):

        self.spatial_frequency = self.B / self.wavelength

    def set_weight(self):
        """
        Set the weights for each data point that can be used for fitting.

        Different types of weights of the data can be chosen that can be used
        in least-squares fitting.
        The options are:
          "no weights": All weights are the same and equal to one.
          "error": The errors of the data define the weights as
            1/error^2.
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
        """

        num_data_points = len(self.wavelength)
        weight_error = 1.0 / self.data_error**2

        if self.weight_mode == "no weights":

            self.weight = np.array([1.0 for i in range(num_data_points)])

        elif self.weight_mode == "error":

            self.weight = weight_error

        elif self.weight_mode == "points per baseline":

            self.weight = np.array(
                [1.0/num_data_points for i in range(num_data_points)]
            )

        elif self.weight_mode == "both":

            weight_per_baseline = 1.0 / num_data_points
            self.weight = weight_per_baseline * weight_error

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

    def __init__(
        self, oifits_file_ls: list[str], wave_min_ls: list[float],
        wave_max_ls: list[float], exclude_baselines_ls_ls: list[list[str]],
        path_to_data: str, fit_vis_or_vis2: str, weight_mode: str,
        unflag_all: bool = False
    ):
        """
        Read input, select wavelength and baseline and create a Full_data_set.

        Args:
            oifits_file_ls: List of the Oifits files to be loaded.
            wave_min_ls: List of the smallest wavelengths considered per file
              listed in oifits_file_ls. Thereby it is possible to select
              different smallest wavelengths for different files.
            wave_max_ls: Same as wave_min_ls, but for the largest wavelengths
              considered.
            exclude_baselines_ls_ls: Nested list that contains a lists of
              baselines to be excluded from the analysis for every file in
              oifits_file_ls. The alphabetical order of the stations in the
               baseline name does not matter, they are alphabetically ordered
               internally.
               Example: oifits_file_ls contains three files. We want to exclude
                the baselines 'A0J3' and 'A4K2' in the second file and 'J3G2'
                in the third file. Then set
                exclude_baseline_ls = [[], [A0J3, A4K2], [J3G2]]
            path_to_data: System path to where the Oifits files are.
            fit_vis_or_vis2: String of either "VISAMP" or "VIS2" to select
              treatment of visibilities (VISAMP) or squared visibilities
              (VIS2).
            weight_mode: String defining the type of weights. Choose from
              'no weights', 'error', 'points per baseline', or 'both'.

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
                but with different spectral dispersions. "both": Both 'error'
                and 'points per baseline' weights combined via multiplication.

        Raises:
            NoDataError: The chosen data, visibility or squared visibility, is
              not present in an input fits file.
        """

        file_data_set_ls =[]

        for (oifits_file, wave_min, wave_max, exclude_baselines_ls) in zip(
            oifits_file_ls, wave_min_ls, wave_max_ls, exclude_baselines_ls_ls
        ):

            oifits_obj = oifits.open(path_to_data+oifits_file)

            # Check whether the desired data, visibility or squared visibility,
            # is present in the chosen file. Raise error if not.
            if fit_vis_or_vis2 == "VISAMP":
                if len(oifits_obj.vis) == 0:
                    pass
                    raise exceptions.NoDataError(oifits_file, fit_vis_or_vis2)
            elif fit_vis_or_vis2 == "VIS2":
                if len(oifits_obj.vis2) == 0:
                    raise exceptions.NoDataError(oifits_file, fit_vis_or_vis2)

            # Unflag (=unmask) all values.
            if unflag_all:

                oifits_obj = unflag_all_wavelengths(
                    oifits_obj, fit_vis_or_vis2
                )

            # Flag wavelengths
            oifits_obj = flag_wavelengths(
                oifits_obj, wave_min, wave_max, fit_vis_or_vis2
            )

            # TODO:
            # Mask baselines using exclude_baselines_ls

            # Loop trough the different data sets of the baselines Oifits file.
            # Create a Baseline object for each set and put everything into an
            # All_Baselines_per_File object.
            # Different observations joint together in one file are not split.

            baseline_ls = []

            # Select VISAMP or VIS2
            if fit_vis_or_vis2 == "VISAMP":
                oifits_data = oifits_obj.vis
            elif fit_vis_or_vis2 == "VIS2":
                oifits_data = oifits_obj.vis2

            for oifits_baseline in oifits_data:

                # Get the name of the baseline. Join the two telescope stations
                # sorted alphabetically.
                station_ls = [oifits_baseline.station[0].sta_name,
                            oifits_baseline.station[1].sta_name]
                station_ls.sort()
                baseline_id = "".join(station_ls)

                # Select again on a deeper level VISAMP or VIS2.
                # Then take the non masked values from each baseline and put
                # them as array in oifits_data, oifits_error, and
                # oifits_wavelength. No masked arrays from this point on, only
                # the chosen data.
                if fit_vis_or_vis2 == "VISAMP":

                    oifits_data_values = oifits_baseline.visamp[
                        np.invert(oifits_baseline.flag)
                    ].data
                    oifits_data_err_values = oifits_baseline.visamperr[
                        np.invert(oifits_baseline.flag)
                    ].data

                elif fit_vis_or_vis2 == "VIS2":

                    oifits_data_values = oifits_baseline.vis2data[
                        np.invert(oifits_baseline.flag)
                    ].data
                    oifits_data_err_values = oifits_baseline.vis2err[
                        np.invert(oifits_baseline.flag)
                    ].data

                wavelength = oifits_baseline.wavelength.eff_wave[
                    np.invert(oifits_baseline.flag)
                ]

                ucoord = oifits_baseline.ucoord
                vcoord = oifits_baseline.vcoord

                baseline = Baseline(
                    baseline_id=baseline_id,
                    data=oifits_data_values,
                    data_error=oifits_data_err_values,
                    wavelength=wavelength,
                    ucoord=ucoord,
                    vcoord=vcoord,
                    weight_mode=weight_mode
                )
                baseline_ls.append(baseline)

            file_data_set_ls.append(
                All_Baselines_per_File(
                    file=oifits_file, baseline_ls=baseline_ls)
            )

        self.file_data_set_ls = file_data_set_ls

    def bin_wavelengths(
        self, bins: int | list[int],
        wave_bins_range: tuple[float, float] | list[tuple[float, float]] | None = None
     ):
        """
        Bin data according to wavelengths using the mean.

        If bins is an integer, the same number of bins are used for every
        data set. If bins is a list, the respective number of bins are used
        for the individual input files. In this case, the length of bins has to
        match the number of input files.

        If wave_bins_range is not provided, the number of bins are spread
        between [min(wavelength), max(wavelength)]. This minimum/maximum
        wavelength are computed from all files if bins is an integer and for
        each file individually if bins is a list.
        With wave_range provided, this defines the wavelength range. Either
        for all files together if wave_bins_range is a tuple or for each file
        individually if it is a list of tuples. In that case the list length
        has to match the number of input files.
        In case some wavelength bins lie outside the wavelength range of the
        data, those empty bins will not be saved.

        Args:
            bins: Can be integer giving the number bins to apply for all input
              files. Or list of integer giving the number of bins individually
              for each input file.
            wave_range: Optional, can be tuple of the form (wave_min, wave_max)
              that defines the wavelengths between the bins are spread. Or can
              be list of tuples of the described form.
        """

        N_file_data_set = len(self.file_data_set_ls)

        # Check input and generalize to lists.
        if type(bins) == int:
            bins = [bins for i in range(N_file_data_set)]
        elif type(bins) == list:
            if len(bins) != N_file_data_set:
                raise ValueError(
                    "Length of bins has to match number of input files.\n"
                    f"  Length of bins: {len(bins)}\n"
                    f"  Number of files: {N_file_data_set}\n"
                    "Alternatively, input bins as integer to use this number "
                    "for all files."
                )
                bins_ls = bins

        if wave_bins_range is None:
            wave_bins_range = [
                None for i in range(N_file_data_set)
            ]
        elif type(wave_bins_range) == tuple:
            wave_bins_range = [
                wave_bins_range
                for i in range(N_file_data_set)
            ]
        elif type(wave_bins_range) == list:
            if len(wave_bins_range) != N_file_data_set:
                raise ValueError(
                    "Length of wave_bins_range has to match number of input "
                    "files.\n"
                    f"  Length of wave_bins_range: {len(wave_bins_range)}\n"
                    f"  Number of files: {N_file_data_set}\n"
                    "Alternatively, input wave_bins_range as tuple to use this"
                    "number for all files."
                )

        for file_data_set, bins_data_set, wave_bins_range_data_set in zip(
            self.file_data_set_ls, bins, wave_bins_range
        ):

            for baseline in file_data_set.baseline_ls:

                # The np.flip is required to sort according to decreasing
                # wavelengths as the binned_statistics returns increasing
                # wavelengths.
                # NaNs result from empty bins. This happens if either to many
                # bins are selected and are thus finer than the data or the
                # wavelength range of the bins is larger than that of the data.

                data_binned = binned_statistic(
                    x=baseline.wavelength,
                    values=baseline.data,
                    bins=bins_data_set,
                    statistic="mean",
                    range=wave_bins_range_data_set
                ).statistic

                data_error_binned = binned_statistic(
                    x=baseline.wavelength,
                    values=baseline.data_error,
                    bins=bins_data_set,
                    statistic="mean",
                    range=wave_bins_range_data_set
                ).statistic

                # Wavelengths have to be binned last, as binning is done based
                # on them.

                wavelength_binned = binned_statistic(
                    x=baseline.wavelength,
                    values=baseline.wavelength,
                    bins=bins_data_set,
                    statistic="mean",
                    range=wave_bins_range_data_set
                ).statistic

                baseline.data = np.flip(
                    data_binned[~np.isnan(data_binned)]
                )
                baseline.data_error = np.flip(
                    data_error_binned[~np.isnan(data_error_binned)]
                )
                baseline.wavelength = np.flip(
                    wavelength_binned[~np.isnan(wavelength_binned)]
                )

                # Recompute spatial frequencies and weights.
                baseline.compute_spatial_frequency()
                baseline.set_weight()

    def get_all_data_flattened(self) -> tuple:
        """
        Return the data of all baselines as a tuple of flattened arrays.

        Returns:
            Returns object attribute arrays in a tuple of the form
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
        put into a list that is returned.
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

class Full_data_set_from_list(Full_data_set):
    """
    Full_data_set created by providing list of All_Baselines_per_File objects.
    """

    def __init__(self, file_data_set_ls: list):


        self.file_data_set_ls = file_data_set_ls

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
          via spatial_frequency = baseline/wavelength and have the unit
          [1/rad]. To get the common representation (e.g., used by the JMMC
          tool Oifitsexplorer) as 'Mega lambda' or '1e6/rad', one has to
          multiply with 1e-6.
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