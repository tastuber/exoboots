import numpy as np
import oifits
from scipy.stats import binned_statistic

from exoboots import exceptions
from exoboots import plotting

def comp_spfrq(u_spfrq, v_spfrq):
    """
    Compute the spatial frequency related to the baseline.

    Args:
        u_spfrq: The spatial frequencies along the u coordinate.
        v_spfrq: The spatial frequencies along the v coordinate.

    Returns:
        spfrq: The spatial frequencies along the baseline.
    """

    spfrq = (u_spfrq**2 + v_spfrq**2)**0.5

    return spfrq

def unflag_all_wavelengths(oifits_DU: oifits.oifits,
                           vis_or_vis2: str) -> oifits.oifits:
    """
    Unflag all measurements.

    Takes the oifits object and returns a modified copy of it. Modified are
    only the flags of the data. The oifits module creates the actual masked
    arrays with data on the fly making use of the flag attribute.

    Args:
        oifits_DU: The oifits data unit (DU) as instance of oifits.oifits.
        vis_or_vis2: String of either "VISAMP" or "VIS2" to select
          treatment of visibilities (VISAMP) or squared visibilities (VIS2).

    Returns:
        oifits_DU: Modified copy of the input oifits with flagged data.
    """

    if vis_or_vis2 == "VISAMP":

        for vis in oifits_DU.vis:

            vis.flag = vis.flag * False
            vis.flag = vis.flag * False

    elif vis_or_vis2 == "VIS2":

        for vis2 in oifits_DU.vis2:

            vis2.flag = vis2.flag * False
            vis2.flag = vis2.flag * False

    return oifits_DU

def flag_wavelengths(oifits_DU: oifits.oifits, min_wave: float,
                     max_wave:float, vis_or_vis2: str) -> oifits.oifits:
    """
    Masks all measurements outside the chosen wavelength range.

    Takes the oifits object and returns a modified copy of it. Modified are
    only the flags of the data. The oifits module creates the actual masked
    arrays with data on the fly making use of the flag attribute.

    Args:
        oifits_DU: The oifits data unit (DU) as instance of oifits.oifits.
        min_wave: The minimum wavelength in units of meter to be considered in
          the analysis. All data corresponding to smaller wavelengths are
          flagged.
        max_wave: The maximum wavelength in units of meter to be considered in
          the analysis. All data corresponding to larger wavelengths are
          flagged.
        vis_or_vis2: String of either "VISAMP" or "VIS2" to select
          treatment of visibilities (VISAMP) or squared visibilities (VIS2).

    Returns:
        oifits_DU: Modified copy of the input oifits with flagged data.
    """

    if vis_or_vis2 == "VISAMP":

        for vis in oifits_DU.vis:

            wave_mask = vis.wavelength.eff_wave < min_wave
            vis.flag[wave_mask] = True
            wave_mask = vis.wavelength.eff_wave > max_wave
            vis.flag[wave_mask] = True

    elif vis_or_vis2 == "VIS2":

        for vis2 in oifits_DU.vis2:

            wave_mask = vis2.wavelength.eff_wave < min_wave
            vis2.flag[wave_mask] = True
            wave_mask = vis2.wavelength.eff_wave > max_wave
            vis2.flag[wave_mask] = True

    return oifits_DU

def sort_station_names(exclude_baselines_per_file: list[list[str]]):
    """
    Sort the two station names making up a baseline ID in alphabetical order

    For instance, is the baseline id J4A0, it is changed into A0J4.

    Args:
        exclude_baselines_per_file: Nested list containining for each oifits file
          the baselines to be flagged.
    """

    for i, exclude_baselines in enumerate(exclude_baselines_per_file):

        for j, baseline_id in enumerate(exclude_baselines):

            ls = [baseline_id[:2], baseline_id[2:]]
            ls.sort()
            new_baseline_id = "".join(ls)
            exclude_baselines_per_file[i][j] = new_baseline_id

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
        spfrq: The measured spatial frequencies. They are computed
          via spatial frequency = baseline/wavelength and have the unit
          [1/rad]. To get the common representation (e.g., used by the JMMC
          tool Oifitsexplorer) as 'Mega lambda' or '1e6/rad', one has to
          multiply with 1e-6.
        u_spfrq: Same as spfrq, but only along the
          u-axis.
        v_spfrq: Same as spfrq, but only along the
          v-axis.
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

        # Compute projected baseline length from u-v-coordinates.
        self.B = np.sqrt(ucoord**2 + vcoord**2)

        self.compute_spfrq()

    def compute_spfrq(self):

        self.spfrq = self.B / self.wavelength
        self.u_spfrq = self.ucoord / self.wavelength
        self.v_spfrq = self.vcoord / self.wavelength

    def set_weight(self, weight_mode: str):
        """
        Set the weights for each data point that can be used for fitting.

        Different types of weights of the data can be chosen that can be used
        in least-squares fitting.

        Args:
            weight_mode: Defines how weights are computed. The options are:
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

        num_data_points = len(self.wavelength)
        weight_error = 1.0 / self.data_error**2

        if weight_mode == "no weights":

            self.weight = np.array([1.0 for i in range(num_data_points)])

        elif weight_mode == "error":

            self.weight = weight_error

        elif weight_mode == "points per baseline":

            self.weight = np.array(
                [1.0/num_data_points for i in range(num_data_points)]
            )

        elif weight_mode == "both":

            weight_per_baseline = 1.0 / num_data_points
            self.weight = weight_per_baseline * weight_error

        else:

            raise ValueError(
                "Wrong 'mode' to set weights. Choose from either 'no weights',"
                "'error', 'points per baseline', or 'both'."
            )

class All_Baselines_per_File():
    """Contains all Baseline objects of one Oifits file."""

    def __init__(self, file: str, baselines: list):

        self.file = file
        self.baselines = baselines

    def __str__(self):

        return f"{len(self.baselines)} baselines from {self.file}."

class Full_data_set():
    """
    Contains the full data set for the fit procedure.

    Attributes:
      file_data_sets: List of All_Baselines_per_File objects.
    """

    def __init__(
        self, oifits_files: str | list[str], path_to_data: str,
        vis_or_vis2: str,
        waves: list[tuple[float, float]] = [(-np.inf, np.inf)],
        exclude_baselines_per_file: list[list[str]] | None = None,
        unflag_all: bool = False
    ):
        """
        Read input, select wavelength and baseline and create a Full_data_set.

        Args:
            oifits_files: Oifits file or list files to be loaded.
            waves: List of the wavelength intervals in the form
              tuple(min_wave, max_wave) considered per file listed in
              oifits_files. Thereby it is possible to select different smallest
              wavelengths for different files. If only one interval is given,
              it is applied to all files. The default is [(-np.inf, np.inf)],
              thus selecting all wavelengths.
            path_to_data: System path to where the Oifits files are.
            vis_or_vis2: String of either "VISAMP" or "VIS2" to select
              treatment of visibilities (VISAMP) or squared visibilities
              (VIS2).
            exclude_baselines_per_file: Nested list that contains a lists of
              baselines to be excluded from the analysis for every file in
              oifits_files. The alphabetical order of the stations in the
               baseline name does not matter, they are alphabetically ordered
               internally.
               Example: oifits_files contains three files. We want to exclude
                the baselines 'A0J3' and 'A4K2' in the second file and 'J3G2'
                in the third file. Then set
                exclude_baselines = [[], [A0J3, A4K2], [J3G2]].
            unflag_all: Set to True to unflag all data before flagging again to
              select the wavelengths interval. Use if data has been
              accidentally flagged in the Oifits file during file creation.

        Raises:
            NoDataError: The chosen data, visibility or squared visibility, is
              not present in an input fits file.
        """

        self.vis_or_vis2 = vis_or_vis2

        file_data_sets = []

        # If oifits_files containes a single string, wrap it in a list.
        if not isinstance(oifits_files, list):

            oifits_files = [oifits_files]

        # Handle multiple lengths of waves.
        if len(oifits_files) != len(waves):

            if len(waves) == 1:
                waves = len(oifits_files) * waves
            else:
                raise ValueError(
                    "The length of waves has to be either 1 (the same "
                    "wavelength range is applied to all oifits files) of has "
                    "to match the length of oifits_files (individual "
                    "wavelengths are applied)."
                )

        # Create empty lists in case no baselines shall be excluded.
        if exclude_baselines_per_file is None:
            exclude_baselines_per_file = [
                [] for i in range(len(oifits_files))
            ]
        # Else sort the telescope pairs making up the baseline name in
        # alphabetical order.
        else:
            sort_station_names(exclude_baselines_per_file)

        for (oifits_file, wave, exclude_baselines) in zip(
            oifits_files, waves, exclude_baselines_per_file
        ):

            oifits_DU = oifits.open(path_to_data+oifits_file)

            # Check whether the desired data, visibility or squared visibility,
            # is present in the chosen file. Raise error if not.
            if self.vis_or_vis2 == "VISAMP":
                if len(oifits_DU.vis) == 0:
                    pass
                    raise exceptions.NoDataError(oifits_file, self.vis_or_vis2)
            elif self.vis_or_vis2 == "VIS2":
                if len(oifits_DU.vis2) == 0:
                    raise exceptions.NoDataError(oifits_file, self.vis_or_vis2)

            # Unflag (=unmask) all values.
            if unflag_all:

                oifits_DU = unflag_all_wavelengths(
                    oifits_DU, self.vis_or_vis2
                )

            # Flag wavelengths
            oifits_DU = flag_wavelengths(
                oifits_DU=oifits_DU,
                min_wave=wave[0],
                max_wave=wave[1],
                vis_or_vis2=self.vis_or_vis2
            )

            # Loop trough the different data sets of the baselines Oifits file.
            # Create a Baseline object for each set and put everything into an
            # All_Baselines_per_File object.
            # Different observations joint together in one file are not split.

            baselines = []

            # Select VISAMP or VIS2
            if self.vis_or_vis2 == "VISAMP":
                oi_data = oifits_DU.vis
            elif self.vis_or_vis2 == "VIS2":
                oi_data = oifits_DU.vis2

            for oifits_baseline in oi_data:

                # Get the name of the baseline. Join the two telescope stations
                # sorted alphabetically.
                stations = [oifits_baseline.station[0].sta_name,
                              oifits_baseline.station[1].sta_name]
                stations.sort()
                baseline_id = "".join(stations)

                # Skip baseline if it shall be excluded from analysis.
                if len(exclude_baselines) != 0:

                    if baseline_id in exclude_baselines:

                        print(f"Exclude baseline: {baseline_id}")
                        continue

                # Select again on a deeper level VISAMP or VIS2.
                # Then take the non masked values from each baseline and put
                # them as array in oi_data, oifits_error, and
                # oifits_wavelength. No masked arrays from this point on, only
                # the chosen data.
                if self.vis_or_vis2 == "VISAMP":

                    oi_data_values = oifits_baseline.visamp[
                        np.invert(oifits_baseline.flag)
                    ].data
                    oi_data_err_values = oifits_baseline.visamperr[
                        np.invert(oifits_baseline.flag)
                    ].data

                elif self.vis_or_vis2 == "VIS2":

                    oi_data_values = oifits_baseline.vis2data[
                        np.invert(oifits_baseline.flag)
                    ].data
                    oi_data_err_values = oifits_baseline.vis2err[
                        np.invert(oifits_baseline.flag)
                    ].data

                wavelength = oifits_baseline.wavelength.eff_wave[
                    np.invert(oifits_baseline.flag)
                ]

                ucoord = oifits_baseline.ucoord
                vcoord = oifits_baseline.vcoord

                baseline = Baseline(
                    baseline_id=baseline_id,
                    data=oi_data_values,
                    data_error=oi_data_err_values,
                    wavelength=wavelength,
                    ucoord=ucoord,
                    vcoord=vcoord,
                )
                baselines.append(baseline)

            file_data_sets.append(
                All_Baselines_per_File(
                    file=oifits_file, baselines=baselines)
            )

        self.file_data_sets = file_data_sets

    def set_weight(self, weight_mode: str):
        """
        Set the fit weights for all baselines.

        Options are: "no weights", "error", "points per baseline", or "both".
        See for more information Baseline.set_weight().
        """

        # Cycle through all Baselines.
        for file_data_set in self.file_data_sets:

            for baseline in file_data_set.baselines:

                baseline.set_weight(weight_mode=weight_mode)


    def bin_wavelengths(
        self, bins: int | list[int],
        wave_bins_range: tuple[float, float] | list[tuple[float, float]] \
            | None = None
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
            wave_range: Optional, can be tuple of the form (min_wave, max_wave)
              that defines the wavelengths between the bins are spread. Or can
              be list of tuples of the described form.
        """

        N_file_data_set = len(self.file_data_sets)

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
            self.file_data_sets, bins, wave_bins_range
        ):

            for baseline in file_data_set.baselines:

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

                # Recompute spatial frequencies.
                baseline.compute_spfrq()

    def get_all_data_flattened(self) -> tuple:
        """
        Return the data of all baselines as a tuple of flattened arrays.

        Returns:
            Returns object attribute arrays in a tuple of the form
            (data, data_error, wavelength, u_spfrq,
            v_spfrq, weight).
        """

        data_per_baseline = []
        data_error_per_baseline = []
        wavelengths_per_baseline = []
        u_spfrq_per_baseline = []
        v_spfrq_per_baseline = []
        weight_per_baseline = []

        for file_data_set in self.file_data_sets:

            for baseline in file_data_set.baselines:

                data_per_baseline.extend(baseline.data)
                data_error_per_baseline.extend(baseline.data_error)
                wavelengths_per_baseline.extend(baseline.wavelength)
                u_spfrq_per_baseline.extend(baseline.u_spfrq)
                v_spfrq_per_baseline.extend(baseline.v_spfrq)
                weight_per_baseline.extend(baseline.weight)

        data = np.asarray(data_per_baseline).flatten()
        data_error = np.asarray(data_error_per_baseline).flatten()
        wavelength = np.asarray(wavelengths_per_baseline).flatten()
        u_spfrq = np.asarray(u_spfrq_per_baseline).flatten()
        v_spfrq = np.asarray(v_spfrq_per_baseline).flatten()
        weight = np.asarray(weight_per_baseline).flatten()

        return (
            data, data_error, wavelength, u_spfrq,
            v_spfrq, weight
        )

    def get_all_baselines(self) -> list:
        """
        Returns all baselines of the data set that has not been excluded.

        Returns:
            baselines: All baselines of all loaded Oifits files that
              have not been excluded from the process.
        """

        all_baselines = []
        for file_data_set in self.file_data_sets:

            for baseline in file_data_set.baselines:

                all_baselines.append(baseline)

        return all_baselines

    def get_data_per_wavelength(self) -> list:
        """
        Returns a list of Data_for_one_wavelength objects.

        Use this function only when analyzing Oifits with the exact same
        wavelength grid or a single file.
        Collects for each individual wavlength all associated data and puts it
        into a Data_for_one_wavelength object. All Data_for_one_wavelength objects are
        put into a list that is returned.
        The length of the output list is equal to the amount of individual
        wavelengths.

        Returns:
            data_per_wavelength: List of Data_for_one_wavelength objects.
        """

        # Check whether all Baseline objects contain the exact same wavelength
        # grid. If not, raise an exception.
        error_message = (
            "The wavelength grids of two baselines do not match. This is "
            "required for the function get_data_per_wavelength() to be "
            "reasonable.\nProbably the data of two Oifits files with "
            "different wavelength grids are loaded."
        )
        wavelength = self.file_data_sets[0].baselines[0].wavelength
        for file_data_set in self.file_data_sets:

            for baseline in file_data_set.baselines:

                if len(wavelength) != len(baseline.wavelength):
                    raise ValueError(error_message)

                elif any (baseline.wavelength != wavelength):
                    raise ValueError(error_message)

        # Sort the data after wavelength.
        data_per_wavelength = []
        all_baselines = self.get_all_baselines()
        N_baselines = len(all_baselines)

        for i_wave, single_wavelength in enumerate(wavelength):

            data = np.zeros(N_baselines)
            data_error = np.zeros(N_baselines)
            spfrq = np.zeros(N_baselines)
            u_spfrq = np.zeros(N_baselines)
            v_spfrq = np.zeros(N_baselines)
            weight = np.zeros(N_baselines)
            ucoord = np.zeros(N_baselines)
            vcoord = np.zeros(N_baselines)
            B = np.zeros(N_baselines)
            baseline_ids = []

            for i_baseline, baseline in enumerate(all_baselines):

                data[i_baseline] = baseline.data[i_wave]
                data_error[i_baseline] = baseline.data_error[i_wave]
                spfrq[i_baseline] = \
                    baseline.spfrq[i_wave]
                u_spfrq[i_baseline] = \
                    baseline.u_spfrq[i_wave]
                v_spfrq[i_baseline] = \
                    baseline.v_spfrq[i_wave]
                weight[i_baseline] = baseline.weight[i_wave]
                ucoord[i_baseline] = baseline.ucoord
                vcoord[i_baseline] = baseline.vcoord
                B[i_baseline] = baseline.B
                baseline_ids.append(baseline.baseline_id)

            data_per_wavelength.append(
                Data_for_one_wavelength(
                    wavelength=single_wavelength,
                    data=data,
                    data_error=data_error,
                    spfrq=spfrq,
                    u_spfrq=u_spfrq,
                    v_spfrq=v_spfrq,
                    weight=weight,
                    ucoord=ucoord,
                    vcoord=vcoord,
                    B=B,
                    baseline_ids=baseline_ids
                )
            )

        return data_per_wavelength

class Full_data_set_from_list(Full_data_set):
    """
    Full_data_set created by providing list of All_Baselines_per_File objects.
    """

    def __init__(self, file_data_sets: list):


        self.file_data_sets = file_data_sets

class Data_for_one_wavelength():
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
        spfrq: The measured spatial frequencies along the baseline.
          They are computed via spatial frequency = baseline/wavelength and
          have the unit [1/rad]. To get the common representation (e.g., used
          by the JMMC tool Oifitsexplorer) as 'Mega lambda' or '1e6/rad', one
          has to multiply with 1e-6.
        u_spfrq: Spatial frequency along u axis in units of 1/rad.
        v_spfrq: Spatial frequency along v axis in units of 1/rad.
        ucoord: The u-coordinate in units of meter [m].
        vcoord: The v-coordinate in units of meter [m].
        B: The projected baseline length in units of meter [m].
    """

    def __init__(
        self, wavelength, data, data_error, spfrq,
        u_spfrq, v_spfrq, weight, ucoord, vcoord, B,
        baseline_ids
    ):

        self.wavelength = wavelength
        self.data = data
        self.data_error = data_error
        self.spfrq = spfrq
        self.u_spfrq = u_spfrq
        self.v_spfrq = v_spfrq
        self.weight = weight
        self.ucoord = ucoord
        self.vcoord = vcoord
        self.B = B
        self.baseline_ids = baseline_ids

    def __str__(self):

        return (f"{len(self.data)} data points for the wavelength "
                f"{np.round(self.wavelength*1e6, 4)} micron")
