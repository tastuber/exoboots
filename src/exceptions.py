class NoDataError(Exception):
    """
    Exception raised in case the demanded VISAMP or VIS2 data is not present.

    Attributes:
        oifits_file: Filename of the Oifits file that has data missing.
        fit_vis_or_vis2: String of either "VISAMP" or "VIS2", selects the
          missing data to be either visibility amplitude (VISAMP) or squared
          visibilities (VIS2).
        message: The error message.
    """

    def __init__(self, oifits_file: str, fit_vis_or_vis2: str):

        self.oifits_file = oifits_file
        self.fit_vis_or_vis2 = fit_vis_or_vis2

        if fit_vis_or_vis2=="VISAMP":
            data_str = "visibility amplitude (VISAMP)"

        elif fit_vis_or_vis2=="VIS2":
            data_str = "squared visibility (VIS2)"

        self.message = (f"The file {oifits_file} contains no {data_str} data.")

    def __str__(self):
        return repr(self.message)