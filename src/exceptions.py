class DataValidationError(Exception):
    """
    Raised when dataset validation fails.
    """

    pass


class MissingColumnError(Exception):
    """
    Raised when required columns are missing.
    """

    pass
