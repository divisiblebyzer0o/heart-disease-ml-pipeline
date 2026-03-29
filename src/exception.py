class ModelNotTrained(Exception):
    pass

class DataValidationError(ValueError):
    pass

class FeatureMismatchError(DataValidationError):
    pass 