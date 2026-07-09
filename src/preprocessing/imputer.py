from sklearn.impute import SimpleImputer


def numeric_imputer():
    """
    Median imputer
    """

    return SimpleImputer(strategy="median")


def categorical_imputer():
    """
    Most frequent category.
    """

    return SimpleImputer(strategy="most_frequent")
