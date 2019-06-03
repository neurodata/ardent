from ardent.preprocessing import preprocess

def basic_preprocessing(data):
    """Call each of the listed preprocessing functions on data and return the result."""

    return preprocess(data, ['normalize_by_MAD', 'center_to_mean', 'pad'])

