"""
==========
Utilities
==========

 Utilities functions for performing data preprocessing

 List of functions:
   - bunch_to_df : Turning bunch objects into pandas DataFrame
   - create_folds : Creating fine grained accessible DataFrame with `kfold` column

 Dependencies:
   - numpy
   - pandas
   - sklearn

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import Bunch


def bunch_to_df(bunch):
    """
    Transforming a scikit-learn Bunch object to pandas DataFrame. Bunch objects are generally returned
    by load_dataset methods of scikit-learn.

    Parameters
    ----------
    bunch : sklearn.utils.Bunch
        Bunch object of scikit-learn package
    """

    if not isinstance(bunch, Bunch):
        raise Exception("Not a Bunch Object")

    if len(bunch.target.shape) > 1 and bunch.target.shape[1] > 1:
        df = pd.DataFrame(data=np.c_[bunch.data, bunch.target],
                          columns=list(bunch.feature_names) + bunch.target_names)
    else:
        df = pd.DataFrame(data=np.c_[bunch.data, bunch.target],
                          columns=list(bunch.feature_names) + ["target"])
    return df


def create_folds(data, target_name, binning=False):
    """
    Description: Creating uniform or stratified folds for cross validation.
    Contrary to sklearn methods, this approach gives fine grained control to
    individual folds. Returned dataframe includes an extra feature column named as 'kfold'.

    Parameters:
    ----------
    data : pandas DataFrame object
    target_name: feature used for stratified fold creating
    """



    data["kfold"] = -1

    if binning:
        num_bins = int(np.floor(1 +np.log2(len(data))))
        data.loc[:,"bins"] = pd.cut(data[target_name], bins=num_bins, labels=False)
        target_name = 'bins'

    kStratFold = StratifiedKFold(n_splits=5)

    for fold, (train_, val_) in enumerate(kStratFold.split(X=data, y=data[target_name].values)):
        data.loc[val_, "kfold"] = fold
    if binning:
        data.drop("bins", axis=1, inplace=True)
    return data




if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.datasets import *
    import pandas as pd
    import numpy as np

# Testing bunch_to_df function for scikit-learn load functions
    for item in dir(datasets):
        if item[0:4] == "load" and item != "load_files" \
        and item != "load_sample_image" and item != "load_sample_images" \
        and item != 'load_svmlight_file' and item !='load_svmlight_files':
            a = item + "()"
            print(a)
            df = eval(a)
            df = bunch_to_df(df)
            df.info()
