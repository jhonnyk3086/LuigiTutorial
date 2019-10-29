import click
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def _save_datasets(train_x, train_y, test_x, test_y, outdir: Path):
    """
    Save data sets into nice directory structure and write SUCCESS flag.
    :param train_x: numpy array of training independent vars
    :param train_y: numpy array of training dependent var
    :param test_x: numpy array of testing independent vars
    :param test_y: numpy array of testing dependent vars
    :param outdir: directory where the dataset will be stored
    """
    out_train_x = outdir / 'X_train.csv'
    out_test_x = outdir / 'X_test.csv'
    out_train_y = outdir / "y_train.csv"
    out_test_y = outdir / "y_test.csv"
    flag = outdir / '.SUCCESS'
    train_x.to_csv(str(out_train_x), index=False)
    train_y.to_csv(str(out_train_y), index=False, header=False)
    test_x.to_csv(str(out_test_x), index=False)
    test_y.to_csv(str(out_test_y), index=False, header=False)
    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    """
    Splits training and testing data while stratifying it to preserve
    the distribution. The stratifying strategy is to label the dependent
    variable into 5 classes eg: class 1 for [80-85] class 2 for [86-90]
    class 3 for [91-95] etc. This will ensure the distribution of the test
    data is approximately same as that in the train

    :param in_csv: path for the wine dataset
    :param out_dir: directory where the dataset will be saved
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_csv).drop("Unnamed: 0", axis=1)
    # drop duplicates and nas
    df = df[~df.duplicated(subset=["description", "points"], keep="first")]
    df = df.dropna(subset=["description", "points"])
    X_train, X_test, y_train, y_test = \
        train_test_split(df.drop("points", axis=1),
                         df["points"], stratify=pd.cut(df["points"], bins=5,
                                                       labels=[1, 2, 3, 4, 5]),
                         random_state=42)
    _save_datasets(X_train, y_train, X_test, y_test, out_dir)


if __name__ == '__main__':
    make_datasets()
