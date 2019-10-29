import click
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.utils._joblib import dump


def _save_model(model, outdir: Path):
	"""
	Saves model and vectorizer offline to load it later
	:param model: model to save
	:param outdir: directory where it'll be saved
	"""
	dump(model, str(outdir / "model.pth"))
	flag = outdir / '.SUCCESS'
	flag.touch()


def drop_cols(data):
	"""
	Removes columns with low importances
	:param data: input dataframe
	:return: dataframe after dropping columns
	"""
	cols = ["description", "title", "region_1"]
	for col in cols:
		data = data.drop(col, axis=1)
	return data


@click.command()
@click.option('--in-dir')
@click.option("--out-dir")
@click.option("--model-name")
def train(in_dir, out_dir, model_name):
	"""
	Trains the `model_name` model and saves it in
	the `out_dir` directory and sets the .SUCCESS
	flag
	:param in_dir: directory where the input data is stored
	:param out_dir: directory where the models will be saved
	:param model_name: name of the model (not required)
	"""
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	train_x = f'{in_dir}/X_train.csv'
	test_x = f'{in_dir}/X_test.csv'
	train_y = f"{in_dir}/y_train.csv"
	test_y = f"{in_dir}/y_test.csv"
	X_train = pd.read_csv(train_x)
	X_train = drop_cols(X_train)
	y_train = pd.read_csv(train_y, header=None)
	X_test = pd.read_csv(test_x)
	X_test = drop_cols(X_test)
	y_test = pd.read_csv(test_y, header=None)
	# replace NaNs by -1
	X_train = X_train.fillna(-1)
	X_test = X_test.fillna(-1)
	categorical_columns = ['country', 'designation', 'province',
						   'region_2', 'taster_name', 'taster_twitter_handle',
						   'variety', 'winery']
	model = CatBoostRegressor(loss_function='RMSE', iterations=400)
	model.fit(X_train, y_train, cat_features=categorical_columns,
			  eval_set=(X_test, y_test))
	_save_model(model, out_dir)


if __name__ == '__main__':
	train()
