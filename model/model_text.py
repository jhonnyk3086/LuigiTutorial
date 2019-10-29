import click
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.utils._joblib import dump
from lightgbm import LGBMRegressor


def _save_model(model, vectorizer, outdir: Path):
	"""
	Saves model and vectorizer offline to load it later
	:param model: model to save
	:param vectorizer: vectorizer to save
	:param outdir: directory where it'll be saved
	"""
	dump(model, str(outdir / "model.pth"))
	dump(vectorizer, str(outdir / "vectorizer.pth"))
	flag = outdir / '.SUCCESS'
	flag.touch()


def get_model(model_name):
	"""
	Fetch the right model from the model zoo
	:param model_name: name of the model
	:return: madel instance
	"""
	return {
		"XGBoost":          XGBRegressor(objective='reg:squarederror'),
		"SVR":              SVR(),
		"LinearRegression": LinearRegression(),
		"LightGBM":         LGBMRegressor(),
	}[model_name]


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
	:param model_name: name of the model
	"""
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	train_x = f'{in_dir}/X_train.csv'
	train_y = f"{in_dir}/y_train.csv"
	X_train = pd.read_csv(train_x)
	y_train = pd.read_csv(train_y, header=None)
	description = X_train['description']
	vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3),
								 max_df=0.5, min_df=0.1)
	X_train_vectorized = vectorizer.fit_transform(description)
	model = get_model(model_name)
	model.fit(X_train_vectorized, y_train)
	_save_model(model, vectorizer, out_dir)


if __name__ == '__main__':
	train()

