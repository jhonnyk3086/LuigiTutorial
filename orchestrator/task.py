import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return 'code-challenge/download-data:0.1'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    @property
    def dataset_path(self):
        return str(f'{self.out_dir}/{self.fname}.csv')

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):
    """Splits data into train and test using stratified split"""

    out_dir = luigi.Parameter("/usr/share/data/processed/")

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        dataset_path = DownloadData().dataset_path
        return ['python', 'dataset.py',
                '--in-csv', dataset_path,
                '--out-dir', self.out_dir]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class XGBoost(DockerTask):
    """Trains XGBoost model on the description text"""

    model = "XGBoost"
    out_dir = luigi.Parameter(f"/usr/share/data/model/{model}/")
    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        dataset_dir = MakeDatasets().out_dir
        return ['python', 'model_text.py',
                '--in-dir', dataset_dir,
                '--out-dir', self.out_dir,
                '--model-name', self.model]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class LightGBM(DockerTask):
    """Trains XGBoost model on the description text"""

    model = "LightGBM"
    out_dir = luigi.Parameter(f"/usr/share/data/model/{model}/")
    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        dataset_dir = MakeDatasets().out_dir
        return ['python', 'model_text.py',
                '--in-dir', dataset_dir,
                '--out-dir', self.out_dir,
                '--model-name', self.model]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class SVR(DockerTask):
    """Trains XGBoost model on the description text"""

    model = "SVR"
    out_dir = luigi.Parameter(f"/usr/share/data/model/{model}/")
    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        dataset_dir = MakeDatasets().out_dir
        return ['python', 'model_text.py',
                '--in-dir', dataset_dir,
                '--out-dir', self.out_dir,
                '--model-name', self.model]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class SimpleLinearRegression(DockerTask):
    """Trains XGBoost model on the description text"""

    model = "LinearRegression"
    out_dir = luigi.Parameter(f"/usr/share/data/model/{model}/")
    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        dataset_dir = MakeDatasets().out_dir
        return ['python', 'model_text.py',
                '--in-dir', dataset_dir,
                '--out-dir', self.out_dir,
                '--model-name', self.model]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class Catboost(DockerTask):
    """Trains Catboost model on the entire dataset"""

    model = "Catboost"
    out_dir = luigi.Parameter(f"/usr/share/data/model/{model}/")
    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        dataset_dir = MakeDatasets().out_dir
        return ['python', 'model_features.py',
                '--in-dir', dataset_dir,
                '--out-dir', self.out_dir,
                '--model-name', self.model]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class TrainModels(luigi.Task):
    """Trains all the models above"""

    out_dir = luigi.Parameter("/usr/share/data/model/")

    def requires(self):
        return XGBoost(), SVR(), SimpleLinearRegression(), LightGBM(), \
               Catboost()

    def run(self):
        print("Running all models")
        self.output().makedirs()
        self.output().open("w").close()

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class Report(DockerTask):
    """Creates Pweave report after training all the models"""
    out_dir = luigi.Parameter("/usr/share/data/report/")

    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return TrainModels()

    @property
    def command(self):
        return ['python', 'eval.py',
                '--out-dir', self.out_dir]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class EvaluateModel(luigi.Task):
    """dummy task to track report generation from pweave"""
    out_dir = luigi.Parameter("/usr/share/data/evaluate/")

    def requires(self):
        return Report()

    def run(self):
        self.output().makedirs()
        self.output().open("w").close()

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )
