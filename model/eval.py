from pathlib import Path

import click
import pweave


@click.command()
@click.option("--out-dir")
def generate_report(out_dir):
	"""
	Generates Pweave report
	:param out_dir: output directory where the HTML will be stored
	"""
	pweave.publish("eval.pmd", output=f"{out_dir}/eval_output.html")
	flag = Path(out_dir) / '.SUCCESS'
	flag.touch()


if __name__ == '__main__':
	generate_report()
