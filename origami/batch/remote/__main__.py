import click
import json
import logging
import importlib
import origami.batch.remote.runner

cli = click.Group()

@cli.command()
@click.argument(
	'config_path',
	type=click.Path(exists=True),
	required=True)
@click.pass_context
def run(ctx, config_path):
	with open(config_path, "r") as f:
		config = json.loads(f.read())

	processors = []

	for task in config["tasks"]:
		package = task["package"]
		task_module = importlib.import_module(package)
		make_processor = task_module.make_processor
		processors.append(ctx.invoke(make_processor, **task["args"]))

	origami.batch.remote.runner.run_on_remote_data(config, processors)


if __name__ == "__main__":
	"""
	python -m origami.batch.remote run /path/to/config.json
	"""

	logging.basicConfig()
	logging.getLogger().setLevel(logging.INFO)

	cli()
