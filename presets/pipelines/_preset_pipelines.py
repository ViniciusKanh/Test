import importlib


def load_pipeline(pipeline_name):
    try:
        module = importlib.import_module(f'presets.pipelines.data.{pipeline_name}')
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Pipeline {pipeline_name} not found")
    return module.pipeline
