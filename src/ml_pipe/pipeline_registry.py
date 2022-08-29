"""Project pipelines."""
from typing import Dict
import warnings
from kedro.pipeline import Pipeline, pipeline
from .pipelines.clust_demo import pipeline as clust_pipe

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    warnings.filterwarnings('ignore')

    return {
        "demo": clust_pipe.create_pipeline()
        #"__default__": pipeline([])
    }
