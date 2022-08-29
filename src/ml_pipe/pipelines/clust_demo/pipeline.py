"""
This is a boilerplate pipeline 'clust_demo'
generated using Kedro 0.18.2
"""

import warnings

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import first

def create_pipeline(**kwargs) -> Pipeline:
    warnings.filterwarnings('ignore')

    return pipeline([
        node(first, inputs = None, outputs = 'res')
    ])
