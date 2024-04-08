import gradio as gr
import os
import sys

if os.environ.get("DEV_MODE"):
    # for local development
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))
from fsrs_optimizer import Optimizer, DEFAULT_WEIGHT


def interface_func(weights: str, ratings: str, request_retention: float) -> str:
    weights = weights.replace("[", "").replace("]", "")
    optimizer = Optimizer()
    optimizer.w = list(map(lambda x: float(x.strip()), weights.split(",")))
    test_sequence = optimizer.preview_sequence(
        ratings.replace(" ", ""), request_retention
    )
    default_preview = optimizer.preview(request_retention)
    return test_sequence, default_preview


iface = gr.Interface(
    fn=interface_func,
    inputs=[
        gr.inputs.Textbox(
            label="weights",
            lines=1,
            default=str(DEFAULT_WEIGHT)[1:-1],
        ),
        gr.inputs.Textbox(label="ratings", lines=1, default="3,3,3,3,1,3,3"),
        gr.inputs.Slider(
            label="Your Request Retention",
            minimum=0.6,
            maximum=0.97,
            step=0.01,
            default=0.9,
        ),
    ],
    outputs=[
        gr.outputs.Textbox(label="test sequences"),
        gr.outputs.Textbox(label="default preview"),
    ],
)

iface.launch()
