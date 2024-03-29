import gradio as gr
import os
import sys

if os.environ.get("DEV_MODE"):
    # for local development
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))
from fsrs_optimizer import Optimizer


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
            default="0.4, 0.9, 2.3, 10.9, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61",
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
