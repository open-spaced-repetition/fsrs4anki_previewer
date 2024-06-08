from typing import List, Tuple
import gradio as gr
import os
import sys

if os.environ.get("DEV_MODE"):
    # for local development
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))
from fsrs_optimizer import Optimizer, DEFAULT_WEIGHT, FSRS, lineToTensor


def interface_func(
    weights: str, ratings: str, delta_ts: str, request_retention: float
) -> str:
    weights = weights.replace("[", "").replace("]", "")
    optimizer = Optimizer()
    optimizer.w = list(map(lambda x: float(x.strip()), weights.split(",")))
    test_sequence = optimizer.preview_sequence(
        ratings.replace(" ", ""), request_retention
    )
    default_preview = optimizer.preview(request_retention)
    if delta_ts != "":
        s_history, d_history = memory_state_sequence(ratings, delta_ts, optimizer.w)
        return (
            test_sequence,
            default_preview,
            f"s: {(', '.join(s_history))}\nd: {', '.join(d_history)}",
        )
    return test_sequence, default_preview, ""


def memory_state_sequence(
    r_history: str, t_history: str, weights: List[float]
) -> Tuple[List[float], List[float]]:
    fsrs = FSRS(weights)
    line_tensor = lineToTensor(list(zip([t_history], [r_history]))[0]).unsqueeze(1)
    outputs, _ = fsrs(line_tensor)
    stabilities, difficulties = outputs.transpose(0, 1)[0].transpose(0, 1)
    return map(lambda x: str(round(x, 2)), stabilities.tolist()), map(
        lambda x: str(round(x, 2)), difficulties.tolist()
    )


iface = gr.Interface(
    fn=interface_func,
    inputs=[
        gr.Textbox(
            label="weights",
            lines=1,
            value=str(DEFAULT_WEIGHT)[1:-1],
        ),
        gr.Textbox(label="ratings", lines=1, value="3,3,3,3,1,3,3"),
        gr.Textbox(label="delta_ts (requried by state history)", lines=1, value=""),
        gr.Slider(
            label="Your Request Retention",
            minimum=0.6,
            maximum=0.97,
            step=0.01,
            value=0.9,
        ),
    ],
    outputs=[
        gr.Textbox(label="test sequences"),
        gr.Textbox(label="default preview"),
        gr.Textbox(label="state history (require delta_ts)"),
    ],
)

iface.launch()
