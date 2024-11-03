from typing import List, Tuple
import gradio as gr  # type: ignore
import os
import sys

if os.environ.get("DEV_MODE"):
    # for local development
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))
from fsrs_optimizer import Optimizer, DEFAULT_PARAMETER, FSRS, lineToTensor  # type: ignore


def convert_delta_ts(delta_ts: str) -> List[str]:
    delta_ts_list = delta_ts.replace(" ", "").split(",")
    converted_delta_ts = []
    for dt in delta_ts_list:
        if dt.endswith("d"):
            converted_delta_ts.append(dt[:-1])
        elif dt.endswith("m"):
            value = float(dt[:-1]) * 30
            converted_delta_ts.append(str(value))
        elif dt.endswith("y"):
            value = float(dt[:-1]) * 365
            converted_delta_ts.append(str(value))
        else:
            converted_delta_ts.append(dt)
    return converted_delta_ts


def interface_func(
    weights: str, ratings: str, delta_ts: str, request_retention: float
) -> Tuple[str, str, str]:
    weights = weights.replace("[", "").replace("]", "")
    optimizer = Optimizer()
    optimizer.w = list(map(lambda x: float(x.strip()), weights.split(",")))
    test_sequence = optimizer.preview_sequence(
        ratings.replace(" ", ""), request_retention
    )
    default_preview = optimizer.preview(request_retention)
    if delta_ts != "":
        ratings_list = ratings.replace(" ", "").split(",")
        delta_ts_list = convert_delta_ts(delta_ts)
        min_len = min(len(ratings_list), len(delta_ts_list))
        ratings = ",".join(ratings_list[:min_len])
        delta_ts = ",".join(delta_ts_list[:min_len])

        s_history, d_history = memory_state_sequence(ratings, delta_ts, optimizer.w)
        return (
            test_sequence,
            default_preview,
            f"s: {(', '.join(s_history))}\nd: {', '.join(d_history)}",
        )
    return test_sequence, default_preview, ""


def memory_state_sequence(
    r_history: str, t_history: str, weights: List[float]
) -> Tuple[List[str], List[str]]:
    fsrs = FSRS(weights)
    line_tensor = lineToTensor(list(zip([t_history], [r_history]))[0]).unsqueeze(1)
    outputs, _ = fsrs(line_tensor)
    stabilities, difficulties = outputs.transpose(0, 1)[0].transpose(0, 1)
    return (
        list(map(lambda x: str(round(x, 2)), stabilities.tolist())),
        list(map(lambda x: str(round(x, 2)), difficulties.tolist())),
    )


iface = gr.Interface(
    fn=interface_func,
    inputs=[
        gr.Textbox(
            label="weights",
            lines=1,
            value=str(DEFAULT_PARAMETER)[1:-1],
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
