import json
import flask
from dash import Dash, html
import dash_cytoscape as cyto


server = flask.Flask(__name__)
app = Dash(__name__, server=server)

# Load data
samples = json.load(open("data/samples.json"))[:100]
labels = json.load(open("data/labels.json"))[:100]
assert len(samples) == len(labels), "Samples and labels must have the same length"

# Create nodes
nodes = []
for i in range(len(samples)):
    nodes.append(
        {
            "data": {
                "id": i,
                "label": ((" ".join(samples[i].split(" ")[:3])[:10]) + "..."),
                "group": labels[i],
            },
            "classes": f"node-{labels[i]}",
            "grabbable": True,
            "selectable": False,
            "selected": False,
        }
    )

# Set stylesheet
default_stylesheet = [
    {
        "selector": "node",
        "style": {"label": "data(label)"},
    },
    {"selector": ".red", "style": {"background-color": "red", "line-color": "red"}},
]
colors = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffffff",
    "#000000",
]
unique_labels = list(set(labels))
for i, label in enumerate(unique_labels):
    default_stylesheet.append(
        {
            "selector": f".node-{label}",
            "style": {
                "background-color": colors[i],
                "line-color": colors[i],
            },
        }
    )

cyto.load_extra_layouts()
graph = cyto.Cytoscape(
    id="graph",
    # layout={"name": "concentric"},
    # layout={"name": "random"},
    layout={"name": "spread"},
    elements=nodes,
    stylesheet=default_stylesheet,
    style={"width": "100%", "height": "100%"},
)

app.layout = html.Div(
    [
        html.Section(
            [html.H1("Note clustering demo", id="title"), graph],
            className="page",
        ),
    ],
    id="frame",
)

if __name__ == "__main__":
    app.run_server(debug=False)
