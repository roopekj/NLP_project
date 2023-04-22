import json

import dash_cytoscape as cyto
import flask
from dash import Dash, Input, Output, dcc, html

MIN_ZOOM_LEVEL = 5
MAX_ZOOM_LEVEL = 20
STEP_ZOOM_LEVEL = 5
DEFAULT_ZOOM_LEVEL = 20
N_NODES = 1000
NODE_SIZE = 1
COLORS = [
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


server = flask.Flask(__name__)
app = Dash(__name__, server=server)


def get_nodes(zoom_level: int, max_nodes: int = -1) -> list[dict]:
    """
    Get the nodes for the graph.

    Parameters
    ----------
    zoom_level -> the zoom level of the graph.
    max_nodes -> the maximum number of nodes to return.

    Returns
    -------
    The nodes for the graph.
    """

    # Load the data
    data = json.load(open("data/data.json"))
    tot_samples = max_nodes if max_nodes > 0 else len(data["dataset"]["samples"])
    samples = [ data["dataset"]["samples"][i]["data"] for i in range(tot_samples) ]
    clusters = data["clusters"][:tot_samples]
    
    cluster_keywords = data["cluster_keywords"]
    labels = [cluster_keywords[str(clusters[i])] for i in range(tot_samples)]

    embeddings = data["tsne_embeddings"][:tot_samples]
    assert (
        len(samples) == len(labels) == len(clusters)
    ), "Samples, clusters and labels must have the same length"

    # Create the nodes
    nodes = []
    for i in range(len(samples)):
        nodes.append(
            {
                "data": {
                    "id": i,
                    "label": labels[i],
                    "group": clusters[i],
                },
                "position": {"x": embeddings[i][0], "y": embeddings[i][1]},
                "classes": f"node-{clusters[i]}",
                "grabbable": False,
                "selectable": True,
                "selected": False,
            }
        )

    return nodes


def get_layout(nodes: list[dict]) -> html.Div:
    """
    Get the layout of the app.

    Parameters
    ----------
    nodes -> the nodes of the graph.
    clusters -> the clusters of the nodes.
    labels -> the labels of the nodes.

    Returns
    -------
    The layout of the app.
    """

    # Set stylesheet
    default_stylesheet = [
        {
            "selector": "node",
            "style": {"width": NODE_SIZE, "height": NODE_SIZE},
        },
    ]

    unique_clusters = list(set([node["data"]["group"] for node in nodes]))
    for i, cluster in enumerate(unique_clusters):
        default_stylesheet.append(
            {
                "selector": f".node-{cluster}",
                "style": {
                    "background-color": COLORS[i],
                    "line-color": COLORS[i],
                },
            }
        )

    # Create the graph (a cytoscape)
    cyto.load_extra_layouts()
    graph = html.Div(
        [
            cyto.Cytoscape(
                id="graph",
                # layout={"name": "concentric"},
                # layout={"name": "random"},
                layout={"name": "preset"},
                elements=nodes,
                stylesheet=default_stylesheet,
                style={"width": "100%", "height": "100%"},
            ),
            dcc.Slider(
                MIN_ZOOM_LEVEL,
                MAX_ZOOM_LEVEL,
                STEP_ZOOM_LEVEL,
                value=DEFAULT_ZOOM_LEVEL,
                id="zoom-slider",
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        id="graph-container",
    )

    # HTML structure
    return html.Div(
        [
            html.Section(
                [html.H1("Note clustering demo", id="title"), graph],
                className="page",
            ),
        ],
        id="frame",
    )


def init_app():
    """
    Initialize the app.
    """

    # Initial nodes at default zoom level
    nodes = get_nodes(DEFAULT_ZOOM_LEVEL, N_NODES)
    app.layout = get_layout(nodes)

    # Set up the callback for the zoom slider
    @app.callback(Output("graph", "elements"), Input("zoom-slider", "value"))
    def update_graph_zoom(value: int) -> list[dict]:
        """
        Update the graph when the zoom level changes.

        Parameters
        ----------
        value -> the new zoom level.

        Returns
        -------
        The new nodes for the graph.
        """

        return get_nodes(value, N_NODES)


if __name__ == "app":
    init_app()
