import json

import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
import flask
import seaborn as sns
from dash import Dash, Input, Output, dcc, html, State

ZOOM_LEVELS = []
CURRENT_ZOOM_LEVEL = 0

server = flask.Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])


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
    samples = [data["dataset"]["samples"][i]["data"]
               for i in range(len(data["dataset"]["samples"]))]
    level_data = data["data"][f"{zoom_level}"]

    clusters = level_data["clusters"]

    cluster_keywords = level_data["cluster_keywords"]
    labels = [cluster_keywords[str(clusters[i])] for i in range(len(clusters))]

    embeddings = level_data["tsne_embeddings"]
    if(zoom_level == -1):
        assert (
            len(samples) == len(labels) == len(clusters)
        ), "Samples, clusters and labels must have the same length"

    # Create the nodes
    nodes = []
    # TODO FIX
    for i in range(len(clusters)):
        nodes.append(
            {
                "data": {
                    "id": i,
                    "label": labels[i],
                    "group": clusters[i],
                    "text": samples[i] if zoom_level == -1 else "This node represents a cluster of articles.\n You can see what the cluster is about from the tags below, or use the slider to increase the zoom level and see the articles inside.",
                },
                "position": {"x": embeddings[i][0], "y": embeddings[i][1]},
                "classes": f"node-{clusters[i]}",
                "grabbable": False,
                "selectable": True,
                "selected": False,
            }
        )

    return nodes

def get_stylesheet(nodes: list[dict]):
    size = max(150 / len(nodes), 2)

    default_stylesheet = [
        {
            "selector": "node",
            "style": {"width": size, "height": size},
        },
    ]

    unique_clusters = list(set([node["data"]["group"] for node in nodes]))
    # generate a seaborn color palette for the clusters
    palette = sns.color_palette("tab20", len(unique_clusters)) + sns.color_palette("tab20b", len(unique_clusters)) + sns.color_palette("tab20c", len(unique_clusters))
    for i, cluster in enumerate(unique_clusters):
        # convert palette to hex
        palette[i] = "#" + "".join([f"{int(255*c):02x}" for c in palette[i]])
        default_stylesheet.append(
            {
                "selector": f".node-{cluster}",
                "style": {
                    "background-color": palette[i],
                    "line-color": palette[i],
                },
            }
        )

    return default_stylesheet

def get_layout(nodes: list[dict], levels: list) -> html.Div:
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

    default_stylesheet = get_stylesheet(nodes)

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
                userZoomingEnabled=True,
                zoomingEnabled=True,
                zoom=100,
                responsive=True
            ),
            dcc.Slider(0, len(levels)-1, step=None,
                id="zoom-slider",
                marks={i: f"Zoom {levels[i]}" for i in range(len(levels))},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        id="graph-container",
    )

    # HTML structure
    return html.Div(
        [
            html.Section(
                [html.H1("News clustering demo", id="title"), graph],
                className="page",
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Article")),
                    dbc.ModalBody("", id="modal-body"),
                    dbc.ModalFooter([
                        html.Span([], id='tags'),
                    ]),
                ],
                id="modal",
                is_open=False,
            ),
        ],
        id="frame",
    )

def init_app():
    """
    Initialize the app.
    """

    # Initial nodes at default zoom level
    data = json.load(open("data/data.json"))
    global ZOOM_LEVELS, CURRENT_ZOOM_LEVEL
    ZOOM_LEVELS = data["zoom_levels"]
    CURRENT_ZOOM_LEVEL = ZOOM_LEVELS[0]
    nodes = get_nodes(CURRENT_ZOOM_LEVEL)
    app.layout = get_layout(nodes, ZOOM_LEVELS)

    # Set up the callback for the zoom slider
    @app.callback([Output("graph", "elements"), Output("graph", "stylesheet")], 
                   Input("zoom-slider", "value"))
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
        global CURRENT_ZOOM_LEVEL
        if value is not None:
            CURRENT_ZOOM_LEVEL = ZOOM_LEVELS[value]
        nodes = get_nodes(CURRENT_ZOOM_LEVEL)
        style = get_stylesheet(nodes)
        return nodes, style

    # add a callback that prints when a node is clicked
    @app.callback([Output("modal", "is_open"), Output("modal-body", "children"), Output("tags", "children")],
                  [Input("graph", "tapNode")],
                  [State("graph", "tapNodeData")])
    def print_node_data(_, data):
        if data:
            tags = []
            for tag in data["label"]:
                tags.append(
                    dbc.Badge(
                            f"{tag}",
                            color="white",
                            text_color="primary",
                            className="border me-1",
                            ),
                )
            return True, data["text"], tags
        return False, "", []
    
    @app.callback(Output("graph", "zoomingEnabled"),
                  Input("graph", "relayoutData"))
    def update_node_sizes(zoom):
        print("UPDATING NODES")
        print(zoom)
        return True



if __name__ == "app":
    init_app()
