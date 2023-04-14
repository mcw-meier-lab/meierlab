# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from pathlib import Path

def gen_dash_template(out_path,figures):
    out_file = Path(out_path) / "template.py"
    with open(out_file,"a") as outf:
        out_str = f"""
import io
import plotly.graph_objects as go
from dash import Dash, dcc, html
from base64 import b64encode

app = Dash(__name__)
buffer = io.StringIO()
for fig in {figures}:
    fig.write_html(buffer)

html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

# define dashboard layout
app.layout = html.Div(
    children=[
       html.Div(children=[
            dcc.Graph(figure=fig) for fig in {figures}
        ]),
        html.A(
            html.Button("Download as HTML"), 
            id="download",
            href="data:text/html;base64," + encoded,
            download="plotly_graph.html"
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
"""
        outf.write(out_str)

    return
