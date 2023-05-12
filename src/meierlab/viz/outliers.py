# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import io
import plotly.graph_objects as go
from dash import dcc, html
from base64 import b64encode


def gen_figs(outlier_df,x_cols,
             y_cols=[('pos_count','neg_count'),('pos_pct','neg_pct')]):
    """Generate plotly figures from an outlier dataframe.

    Parameters
    ----------
    outlier_df : :class:`~pandas.DataFrame`
        Dataframe with outlier info (from meierlab.outliers)
    x_cols : list
        x-axis columns to use.
    y_cols : list, optional
        y-axis columns to use, by default [('pos_count','neg_count'),('pos_pct','neg_pct')]

    Returns
    -------
    list
        List of :class:`~plotly.graph_objects.Figure` violin plots.
    """
    figs = list(zip(x_cols, y_cols))
    out_figs = []
    for x,y in figs:
        fig = go.Figure()
        fig.add_trace(go.Violin(
            x=outlier_df[x],
            y=outlier_df[y[0]],
            line_color='blue' if 'neg' in y[0] else 'orange',
            name=y[0],
            pointpos=1.2,
            text=outlier_df['subject'],
            hovertemplate="%{y}: <br>%{text}"
        ))
        fig.add_trace(go.Violin(
            x=outlier_df[x],
            y=outlier_df[y[1]],
            line_color='blue' if 'neg' in y[1] else 'orange',
            name=y[1],
            pointpos=1.2,
            text=outlier_df['subject'],
            hovertemplate="%{y}: <br>%{text}" 
        ))
        fig.update_layout(violinmode='group')
        fig.update_traces(points='all',jitter=0.05,scalemode='count',meanline_visible=True)
        fig.update_xaxes(title=x)
        
        out_figs.append(fig)

    return out_figs

def get_app(figures,app_type=None):
    """Get a dash app to view data.

    Parameters
    ----------
    figures : list
        List of :class:`~plotly.graph_objects` figures to view.
    app_type : str, optional
        Use "jupyter" if viewing via Jupyter Notebook/Lab, by default None

    Returns
    -------
    :class:`~dash.Dash` or :class:`~jupyter_dash.JupyterDash`
        Dash HTML app with figures to view. Run via `app.run()` and open a browser with the local host address (or view through Jupyter).
    """
    if app_type == "jupyter":
        from jupyter_dash import JupyterDash
        app = JupyterDash(__name__)
    else:
        from dash import Dash
        app = Dash(__name__)

    buffer = io.StringIO()
    for fig in figures:
        fig.write_html(buffer)

    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    graphs = []
    for fig in figures:
        graphs.append(dcc.Graph(figure=fig))

    # define dashboard layout
    app.layout = html.Div(
        children=[
            html.Div(children=graphs),
            html.A(
                html.Button("Download as HTML"), 
                id="download",
                href="data:text/html;base64," + encoded,
                download="plotly_graph.html"
            )
        ]
    )

    return app
