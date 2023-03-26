import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import igviz as ig
import networkx as nx
from plotly.offline import plot, iplot, init_notebook_mode
from sklearn.model_selection import train_test_split
# init_notebook_mode(connected=True)


def visualize_celllevel_graph(df, gene, title, edge_trace = None, publication=False):
    '''
    Displays Cell Level (graph of single cells) in plotly visualization.
    
    df: pd.DataFrame : represents the spatial data and contains the following columns ["Cell_ID", "X", "Y", "Cell_Type", "Gene 1", ..., "Gene n"]
    title: str: User defined title of plot
    gene: str : represents the gene name (out of possible gene names ["Gene 1", ..., "Gene n"]) whose column (df[gene]) will be used for hue 
    edge_trace: tuple(list(),list()) : output of preprocess.get_proximal_cells() where the first list represent the x coordinates of the edges and the second list represents the y coordinates
    publication: boolean : True if user wants publication style plot
    
    return: plotly express figure
    '''

    # fig = px.scatter(df, x="X", y="Y", custom_data=["Cell_ID"], color=gene, color_continuous_scale="sunsetdark",width=700, height=650, title=title)

    # if publication:
    #     fig = px.scatter(df, x="X", y="Y", custom_data=["Cell_ID"], color=gene, color_discrete_sequence=px.colors.sequential.Sunsetdark_r,width=700, height=650, title=title)
    # else:
    #     fig = px.scatter(df, x="X", y="Y", custom_data=["Cell_ID"], color=gene, color_continuous_scale="sunsetdark",width=700, height=650, title=title)
    fig = px.scatter(df, x="X", y="Y", custom_data=["Cell_ID"], color=gene,width=700, height=650,color_continuous_scale="sunsetdark", title=title)


    if edge_trace is not None:
        fig.add_trace(go.Scatter( x=edge_trace[0], y=edge_trace[1],
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines',name="Edges"))

    fig.update_traces(
        hovertemplate="<br>".join([
            "Cell_ID: %{customdata[0]}",
            "X: %{x}",
            "Y: %{y}",
        ]),
    )
 
    # fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(1,1,1,0)",font_color="lightgray")    
    if not publication:
        fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=-0.2))

        # pio.renderers.default = "notebook_connected"
        fig.update_xaxes(showline=False, linewidth=2, linecolor='rgba(1,1,1,0)', gridcolor='rgba(1,1,1,0)',zeroline=False)
        fig.update_yaxes(showline=False, linewidth=2, linecolor='rgba(1,1,1,0)', gridcolor='rgba(1,1,1,0)',zeroline=False)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='rgba(1,1,1,.2)', gridcolor='rgba(1,1,1,.2)',zeroline=False)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='rgba(1,1,1,.2)', gridcolor='rgba(1,1,1,.2)',zeroline=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(1,1,1,0)",font_color="lightgray")
    return fig



def visualize_grn_igviz(grn, gene_attributes, title):
    '''
    Displays Gene Regulatory Network Graph with igviz.
    
    grn: nd.array(shape=(n,n)) : gene by gene adjacency matrix representing the gene regulatory network (GRN)
    gene_attributes: dict() : dictonary that maps index to gene name (see networkX definition for more detail)
    title: str : User defined title of plot
    
    return: plotly express figure
    '''
    
    G =  nx.from_numpy_matrix(grn)
    nx.set_node_attributes(G, gene_attributes, "Gene Name")

    color_list = []
    sizing_list = []
    for node in G.nodes():
        size_and_color = G.degree(node)
        color_list.append(size_and_color)
        sizing_list.append(size_and_color)
        
    fig = ig.plot(
        G,
        title=title,
        layout="spring",
        node_text = ["Gene Name"],
        size_method=sizing_list, # Makes node sizes the size of the "prop" property
        color_method=color_list, # Colors the nodes based off the "prop" property and a color scale
        colorscale="Sunsetdark"
    )
    fig.update_traces(marker=dict(line=dict(width=0.5)))
    fig.update_layout(autosize = False,height=600,width=700,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(1,1,1,0)",font_color="lightgray")
    fig.update_xaxes(showline=False, linewidth=2, linecolor='rgba(1,1,1,.2)', gridcolor='rgba(1,1,1,.2)',zeroline=False)
    fig.update_yaxes(showline=False, linewidth=2, linecolor='rgba(1,1,1,.2)', gridcolor='rgba(1,1,1,.2)',zeroline=False)
    fig.data[0].line.color = "rgba(255, 83, 92, 0.15)"
    
    return fig



def visualize_metrics(df, baseline_name, data_name, split, metric_list=["AP","ROC"]):
    figs = []
    for i,metric in enumerate(metric_list):
        fig = go.Figure()
        xaxistitle = "Epoch"
        yaxistitle = f"{metric}"
        fig.add_trace(
            go.Scatter(x=df[xaxistitle],y=df[f"CLARIFY Test {metric}"], line=dict(color="#d14078"),
                       name="Clarify"),
        )
        fig.add_trace(
            go.Scatter(x=df[xaxistitle],y=df[f"{baseline_name} Test {metric}"], line=dict(color="#345c72"),
                       name=baseline_name),
            
        )
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black',tickfont=dict( size=17, color='black'))
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black',tickfont=dict( size=17, color='black'))
        fig.update_layout(
            title=f"({data_name.upper()}) {yaxistitle} over {xaxistitle}s on {split} testing edges",
            xaxis_title=xaxistitle,
            yaxis_title=yaxistitle,
            legend_title="Model",
            width=800,
            height=600,
            plot_bgcolor="white",
            xaxis=dict( mirror=True,
            ticks='outside',
            showline=True,title=xaxistitle),
            yaxis=dict( mirror=True,
            ticks='outside',
            showline=True,title=yaxistitle)
        )
        figs.append(fig)
    return figs