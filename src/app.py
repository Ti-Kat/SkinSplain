# Import packages
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, callback, ctx, dash_table
from dash.exceptions import PreventUpdate
from functools import reduce
import dash_daq as daq
import plotly.graph_objs as go

import torchvision
from PIL import Image
import base64
import cv2
import io
import random
import numpy as np
import pandas as pd
import csv
import datetime
import os

from webinterface.dashboard_utils import img_to_fig, crop_resize_img, create_result_plot
from common.data_handling import generate_datasets, generate_dataloader, TRANSFORMS_BASE, TRANSFORMS_TRAIN
from common.diagnosis_mapping import DIAGNOSIS_TO_NAME
from common.modelling_base import ModelHandler, MODELNAME_BINARY

from common.path_constants import SALIENCY_MAP_PATH
from common.path_constants import IMG_REL_INPUT_PATH

EXCLUDED_COLUMNS = ['image_path']

full_dataset, train_dataset, val_dataset, test_dataset = generate_datasets(train_transform=TRANSFORMS_TRAIN,
                                                                            base_transform=TRANSFORMS_BASE,
                                                                            split_all_vs_2018=True)
full_dataset_loader, train_dataset_loader, val_dataset_loader, test_dataset_loader \
    = generate_dataloader(full_dataset, train_dataset, val_dataset, test_dataset, weighted_sampling=True)

indices = {
    'train_indices': train_dataset.indices,
    'val_indices': val_dataset.indices,
    'test_indices': test_dataset.indices,
}

model_handler = ModelHandler.load_models(dataset=full_dataset,
                                         time_stamp_path='2024_10_16-22_54_58', # Optional: Specifies model, otherwise uses most recent
                                         model_name=MODELNAME_BINARY)
metadata = full_dataset.meta_data

img_0_str = metadata.at[0, 'image_path']
fig_0 = img_to_fig(Image.open(img_0_str).resize((600,400)))
meta_0 = (
    metadata[metadata['image_name'] == full_dataset.get_image_name_by_index(0)]
    .drop(columns=EXCLUDED_COLUMNS)
    .to_dict(orient='records')
)

# Initialize the app
app = Dash(__name__,  external_stylesheets=[dbc.themes.GRID, dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    dbc.Row(dbc.Col(html.H1(children='SkinsPlain @ RC Trust',
                            className="main_title",
                            style={
                                "font-weight": "bold",
                                "color": "black",
                                "textAlign": "center",
                                'margin-top': '20px',
                                'margin-bottom': '20px'
                            }))),                 
    dbc.Row(dbc.Col(html.Div(
                                [
                                    dbc.Button("Information", id="open", n_clicks=0,
                                               style = {'font-size': '20px', 'width': '200px', 'height' : '50px',
                                                        'backgroundColor': 'white', 
                                                        'color':'#003383',}),
                                    dbc.Modal(
                                                html.Div(
                                                    [
                                                        dbc.ModalHeader(
                                                            dbc.ModalTitle("More information on the task", style={'fontSize': '36px'}),
                                                            close_button=False  # Removes the default closing "x"
                                                        ),
                                                        dbc.ModalBody(
                                                            "On the left part of the dashboard you see the current input image, while the right part portrays the analysis results, which are computed after pressing the button 'Analyse Image'. After conducting multiple manipulations, please give a trust rating (at the bottom part) to continue with the next input image.",
                                                            style={'fontSize': '24px'}
                                                        ),
                                                        dbc.ModalBody(
                                                            "Note that if you want to zoom into a zoomed in picture, please reset the image and then select your desired area of zoom.",
                                                            style={'fontSize': '24px'}
                                                        ),
                                                        dbc.ModalFooter(
                                                            dbc.Button(
                                                                "Close", 
                                                                id="close", 
                                                                className="ms-auto", 
                                                                n_clicks=0,
                                                                style={'font-size': '24px', 'width': '150px', 'height': '50px'}
                                                            )
                                                        ),
                                                    ],
                                                    style={
                                                        "maxWidth": "1000px", 
                                                        "width": "1000px", 
                                                        "margin": "auto",
                                                        "backgroundColor": "white",
                                                    }
                                                ),
                                                id="modal",
                                                is_open=False,
                                                centered=True,
                                                style={
                                                    "display": "flex", 
                                                    "alignItems": "center", 
                                                    "justifyContent": "center"
                                                },
                                            ),
                                ],
                                style = {'margin-bottom': '2em'}
                            ))),
    dbc.Row([
        dbc.Col([   
            html.H2(children='Input Image',
                    style={
                        "font-weight": "bold",
                        "color": "black",
                        "textAlign": "center",
                    }),
            dcc.Loading(
                id='loading',
                type="circle",
                children=[
                    dcc.Graph(
                        id='image-output2',
                        figure=fig_0,
                        style={
                            'display': 'flex',
                            'justify-content': 'center',
                            'align-items': 'center',
                            'height': '50%',
                            'margin-bottom': '2em',
                            'margin-top': '25px',
                        }
                    ),
                ]
            ),
            html.Div(id='image-baseName', children=full_dataset.get_image_name_by_index(0), style={'display': 'none'}),

            # Wrap the metadata table + first dropdown row in a Div with left margin
            html.Div([
                dash_table.DataTable(
                    meta_0,
                    [{"name": i, "id": i} for i in metadata.columns if i not in EXCLUDED_COLUMNS],
                    id='image-metadata',
                    style_cell={'marginLeft': 'auto', 'marginRight': 'auto', 'text-align': 'center'}
                ),
                dbc.Row([
                    dbc.Col([dcc.Dropdown(
                        options=[
                            {'label': DIAGNOSIS_TO_NAME['MEL'] + ' (Malignant)', 'value': 'MEL'},
                            # {'label': DIAGNOSIS_TO_NAME['BCC'], 'value': 'BCC'},
                            # {'label': DIAGNOSIS_TO_NAME['AK'], 'value': 'AK'},
                            # {'label': DIAGNOSIS_TO_NAME['SCC'], 'value': 'SCC'},
                            {'label': DIAGNOSIS_TO_NAME['NV'] + ' (Benign)', 'value': 'NV'}, 
                            # {'label': DIAGNOSIS_TO_NAME['BKL'], 'value': 'BKL'},
                            # {'label': DIAGNOSIS_TO_NAME['VASC'], 'value': 'VASC'},
                            # {'label': DIAGNOSIS_TO_NAME['DF'], 'value': 'DF'},
                            # {'label': DIAGNOSIS_TO_NAME['UNK'], 'value': 'UNK'},
                        ],
                        value='',
                        placeholder="Diagnosis",
                        id='dropdown-diagnosis',
                        style={'width': '250px'}
                    )], width=3),
                    dbc.Col([dcc.Dropdown(
                        options=[
                            {'label': 'Male', 'value': 'male'},
                            {'label': 'Female', 'value': 'female'}
                        ],
                        value='',
                        placeholder="Sex",
                        id='dropdown-sex',
                        style={'width': '250px'}
                    )], width=3),
                    dbc.Col([dcc.Dropdown(
                        options=[
                            {'label': '<= 50 years old', 'value': '0_50'},
                            {'label': '> 50 years old', 'value': '51_150'}
                        ],
                        value='',
                        placeholder="Age",
                        id='dropdown-age',
                        style={'width': '250px'}
                    )], width=3),
                    dbc.Col([dcc.Dropdown(
                        options=[
                            {'label': 'Head / Neck', 'value': 'head/neck'},
                            {'label': 'Upper extremity', 'value': 'upper extremity'},
                            {'label': 'Lower extremity', 'value': 'lower extremity'},
                            {'label': 'Torso', 'value': 'torso'},
                            {'label': 'Anterior torso', 'value': 'anterior torso'},
                            {'label': 'Posterior torso', 'value': 'posterior torso'},
                            {'label': 'Lateral torso', 'value': 'lateral torso'},
                            {'label': 'Hand', 'value': 'hand'},
                            {'label': 'Foot', 'value': 'foot'},
                            {'label': 'Palms / Soles', 'value': 'palms/soles'},
                            {'label': 'Oral / Genital', 'value': 'oral/genital'}
                        ],
                        value='',
                        placeholder="Body location",
                        id='dropdown-loc',
                        style={'width': '250px'}
                    )], width=3),
                ], style={'margin-bottom': '2em', 'margin-top': '25px'}),
            ], style={'margin-left': '50px'}),  # ← ADDED MARGIN HERE

            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'New Image',
                        id='choose_new_img',
                        n_clicks=0,
                        style={
                            'margin-left': '75px',
                            'backgroundColor': '#003383', 
                            'color': 'white',
                            'font-size': '20px',
                            'width': '200px',
                            'height': '50px',
                            'padding': '0',
                            'border': 'none',
                        }
                    )
                ]),
                dbc.Col([
                    dbc.Button(
                        'Analyse Image',
                        id='analysis_trigger',
                        style={
                            'margin-left': '-55px',
                            'color': 'white',
                            'backgroundColor': '#003383', 
                            'font-size': '20px',
                            'width': '200px',
                            'height': '50px',
                            'padding': '0',
                            'border': 'none',
                        }
                    )
                ]),
            ],
            style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'margin-bottom': '3em',
            }),

            dbc.Row([
                dbc.Col([html.P('Brightness', style={"font-weight": "bold", "textAlign": "left"})],
                        width=1),
                dbc.Col([
                    dcc.Slider(
                        0, 2, 0.2,
                        value=1,
                        id='brightness',
                        className='slider',
                        marks={
                            0: '-5',
                            0.2: '-4',
                            0.4: '-3',
                            0.6: '-2',
                            0.8: '-1',
                            1: 'Neutral',
                            1.2: '+1',
                            1.4: '+2',
                            1.6: '+3',
                            1.8: '+4',
                            2: '+5'
                        }
                    )
                ],
                style={
                    'margin-left': '80px',
                    'width': '70%'
                },
                width=12)
            ],
            style={
                'margin-left': '60px',
                'margin-bottom': '50px',
                'margin-top': '15px'
            }),

            dbc.Row([
                dbc.Col([html.P('Blur', style={"font-weight": "bold", "textAlign": "left"})],
                        width=1),
                dbc.Col([
                    dcc.Slider(
                        0, 10, 1,
                        value=0,
                        id='blurring',
                        className='slider'
                    )
                ],
                style={
                    'margin-left': '80px',
                    'width': '70%'
                },
                width=12)
            ],
            style={
                'margin-left': '60px',
                'margin-bottom': '50px'
            }),

            dbc.Row([
                dbc.Col([html.P('Rotation', style={"font-weight": "bold", "textAlign": "left"})],
                        width=1),
                dbc.Col([
                    dcc.Slider(
                        0, 360, 40,
                        value=0,
                        id='rotation',
                        className='slider',
                        marks={
                            0: '0°',
                            40: '40°',
                            80: '80°',
                            120: '120°',
                            160: '160°',
                            200: '200°',
                            240: '240°',
                            280: '280°',
                            320: '320°',
                            360: '360°'
                        }
                    )
                ],
                style={
                    'margin-left': '80px',
                    'width': '70%'
                },
                width=12)
            ],
            style={
                'margin-left': '60px',
            }),

            html.Img(
                id='image-output',
                src=img_0_str,
                style={
                    'display': 'none',
                }
            ),
        ], width=6),

        dbc.Col([
            html.H2(
                children='Analysis Results',
                style={
                    "font-weight": "bold",
                    "color": "black",
                    "textAlign": "center",
                    "margin-bottom": "25px",
                }
            ),
            dcc.Loading(
                id='loading2',
                type="circle",
                children=[
                    dbc.Row([
                        # LEFT COLUMN: Visual Explanation
                        dbc.Col([
                            html.H5(
                                "Visual Explanation",
                                style={
                                    "font-weight": "bold",
                                    "textAlign": "center",
                                    'margin-top': '20px',
                                }
                            ),
                            html.Img(
                                id='image-saliency',
                                style={
                                    'height': '100%',
                                    'display': 'block',
                                    'margin': '0 auto'
                                }
                            ),
                        ], width=6),

                        # RIGHT COLUMN: Melanoma Score (radial gauge) & Reliability Score
                        dbc.Col([
                            html.H5(
                                "Melanoma Score",
                                style={
                                    "font-weight": "bold",
                                    "textAlign": "center",
                                    'margin-top': '20px',
                                    'margin-bottom': '1em'
                                }
                            ),
                            dcc.Graph(
                                id='pred-result-input',
                                config={'displayModeBar': False},
                                figure=go.Figure(
                                    go.Indicator(
                                        mode="gauge+number",
                                        value=5,
                                        gauge={
                                            'axis': {'range': [0, 10]},
                                            'bar': {'color': "gray"},
                                            'bgcolor': "white",
                                            'borderwidth': 1,
                                            'bordercolor': "lightgray",
                                            'shape': "angular",
                                            'steps': [
                                                {'range': [0, 3], 'color': "#e0e0e0"},
                                                {'range': [3, 7], 'color': "#c0c0c0"},
                                                {'range': [7, 10], 'color': "#a0a0a0"},
                                            ],
                                        },
                                        domain={'x': [0, 1], 'y': [0, 1]}
                                    )
                                ).update_layout(
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    height=250
                                )
                            ),

                            html.H5(
                                "Reliability Score",
                                style={
                                    "font-weight": "bold",
                                    "textAlign": "center",
                                    'margin-top': '40px',
                                    'margin-bottom': '1em'
                                }
                            ),
                            html.Img(
                                id='reliab-result-input',
                                style={
                                    'height': '120px',
                                    'display': 'block',
                                    'margin': '0 auto'
                                }
                            ),
                        ], width=6)
                    ], style={'margin-top': '2em'}),

                    dbc.Row([
                        dbc.Col([
                            html.H5(
                                "Most similar predicted malignant image",
                                style={
                                    "font-weight": "bold",
                                    "textAlign": "center",
                                    'margin-top': '20px',
                                    'margin-bottom': '1em'
                                }
                            ),
                            html.Img(
                                id='image-ce-output-m',
                                style={
                                    'height': '60%',
                                    'margin-bottom': '1em'
                                }
                            ),
                        ], width=6),
                        dbc.Col([
                            html.H5(
                                "Most similar predicted benign image",
                                style={
                                    "font-weight": "bold",
                                    "textAlign": "center",
                                    'margin-top': '20px',
                                    'margin-bottom': '1em'
                                }
                            ),
                            html.Img(
                                id='image-ce-output-b',
                                style={
                                    'height': '60%',
                                    'margin-bottom': '1em',
                                }
                            ),
                        ], width=6)
                    ],
                    style={
                        'margin-bottom': '-150px',
                        'margin-top': '50px'
                    }),
                ]
            )
        ], width=6),
    ], style={'textAlign': 'center'}),

    dbc.Row(
        html.H2(
            children='Trust Evaluation',
            style={
                "font-weight": "bold",
                "color": "black",
                "textAlign": "center",
                'margin-top': '2em',
                'margin-bottom': '40px'
            }
        )
    ),
    dbc.Row([
        dbc.Col(
            html.P(
                'Level of Trust',
                style={
                    'margin-right': '2em',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'font-weight': 'bold'
                }
            ),
            width=2
        ),
        dbc.Col(
            html.Div(
                dcc.Slider(0, 10, 1, value=0, id='trust', className='slider'),
                style={
                    'justify-content': 'center',
                    'align-items': 'center',
                    'width': '70%'
                }
            ),
            width=9
        )
    ],
    style={
        'justify-content': 'center',
        'align-items': 'center',
        'margin-left': '150px',
        'margin-bottom': '30px'
    }),
    dbc.Row(
        dbc.Button(
            'Submit Trust Score',
            id='submit_trust_score',
            n_clicks=0,
            style={
                'margin-bottom': '1em',
                'backgroundColor': '#d73027',
                'color': 'white',
                'font-size': '20px',
                'width': '200px',
                'height': '50px',
                'padding': '0',
                'border': 'none',
            }
        ),
        style={
            'justify-content': 'center',
            'align-items': 'center'
        }
    )
], style={'textAlign': 'center'})


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    [Output('choose_new_img', 'n_clicks'), 
     Output('trust', 'value')],
    [Input('submit_trust_score', 'n_clicks')],
    [State('brightness', 'value'),
     State('blurring', 'value'),
     State('rotation', 'value'),
     State('image-baseName', 'children'),
     State('trust', 'value'),
     State('choose_new_img', 'n_clicks'),
    ])
def pred_ce(submit_click, brightness, blurring, rotation, input_img, trust, n_clicks):
    if ctx.triggered_id is None:
        raise PreventUpdate

    new_n_clicks = n_clicks + 1

    # Defines the CSV file path (same directory as the app.py)
    csv_file = 'trust_results.csv'
    file_exists = os.path.isfile(csv_file)
    
    # Get current datetime as a string
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Open the file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(['image_name', 'trust', 'brightness', 'blurring', 'rotation', 'time'])
        # Write the current row data
        writer.writerow([input_img, trust, brightness, blurring, rotation, current_time])

    # Reset the trust score for the next input
    trust_reset = 0
    return new_n_clicks, trust_reset

@callback(
    [Output('image-baseName', 'children'), 
     Output('image-metadata', 'data'), 
     ],
    [Input('choose_new_img', 'n_clicks'),
     State('dropdown-diagnosis', 'value'),
     State('dropdown-sex', 'value'),
     State('dropdown-age', 'value'),
     State('dropdown-loc', 'value'),
    ])
def select_new_base_img(_, drop_diagnosis, drop_sex, drop_age, drop_loc):
    # Check which conditions were selected
    conditions = []
    if drop_diagnosis:
        conditions.append(metadata['diagnosis'] == drop_diagnosis)
    if drop_sex:
        conditions.append(metadata['sex'] == drop_sex)
    if drop_loc:
        conditions.append(metadata['anatom_site_general'] == drop_loc)
    if drop_age:
        age_low, age_high = drop_age.split('_')
        conditions.append(metadata['age_approx'] >= int(age_low))
        conditions.append(metadata['age_approx'] <= int(age_high))

    # Combine all selected conditions
    if conditions:
        combined_condition = reduce(lambda x, y: x & y, conditions)
        filtered_metadata = metadata[combined_condition]
    else:
        filtered_metadata = metadata
    candidate_img_names = list(filtered_metadata['image_name'])

    if len(candidate_img_names) == 0:
        print('No image found for given selection. Proceed with random image.')
        img_name = random.choice(metadata['image_name'])
    else:
        img_name = random.choice(candidate_img_names)

    meta_i = metadata[metadata['image_name'] == img_name].to_dict('records')
    return img_name, meta_i


@callback(
    [Output('image-output', 'src'), 
      Output('image-output2', 'figure'), 
      Output('brightness', 'value'),
      Output('blurring', 'value'),
      Output('rotation', 'value'),
      Output('image-output2', 'relayoutData')
     ],
     [Input('brightness', 'value'),
      Input('blurring', 'value'),
      Input('rotation', 'value'),
      Input('image-baseName', 'children'),
      Input('image-output2', 'relayoutData'),
    ])
def update_img(bright, blur, angle, input_img, relayData): 
    if ctx.triggered_id == 'choose_new_img':
        blur = 0
        angle = 0
        bright = 1
        relayData = None
    
    img = Image.open(full_dataset.get_image_path(image_name=input_img)).resize((600,400))

    new_img = torchvision.transforms.functional.adjust_brightness(img, float(bright))
    if blur != 0:
        new_img = torchvision.transforms.functional.gaussian_blur(new_img, kernel_size=int(blur*2 -1))

    if relayData:
        if 'xaxis.range[0]' in relayData.keys():
            new_img = crop_resize_img(new_img, relayData)
    if angle != 0:
        new_img = torchvision.transforms.functional.rotate(new_img, angle=int(angle))   

    fig = img_to_fig(new_img)
    
    # Ensure the directory exists
    save_dir = "buffer"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a filename based on the input image name (you may adjust this naming as needed)
    save_path = os.path.join(save_dir, f"transformed_{input_img}.png")
    print("Saved image path:", save_path)
    
    # Save the image with 400 dpi; the dpi argument must be a tuple.
    final_img = new_img.resize((600, 600))
    final_img.save(save_path, format="PNG", dpi=(400, 400))
    
    
    return new_img.resize((6000,4000)), fig, bright, blur, angle, relayData


@callback(
    [Output('image-ce-output-m', 'src'), 
     Output('image-ce-output-b', 'src'), 
     Output('pred-result-input', 'figure'), 
     Output('reliab-result-input', 'src'),
     Output('image-saliency', 'src'), 
     ],
    [Input('analysis_trigger', 'n_clicks'),
     State('image-output', 'src'),
     Input('choose_new_img', 'n_clicks'),
     Input('image-baseName', 'children'), 
    ])
def pred_ce(_, input_img, n_clicks_new_img, img_name):
    if input_img[-4:] == '.jpg':
        img = Image.open(input_img)
        img_name = full_dataset.get_image_name_by_index(n_clicks_new_img)
    elif ctx.triggered_id == 'choose_new_img':
        img = Image.open(full_dataset.get_image_path(image_name=img_name))
    else:
        image_bytes = base64.b64decode(input_img.split(',')[1])
        img = Image.open(io.BytesIO(image_bytes))
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    ce_m, ce_b = model_handler.get_most_similar_images(image_name=img_name, image=img_np, save_plot=False)

    pred_input = model_handler.get_melanoma_score(image=full_dataset.transform_image(img_np)) # Melanoma Scores: 10 maximal malignant
    print(f'Melanoma Score: {pred_input}')

    # TODO: Update this with final reliability Score
    img_rel = model_handler.get_reliability_score(image_name=img_name, image=img_np)
    print(f'Reliability Score: {img_rel}')
    create_result_plot(img_rel, 'input_img_rel.jpg', redToGreen = True)

    model_handler.get_saliency_map(image=img_np, single=True) # via https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py

    saliency_map = Image.open(SALIENCY_MAP_PATH).resize((600,400))
    ce_m = Image.open(full_dataset.get_image_path(image_name=ce_m)).resize((600,400))
    ce_b = Image.open(full_dataset.get_image_path(image_name=ce_b)).resize((600,400))
    img_rel_input = Image.open(IMG_REL_INPUT_PATH)
    
    melanoma_gauge_figure = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=float(pred_input),
        number={'font': {'color': "black"}},  # Value number color
        gauge={
            'axis': {
                'range': [0, 10],
                'tickmode': 'linear',  # Use linear tick mode
                'dtick': 2,            # Show a tick every 2 units
                'tickcolor': 'black',
                'tickfont': {'color': 'black'}
            },
            'bar': {'color': "black"},  # The filled portion of the arc
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "lightgray",
            'shape': "angular",
            'steps': [
                {'range': [0, 2], 'color': "#00cc96"},   # Green
                {'range': [2, 5], 'color': "#ffa600"},   # Yellow/Orange
                {'range': [5, 10], 'color': "#ef553b"},  # Red
            ]
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    )
    )
    melanoma_gauge_figure.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=250
    )
    
    

    return ce_m, ce_b, melanoma_gauge_figure, img_rel_input, saliency_map


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
