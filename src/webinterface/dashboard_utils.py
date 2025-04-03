import torchvision
import plotly.express as px
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from common.path_constants import BUFFER

def img_to_fig(img):
    fig = px.imshow(torchvision.transforms.functional.pil_to_tensor(img).permute(1, 2, 0))
    fig.update_xaxes(
        visible=False,
        )
    fig.update_yaxes(
        visible=False,
        scaleanchor="x"
        )
    fig.update_layout(hovermode = False, margin = {'l': 0, 'r': 0, 't': 0, 'b': 0})
    return fig


def crop_resize_img(img, points):
    img = torchvision.transforms.functional.pil_to_tensor(img)
    size = img.size()[1:]
    points = {k: round(v) for k, v in points.items()}
    img = torchvision.transforms.functional.resized_crop(img, top = points['yaxis.range[1]'], left = points['xaxis.range[0]'],
                                                 height = points['yaxis.range[0]'] - points['yaxis.range[1]'],
                                                 width = points['xaxis.range[1]'] - points['xaxis.range[0]'],
                                                 size = size,
                                                 interpolation = torchvision.transforms.InterpolationMode.BILINEAR
                                                )
    img = torchvision.transforms.functional.to_pil_image(img)
    return img


def create_result_plot(result, filename, min = 0, max = 10, redToGreen = False):
    # Create single bar plot
    num_sections = max-min
    
    fig = Figure(figsize=(20, 4))
    ax = fig.subplots()

    #fig, ax = plt.subplots(figsize=(10, 2))
    bar = ax.barh(0, num_sections, alpha=0.8)

    # Set gradient
    grad = np.atleast_2d(np.linspace(0,1,256))

    bar[0].set_zorder(1)
    bar[0].set_facecolor("none")
    x,y = bar[0].get_xy()
    w, h = bar[0].get_width(), bar[0].get_height()
    if redToGreen:
        colmap = plt.cm.RdYlGn
    else:
        colmap = plt.cm.RdYlGn_r

    ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", zorder=0, cmap=colmap)

    # Draw a triangle above the corresponding section based on the value of "result"
    ax.fill([result - 0.3, result, result + 0.3], [1.2, 0.7, 1.2], 'black')

    # Set x-ticks and labels
    ax.set_xticks(range(min, max + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid(False)
    ax.xaxis.set_tick_params(labelsize=30)


    fig.savefig(f'{BUFFER}/{filename}')
    print(f"Saved result plot {filename}")
    pass
