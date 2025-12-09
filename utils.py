import pickle

import numpy as np

from scipy import stats

from torchvision.transforms import Resize

import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)


def save_np_file(filepath, array):
    with open(filepath, 'wb') as f:
        np.save(f, array)


def load_np_file(filepath):
    with open(filepath, 'rb') as f:
        a = np.load(f)
    return a


def save_pickle(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def stitch_plots(filepath1, filepath2, filepath_result, title1=None, title2=None):

    f, axarr = plt.subplots(1, 2, figsize=(30, 20))

    # Show the image on the first subplot
    # axarr[0].imshow(mpimg.imread(filepath1))
    img1 = cv2.imread(filepath1)[:, :, ::-1]
    axarr[0].imshow(img1)
    axarr[0].axis("off")
    if title1:
        axarr[0].set_title(title1)

    # Show the image on the second subplot
    # axarr[1].imshow(mpimg.imread(filepath2))
    img2 = cv2.imread(filepath2)[:, :, ::-1]
    axarr[1].imshow(img2)
    axarr[1].axis("off")
    if title2:
        axarr[1].set_title(title2)

    plt.savefig(filepath_result, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()


def rescale_image(img, new_width):
    """Rescale an image to a new width while preserving the aspect ratio."""
    aspect_ratio = float(img.shape[1]) / float(img.shape[2])
    new_height = int(new_width * aspect_ratio)
    img_resized = Resize((new_height, new_width), antialias=True)(img)
    return img_resized


def print_image(image, path=None, title=None):

    """Plot and save an image."""

    plt.imshow(image.astype(np.uint8))

    if title:
        plt.title(title, fontsize=20)
    plt.axis("off")
    if path:
        plt.savefig(path, bbox_inches='tight', facecolor='white')
    else:
        plt.show()
    plt.close()
    return


def imshow(img, img_title=None, fontsize=10):

    plt.rcParams['figure.facecolor'] = 'white'
    plt.imshow(img.astype(np.uint8))

    if img_title:
        plt.title(img_title, fontsize=fontsize)
    plt.axis("off")
    plt.show()



def bar_plot(bar, values, figpath, bar_labels=None, bar_colors=None, title="", ylabel="", figsize=(10, 10)):
    """Create and save a bar plot."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bar, values, label=bar_labels, color=bar_colors)
    ax.set_title(title, size=30)
    plt.ylabel(ylabel, size=20)
    plt.xticks(rotation=25, size=17, ha='right')
    plt.yticks(size=20)
    ax.legend(prop={'size': 17})
    plt.savefig(figpath, bbox_inches="tight")


def add_image(ax, figpath, x, y, zoom=0.15):
    """Add an image to a plot."""

    # Load the image
    img = mpimg.imread(figpath)

    # The OffsetBox is a simple container artist.
    # The child artists are meant to be drawn at a relative position to its #parent.

    imagebox = OffsetImage(img, zoom)
    # Annotation box for solar pv logo
    # Container for the imagebox referring to a specific position *xy*.

    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

    return ax


def make_scatter_plot(x, y, figpath, figsize=(5, 15), title="", subtitle="", colors=None, labels=None,
                      fontsize=20, x_label=None, y_label=None, xlim=None, ylim=None, ):
    """Create a two-dimensional scatterplot of two variables. """

    fig = plt.figure(figsize=figsize)

    # Create axes
    ax = fig.add_subplot()
    ax.scatter(x=x, y=y, c=colors, cmap='plasma')

    # If labels are provided, create a custom legend
    if labels is not None:
        # Create a custom legend
        colors = [plt.cm.plasma(i / len(labels)) for i in range(len(labels))]  # Get color for each unique class

        # Generate custom legend handles
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=20)
                   for color in colors]

        # Add the custom legend to the plot, outside the plot area
        ax.legend(handles, [l.replace("_", " ") for l in labels], loc="center left", bbox_to_anchor=(1.0, 0.5),
                  title="", fontsize=fontsize, title_fontsize=fontsize)

    plt.suptitle(title, fontsize=fontsize + 2)
    ax.set_title(subtitle, fontsize=fontsize - 4)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 4)
    plt.yticks(fontsize=fontsize - 4)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    plt.grid(False)
    plt.savefig(figpath, bbox_inches='tight')

    return ax


def correlation_test(x, y, figpath, title="", x_label="", y_label="", figsize=(20, 15), fontsize=30,
                     colors=None, labels=None):
    """Compute PCC and SRCC statistics for two variables. Plot the two variables."""

    r, r_p_value = stats.pearsonr(x, y)
    s, s_p_value = stats.spearmanr(x, y)
    r = round(r, 3)
    s = round(s, 3)
    r_p_value = round(r_p_value, 3)
    s_p_value = round(s_p_value, 3)
    subtitle = "Pearson's r " + str(r) + ", p=" + str(r_p_value) + r", Spearman's $\rho$ " + str(s) + ", p=" + str(
        s_p_value)
    ax = make_scatter_plot(x, y, figsize=figsize, fontsize=fontsize,
                           figpath=figpath, title=title, subtitle=subtitle,
                           x_label=x_label, y_label=y_label, colors=colors, labels=labels)
    print(x_label, "vs.", y_label, " (", len(x), "images):", "Pearson", r, "p", r_p_value, ", Spearman", s, "p",
          s_p_value)
    return (r, r_p_value, s, s_p_value), ax

def resize_max_dim(img, target_size):
    """
    Resize an image so its largest dimension becomes `target_size`
    without changing aspect ratio (no warping).

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR)
    target_size : int
        The desired size of the largest dimension (width or height)

    Returns
    -------
    resized : np.ndarray
        Resized image
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized
