import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from color_pixel_projection import map_pixel

def mean_hue_from_hsv(hsv_img: np.ndarray) -> float:
    """
    Computes the circular mean hue of an HSV image (OpenCV format).

    Parameters
    ----------
    hsv_img : np.ndarray
        HSV image from cv2.cvtColor(img, cv2.COLOR_BGR2HSV).
        Hue channel must be in [0, 179].

    Returns
    -------
    float
        Mean hue in degrees (0–360).
    """
    # Extract hue channel (0–179)
    hue = hsv_img[:, :, 0].astype(np.float32)

    # Convert to degrees (0–360)
    hue_deg = hue * 2.0

    # Convert to radians
    rad = np.deg2rad(hue_deg)

    # Circular mean
    mean_x = np.mean(np.cos(rad))
    mean_y = np.mean(np.sin(rad))
    mean_angle = np.arctan2(mean_y, mean_x)

    # Convert back to degrees
    mean_deg = np.rad2deg(mean_angle)

    # Normalize to [0, 360)
    if mean_deg < 0:
        mean_deg += 360

    return mean_deg

def compute_angle(orange_teal, mean_hue, c, l, l_float: float = 1/3):
    """
    * C1: in the direction of the orange-teal axis 
    * C2: away from the orange-teal axis 
    * L1: 1/3rd of the mean rotation angle needed to align the pixels maximally with the orange-teal axis 
    * L2: the maximum rotation angle 
    """
    difference = orange_teal - mean_hue

    if l == 1:
        difference = difference * l_float

    if c == 2:
        difference *= -1
    
    return mean_hue + difference



def recolor_image(hsv_img, hue_degree, rescale_shift, result_path):
    """
    Recolor an HSV image by shifting each pixel's hue toward a specified hue axis.

    Parameters
    ----------
    hsv_img : numpy.ndarray
        Input image in HSV color space (as uint8, OpenCV format: H in [0,179], S,V in [0,255]).
    hue_degree : float
        Target hue axis in degrees (0-360). The algorithm maps each pixel's hue toward
        the nearest direction of this axis (or its opposite at +180 degrees).
    rescale_shift : float
        Factor to scale the amount of hue shift applied. 1 applies the full computed shift,
        <1 reduces the shift, >1 increases it.
    result_path : str
        Filesystem path where the resulting image will be written (saved as an image file).

    Returns
    -------
    None
        The function saves the recolored RGB image to result_path. It does not return a value.
    """
    # Save height and width
    height, width = hsv_img.shape[:2]

    # Reshape image and extract values for hues (H)
    hsv_img = hsv_img.astype(float) # If you remove this all calculations are messed up
    HSV_img_pixels = hsv_img.transpose(2, 0, 1).reshape((3, -1)).transpose(1, 0)

    H = HSV_img_pixels[:, 0] * 2  # Hue in degrees

    # Map hues to color axis
    H = np.array(list(map(lambda x: map_pixel(x, hue_degree, rescale_shift=rescale_shift), H)))

    # Reconstruct image
    HSV_img_pixels = np.concatenate((np.expand_dims((H / 2), axis=1),
                                     np.expand_dims(HSV_img_pixels[:, 1], axis=1),
                                     np.expand_dims(HSV_img_pixels[:, 2], axis=1)), axis=1).astype(np.uint8)
    new_HSV_img = HSV_img_pixels.transpose(1, 0).reshape((3, height, width)).transpose(1, 2, 0)
    RGB_img_pixels = cv2.cvtColor(new_HSV_img, cv2.COLOR_HSV2RGB)

    # Show and save result
    cv2.imwrite(result_path, RGB_img_pixels[:, :, ::-1])

def plot_color_wheel_and_mean_angle(HSV_img, result_path, hue_degree, downsampling_factor=1, n_pixels=30):
    """
    Plot a color wheel of the input HSV image and a side-by-side mean-angle representation,
    then save the resulting figure to disk.

    Parameters
    ----------
    HSV_img : numpy.ndarray
        Image in OpenCV HSV format with shape (H, W, 3) and dtype uint8.
        Hue is expected in OpenCV range [0, 179], saturation and value in [0, 255].
    result_path : str
        Path where the output figure will be saved (e.g. 'color_wheel.png').
    hue_degree : float
        Mean hue in degrees (0-360). The function will plot this hue and its opposite (hue+180).
    downsampling_factor : int, optional
        Step for subsampling pixels when plotting the full color wheel. Larger values speed up plotting.
    n_pixels : int, optional
        Number of radial samples per hue direction used to draw the mean-angle representation.

    Returns
    -------
    None
        The function saves a figure to result_path and closes the matplotlib figure.
    """
    # COLORWHEEL
    hsv_downsampled = HSV_img.transpose(2, 0, 1).reshape((3, -1)).transpose(1, 0)[::downsampling_factor]
    colors = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2BGR).reshape(-1, 3)[::downsampling_factor][:, ::-1]

    theta = hsv_downsampled[:, 0].astype('float16') * 2 * np.pi / 180.0
    r = hsv_downsampled[:, 1]
    x_coordinates = r * np.cos(theta)
    y_coordinates = r * np.sin(theta)

    # MEAN ANGLE
    r2 = np.linspace(0, 255, num=n_pixels, dtype=int)
    theta1 = (hue_degree % 360) * np.ones(n_pixels) * np.pi / 180.0
    theta2 = ((hue_degree + 180) % 360) * np.ones(n_pixels) * np.pi / 180.0

    x_coordinates1 = r2 * np.cos(theta1)
    y_coordinates1 = r2 * np.sin(theta1)
    x_coordinates2 = r2 * np.cos(theta2)
    y_coordinates2 = r2 * np.sin(theta2)

    x_coordinates_mean = np.concatenate((x_coordinates1, x_coordinates2))
    y_coordinates_mean = np.concatenate((y_coordinates1, y_coordinates2))

    h = np.expand_dims(np.concatenate((theta1, theta2), axis=0) * 180 / (2 * np.pi), axis=1)
    s = np.expand_dims(np.concatenate((r2, r2), axis=0), axis=1)
    v = np.ones((len(h), 1)) * 255

    hsv_pixels = np.concatenate((h, s, v), axis=1).astype(np.uint8)
    hsv_pixels = hsv_pixels.reshape((1, -1, 3))
    colors2 = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2RGB).reshape((-1, 3))

    # --- SIDE-BY-SIDE PLOTS ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.5))

    # --- LEFT: Color Wheel ---
    axs[0].scatter(x_coordinates, y_coordinates, c=colors / 255)
    theta_circle = np.linspace(0, 2 * np.pi, 1000)
    x_circle = 255 * np.cos(theta_circle)
    y_circle = 255 * np.sin(theta_circle)
    axs[0].set_title('Color Wheel', size=20)
    axs[0].scatter(x_circle, y_circle, c=np.zeros((len(x_circle), 3)), s=0.1)
    axs[0].set_xlim([-255, 255])
    axs[0].set_ylim([-255, 255])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].axis("off")

    # --- RIGHT: Mean Angle Representation ---
    axs[1].set_title(f'Mean Angle Representation: {hue_degree:.2f}°', size=20)
    axs[1].scatter(x_coordinates_mean, y_coordinates_mean, c=colors2 / 255)
    axs[1].scatter(x_circle, y_circle, c=np.zeros((len(x_circle), 3)), s=0.1)
    axs[1].set_xlim([-255, 255])
    axs[1].set_ylim([-255, 255])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(result_path, dpi=300)
    plt.close()

def display_original_and_variants(folder_path, save_path):

    # Find matching files by suffix
    files = os.listdir(folder_path)

    def find_image(suffix):
        for f in files:
            if f.endswith(suffix):
                return os.path.join(folder_path, f)
        raise FileNotFoundError(f"No file ending with {suffix}")

    # Load images
    original_img = mpimg.imread(find_image("original.png"))

    variant_suffixes = ["L1C1.png", "L1C2.png", "L2C1.png", "L2C2.png"]
    variants = [mpimg.imread(find_image(suf)) for suf in variant_suffixes]

    # Create figure
    fig = plt.figure(figsize=(14, 6))

    # ORIGINAL (left)
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(original_img)
    ax0.set_title("Original", fontsize=16)
    ax0.axis("off")

    # Placeholder right panel
    grid_ax = fig.add_subplot(1, 3, (2, 3))
    grid_ax.axis("off")

    # Create 2x2 sub-grid
    gs = grid_ax.get_subplotspec().subgridspec(2, 2)

    for i, img in enumerate(variants):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.set_title(variant_suffixes[i].replace(".png", ""), fontsize=13)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
