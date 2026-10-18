"""Additional camera/sensor nuisances with shared Anchor/Causal context."""

import numpy as np
from PIL import Image, ImageFilter


def motion_blur(image, length):
    kernel = np.zeros((5, 5), dtype=float)
    kernel[2, 2 - length // 2 : 3 + length // 2] = 1
    return image.filter(
        ImageFilter.Kernel((5, 5), kernel.ravel().tolist(), scale=kernel.sum())
    )


def chromatic_shift(image, pixels):
    arr = np.asarray(image.convert("RGB")).copy()
    # Edge replication avoids wrapping unrelated content across image borders.
    if pixels:
        arr[:, pixels:, 0] = arr[:, :-pixels, 0]
        arr[:, :pixels, 0] = arr[:, pixels : pixels + 1, 0]
        arr[:, :-pixels, 2] = arr[:, pixels:, 2]
        arr[:, -pixels:, 2] = arr[:, -pixels - 1 : -pixels, 2]
    return Image.fromarray(arr)


def sensor_banding(image, amplitude):
    arr = np.asarray(image.convert("RGB"), dtype=float)
    rows = 1 + amplitude * np.sin(2 * np.pi * np.arange(arr.shape[0]) / 41)
    return Image.fromarray(np.clip(arr * rows[:, None, None], 0, 255).astype("uint8"))


def shot_noise(image, photons, seed=20261019):
    arr = np.asarray(image.convert("RGB"), dtype=float) / 255
    # A shared standardized field keeps non-target pixels matched across A/C.
    # Direct Poisson sampling consumes a variable number of RNG draws per pixel.
    field = np.random.default_rng(seed).normal(size=arr.shape)
    values = arr + np.sqrt(arr / photons) * field
    return Image.fromarray(np.clip(values * 255, 0, 255).astype("uint8"))


def digital_view_scale(image, scale):
    """Change apparent scale without cropping any decision-critical evidence."""
    result = Image.new("RGB", image.size, (127, 127, 127))
    size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    result.paste(resized, ((image.width - size[0]) // 2, (image.height - size[1]) // 2))
    return result


EXTRA_PROFILES = {
    "motion_blur": (
        lambda im: motion_blur(im, 1),
        lambda im: motion_blur(im, 5),
        {"context_kernel_length": 1, "nuisance_kernel_length": 5},
    ),
    "chromatic_aberration": (
        lambda im: chromatic_shift(im, 1),
        lambda im: chromatic_shift(im, 4),
        {"context_channel_shift_pixels": 1, "nuisance_channel_shift_pixels": 4},
    ),
    "sensor_banding": (
        lambda im: sensor_banding(im, 0.015),
        lambda im: sensor_banding(im, 0.13),
        {"context_amplitude": 0.015, "nuisance_amplitude": 0.13, "period_pixels": 41},
    ),
    "shot_noise": (
        lambda im: shot_noise(im, 4096),
        lambda im: shot_noise(im, 256),
        {
            "context_photons": 4096,
            "nuisance_photons": 256,
            "seed": 20261019,
            "model": "normal_approximation_to_poisson",
            "common_standardized_noise": True,
        },
    ),
    "digital_view_scale": (
        lambda im: digital_view_scale(im, 0.98),
        lambda im: digital_view_scale(im, 0.88),
        {"context_scale": 0.98, "nuisance_scale": 0.88, "padding_rgb": [127, 127, 127]},
    ),
}
