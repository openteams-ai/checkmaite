from upath import UPath

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_image_path(path: UPath) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES
