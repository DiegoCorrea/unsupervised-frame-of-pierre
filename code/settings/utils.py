
from psutil import virtual_memory


def get_size(mbytes=virtual_memory().total, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if mbytes < factor:
            return f"{mbytes:.2f}{unit}{suffix}"
        mbytes /= factor
