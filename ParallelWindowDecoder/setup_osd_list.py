from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent

extensions = [
    Extension(
        "osd_list",
        [str(ROOT / "osd_list.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="parallel_window_osd_list",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)

