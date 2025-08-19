"""Setup script for Cython compilation with production optimizations."""
# pylint: disable=import-error,invalid-name

from __future__ import annotations

import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Production-optimized compilation flags
PRODUCTION_COMPILE_ARGS = [
    "-O3",  # Maximum optimization
    "-march=native",  # Optimize for local CPU
    "-ffast-math",  # Fast floating point math
    "-DNDEBUG",  # Disable debug assertions
    "-fomit-frame-pointer",  # Omit frame pointers for smaller size
    "-funroll-loops",  # Loop unrolling optimization
]

# Production-optimized linking flags
PRODUCTION_LINK_ARGS = [
    "-O3",
    "-s",  # Strip all symbol table and relocation info
    "-Wl,--strip-all",  # Additional stripping
    "-Wl,--as-needed",  # Only link needed libraries
]

# Debug vs Production build selection
DEBUG_BUILD = os.environ.get("CYTHON_DEBUG", "0") == "1"

if DEBUG_BUILD:
    compile_args = ["-O0", "-g", "-DDEBUG"]
    link_args = ["-g"]
    annotate = True
    profile = True
    linetrace = True
else:
    compile_args = PRODUCTION_COMPILE_ARGS
    link_args = PRODUCTION_LINK_ARGS
    annotate = False  # Don't generate .html files in production
    profile = False
    linetrace = False

extensions = [
    Extension(
        "shelly_speedwire_gateway.fast_calc",
        ["shelly_speedwire_gateway/fast_calc.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_WITHOUT_ASSERTIONS", None) if not DEBUG_BUILD else ("CYTHON_WITH_ASSERTIONS", None),
        ],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": DEBUG_BUILD,  # Only in debug builds
            "wraparound": DEBUG_BUILD,  # Only in debug builds
            "cdivision": True,  # Always use C division
            "language_level": 3,  # Python 3 syntax
            "profile": profile,  # Profiling support
            "linetrace": linetrace,  # Line tracing support
            "embedsignature": DEBUG_BUILD,  # Embed signatures only in debug
            "optimize.use_switch": True,  # Use switch for constant comparisons
            "optimize.unpack_method_calls": True,  # Unpack method calls
        },
        annotate=annotate,  # Generate HTML annotation files only if needed
        nthreads=os.cpu_count(),  # Use all available CPUs for compilation
    ),
    zip_safe=False,
)
