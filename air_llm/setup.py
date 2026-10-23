import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# A concise package README is bundled locally so building an extracted source
# distribution does not depend on a parent repository checkout.
long_description = ""
for _readme in (os.path.join(here, "README.md"), os.path.join(here, os.pardir, "README.md")):
    if os.path.exists(_readme):
        with open(_readme, "r", encoding="utf-8") as fh:
            long_description = fh.read()
        break

setuptools.setup(
    name="airllm",
    version="5.0.0a1",
    author="Gavin Li",
    author_email="gavinli@animaai.cloud",
    description="Local out-of-core inference with selective reads, bounded caches and native MLX backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iliasm-a11y/airllm-next",
    license="Apache-2.0",
    python_requires='>=3.12',
    packages=setuptools.find_packages(exclude=('tests','tests.*')),
    license_files=['LICENSE','NOTICE_NEXT.txt','NOTICE_ULTRA.txt'],
    # Keep the dependency surface small and pinned to a tested range so a plain
    # `pip install airllm` gives users a known-good stack with no manual upgrades.
    install_requires=[
        'tqdm',
        'torch>=2.4',
        'transformers>=4.49,<6',
        'accelerate>=1.0',
        'safetensors',
        'huggingface-hub',
        'scipy',
        'sentencepiece',
        'psutil>=5.9',
        'numpy>=2.2,<3',
        # 'bitsandbytes' is optional (used only for --compression); we fall back gracefully when absent.
        # 'compressed-tensors' is optional too: only checkpoints stored in that format (Kimi K3's
        # MXFP4 weights) need it, and transformers raises a clear error naming it when it is missing.
    ],
    extras_require={
        'ultra': ['transformers==5.20.2', 'safetensors>=0.8.0'],
        'next': ['transformers==5.20.2','safetensors>=0.8.0',
                 'mlx==0.32.4; sys_platform == "darwin"', 'mlx[cpu]==0.32.4; sys_platform == "linux"'],
        'macos': ['mlx==0.32.4; sys_platform == "darwin" and platform_machine == "arm64"'],
        'test': ['pytest>=8'],
    },
    entry_points={'console_scripts': ['airllm-ultra=airllm.ultra.cli:main', 'airllm-next=airllm.nextgen.cli:main']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
