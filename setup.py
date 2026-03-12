from setuptools import setup, find_packages

setup(
    name='motionclip',
    version='1.0.0',
    description='MotionCLIP: Text-Driven Human Motion Generation',
    author='MotionCLIP Authors',
    packages=find_packages(),
    python_requires='>=3.8',
    setup_requires=['setuptools<83'],
    install_requires=[
        'torch>=1.7.0',
        'torchvision',
        'numpy',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'joblib',
        'smplx',
        'gdown',
        'tqdm',
        'pyyaml',
        'omegaconf',
        'chumpy-fork',
        'tensorboard',
        'imageio',
        'pandas',
        'human_body_prior',
        'clip @ git+https://github.com/openai/CLIP.git',
    ],
    extras_require={
        'dev': [
            'ipdb',
            'ipython',
        ],
    },
)
