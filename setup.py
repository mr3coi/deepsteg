#nsml: dongkyu/pytorch-opencv:latest
from distutils.core import setup
setup(
    name='nsml WMNet',
    version='1.0',
    description='ns-ml',
    install_requires=[
        'visdom',
        'numpy',
        'pillow',
        'scikit-image',
        'scikit-learn'
    ]
)
