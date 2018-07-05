#nsml: dongkyu/pytorch-opencv:latest
from distutils.core import setup
setup(
    name='nsml WMNet',
    version='1.0',
    description='ns-ml',
    install_requires=[
        'visdom==0.1.8.3',
        'numpy',
        'pillow',
        'scikit-image',
        'scikit-learn'
    ]
)
