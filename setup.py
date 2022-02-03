from setuptools import setup

setup(
    name='spectrogram',
    version='0.0.1',
    packages=['spectrogram'],
    python_requires='>=3.6,<4.0',
    install_requires=['numpy>=1.22,<2', 'matplotlib>=3.5,<4', 'progressbar2>=4.0,<5'],
    entry_points={
        'console_scripts': [
            'spectrogram=spectrogram.spectrogram:main',
        ],
    }
)
