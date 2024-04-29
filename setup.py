from setuptools import setup

APP = ['main.py']  # Path to your main application file
DATA_FILES = []  # Include any additional data files here
OPTIONS = {
    'argv_emulation': True,
    'packages': ['numpy', 'pandas', 'matplotlib', 'statsmodels', 'sklearn', 'prophet', 'customtkinter', 'tkinter'],
    'excludes': ['PyInstaller'],  # Exclude PyInstaller if not actually used
}

setup(
    app=APP,
    name="TruCast",
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app']
)
