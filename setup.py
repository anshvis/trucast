from setuptools import setup

APP = ['main.py']  # Path to your main application file
OPTIONS = {
    'argv_emulation': True,
}

setup(
    app=APP,
    name="TruCast",
    options={'py2app': OPTIONS},
    setup_requires=['py2app']
)
