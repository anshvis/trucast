import cx_Freeze, sys
base = None
print(sys.platform)

if sys.platform == 'win32':
    base = "Win32GUI"

executables = [cx_Freeze.Executable("main.py", base=base, target_name="trucast.exe")]

cx_Freeze.setup(
    name="trucast.exe",
    options={"build_exe": {"packages": ["tkinter", "time", "numpy", "pandas", 
                                        "matplotlib", "statsmodels", "sklearn", 
                                        "prophet", "customtkinter", "threading", 
                                        "tkinter"]}}, 
    version="1.0",
    description="Trucast predicts financial forecasting for TrueBridge Inc.",
    executables=executables

)