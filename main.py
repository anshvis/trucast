import customtkinter as ctk
from tkinter import filedialog, Canvas
import TruCast



# CustomTkinter settings
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "green" (default), "blue", "dark-blue"

class MainApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('TruCast')
        self.geometry('700x500')

        # Left side panel
        left_frame = ctk.CTkFrame(master=self, width=430, corner_radius=0)
        left_frame.pack(side='left', fill='y')

        welcome_label = ctk.CTkLabel(master=left_frame, text="Welcome to TruCast!", font=("Roboto Medium", -35))
        welcome_label.pack(pady=80, padx=20)

        description_label = ctk.CTkLabel(master=left_frame,
                                          text="TruCast is an application built to help forecast\n"
                                               "revenue for TruBridge. It utilizes modern\n"
                                               "statistical and machine learning models to\n"
                                               "predict revenue, such as ARIMA and Facebook's\n"
                                               "PROPHET model.",
                                          font=("Roboto", -16),
                                          justify='left',
                                          anchor='w',
                                          width=380)
        description_label.pack(pady=20, padx=20)

        # Right side panel
        right_frame = ctk.CTkFrame(master=self)
        right_frame.pack(side='right', fill='both', expand=True)

        import_label = ctk.CTkLabel(master=right_frame, text="Import financial data", font=("Roboto Medium", -25))
        import_label.pack(pady=30)

        self.filepath_entry = ctk.CTkEntry(master=right_frame, placeholder_text="Choose file...")
        self.filepath_entry.pack(pady=10, padx=50)

        import_button = ctk.CTkButton(master=right_frame, text="Import file", command=self.import_file)
        import_button.pack(pady=10)

        self.export_path_entry = ctk.CTkEntry(master=right_frame, placeholder_text="Export path")
        self.export_path_entry.pack(pady=10, padx=50)

        export_button = ctk.CTkButton(master=right_frame, text="Export path", command=self.export_path)
        export_button.pack(pady=10)

        start_button = ctk.CTkButton(master=right_frame, text="Start", command=self.start_backend_process)
        start_button.pack(pady=30)

    def import_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.filepath_entry.delete(0, 'end')  # Clear the existing content
            self.filepath_entry.insert(0, filepath)  # Insert the new file path

    def export_path(self):
        directory = filedialog.askdirectory()
        if directory:
            self.export_path_entry.delete(0, 'end')  # Clear the existing content
            self.export_path_entry.insert(0, directory)  # Insert the new directory


    def start_backend_process(self):
        # Check if inputs are valid (i.e., file path and export path are not empty)
        if self.filepath_entry.get() and self.export_path_entry.get() and (self.filepath_entry.get().endswith('.xls') or self.filepath_entry.get().endswith('.xlsx')):
            new_export_path = self.export_path_entry.get() + "/TruCast_Output.xlsx"
            output = TruCast.process(self.filepath_entry.get(), new_export_path)
            self.destroy()  # Close the current window
            app = GraphsApp()  # Start the graphs interface
            app.mainloop()
        else:
            print("Please select both an input file and an output directory.")

class GraphsApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('Revenue Forecasting')
        self.geometry('800x600')

        # Header label
        header_label = ctk.CTkLabel(master=self, text="Revenue Forecasting", font=("Roboto", 16), fg_color="#04B540", text_color="#FFFFFF")
        header_label.pack(fill='x', pady=10)

        # Button frame
        button_frame = ctk.CTkFrame(master=self)
        button_frame.pack(pady=20, padx=20)

        # Buttons for Prophet, ARIMA, and 3-Month Avg
        prophet_button = ctk.CTkButton(master=button_frame, text="Prophet", command=lambda: self.switch_panel("prophet"))
        prophet_button.grid(row=0, column=0, padx=10)

        arima_button = ctk.CTkButton(master=button_frame, text="ARIMA", command=lambda: self.switch_panel("arima"))
        arima_button.grid(row=0, column=1, padx=10)

        avg_button = ctk.CTkButton(master=button_frame, text="3-Month Avg.", command=lambda: self.switch_panel("avg"))
        avg_button.grid(row=0, column=2, padx=10)

        # Graph panel
        self.graph_panel = Canvas(master=self, bg="#f0f0f0", height=400, width=600)
        self.graph_panel.pack(pady=20, padx=20)

    def switch_panel(self, model):
        print(f"Switched to {model} panel")
        # Here, the canvas color change is a placeholder. Integrate your actual plotting logic here.
        self.graph_panel.config(bg="#f0f0f0" if model != "avg" else "#d0d0d0")

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
