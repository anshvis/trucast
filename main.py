import customtkinter as ctk
from tkinter import filedialog, Canvas
import TruCast
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt



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
            app = GraphsApp(output)  # Start the graphs interface
            app.mainloop()
        else:
            print("Please select both an input file (.xls or .xlsx) and an output directory.")

class GraphsApp(ctk.CTk):
    def __init__(self, output_data):
        super().__init__()
        self.title('Revenue Forecasting')
        self.geometry('800x600')
        self.output_data = output_data  # This is the data you pass from the main app.
        header_label = ctk.CTkLabel(master=self, text="Revenue Forecasting", font=("Roboto", 16), fg_color="#04B540", text_color="#FFFFFF")
        header_label.pack(fill='x', pady=10)

        self.initialize_graph()

    def initialize_graph(self):
        # This method will handle the plotting of the graph directly when the window is initialized.
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes.
        NUMBER_OF_MONTHS = 12  # Adjust this as necessary.
        past_revenue = self.output_data.sum().values[0:-NUMBER_OF_MONTHS] / 1e6
        forecasted_revenue = self.output_data.sum().values[-NUMBER_OF_MONTHS:] / 1e6

        ax.plot(self.output_data.columns[0:-NUMBER_OF_MONTHS], past_revenue, 'bo-', label='Past Revenue')
        ax.plot(self.output_data.columns[-NUMBER_OF_MONTHS:], forecasted_revenue, 'ro--', label='Forecasted Revenue')

        ax.set(title='Previous Data and Projections', xlabel='Month', ylabel='Revenue (Millions)')
        ax.legend()

        # Rotate and set the frequency of x-axis labels
        ax.set_xticks(ax.get_xticks()[::3])  # Show every third tick label to reduce clutter.
        ax.set_xticklabels(self.output_data.columns[::3], rotation=90)  # Rotate tick labels to 90 degrees.

        # Embedding the matplotlib plot in the tkinter window.
        canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
