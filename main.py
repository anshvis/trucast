import customtkinter as ctk
from tkinter import filedialog

# CustomTkinter settings
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "green" (default), "blue", "dark-blue"

class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title('TruCast')
        self.geometry('700x500')
        self.configure(bg="#FFFFFF") 

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
        self.right_frame = ctk.CTkFrame(master=self)
        self.right_frame.pack(side='right', fill='both', expand=True)

        import_label = ctk.CTkLabel(master=self.right_frame, text="Import financial data", font=("Roboto Medium", -25))
        import_label.pack(pady=30)

        self.filepath_entry = ctk.CTkEntry(master=self.right_frame, placeholder_text="Choose file...")
        self.filepath_entry.pack(pady=10, padx=50)

        self.import_button = ctk.CTkButton(master=self.right_frame, text="Import file", command=self.import_file)
        self.import_button.pack(pady=10)

        self.export_path_entry = ctk.CTkEntry(master=self.right_frame, placeholder_text="Export path")
        self.export_path_entry.pack(pady=10, padx=50)

        self.export_button = ctk.CTkButton(master=self.right_frame, text="Export path", command=self.export_path)
        self.export_button.pack(pady=10)

        self.start_button = ctk.CTkButton(master=self.right_frame, text="Start", command=self.start_backend_process)
        self.start_button.pack(pady=30)

    def import_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.filepath_entry.set(filepath)
            # Add additional functionality if needed
    
    def export_path(self):
        directory = filedialog.askdirectory()
        if directory:
            self.export_path_entry.set(directory)
            # Add additional functionality if needed

    def start_backend_process(self):
        # Placeholder for backend code
        print("Backend process started...")

if __name__ == "__main__":
    app = App()
    app.mainloop()



