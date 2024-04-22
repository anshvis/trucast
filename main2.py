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

        # Main containers setup
        left_frame = ctk.CTkFrame(master=self, width=430, corner_radius=0, fg_color="#04B540")
        left_frame.pack(side='left', fill='both', expand=False)
        left_frame.pack_propagate(False)  # Prevents the frame from resizing to fit its contents

        right_frame = ctk.CTkFrame(master=self, width=270, corner_radius=0)
        right_frame.pack(side='left', fill='both', expand=True)
        right_frame.pack_propagate(False)

        # Left side content
        welcome_label = ctk.CTkLabel(master=left_frame, text="Welcome to TruCast!", 
                                     font=("Roboto Medium", -35), text_color="#FFFFFF", fg_color="#04B540")
        welcome_label.place(relx=0.5, rely=0.2, anchor=ctk.CENTER)

        description_label = ctk.CTkLabel(master=left_frame, 
                                          text="TruCast is an application built to help forecast\n"
                                               "revenue for TruBridge. It utilizes modern\n"
                                               "statistical and machine learning models to\n"
                                               "predict revenue, such as ARIMA and Facebook's\n"
                                               "PROPHET model.",
                                          font=("Roboto", -16), 
                                          justify='left', 
                                          text_color="#FFFFFF",
                                          fg_color="#04B540",
                                          width=380)
        description_label.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

        # Right side content
        import_label = ctk.CTkLabel(master=right_frame, text="Import financial data", font=("Roboto Medium", -25))
        import_label.place(relx=0.5, y=100, anchor=ctk.N)

        self.filepath_entry = ctk.CTkEntry(master=right_frame, placeholder_text="Choose file...")
        self.filepath_entry.place(relx=0.5, y=200, anchor=ctk.CENTER)

        self.import_button = ctk.CTkButton(master=right_frame, text="Import file", command=self.import_file)
        self.import_button.place(relx=0.5, y=240, anchor=ctk.CENTER)

        self.export_path_entry = ctk.CTkEntry(master=right_frame, placeholder_text="Export path")
        self.export_path_entry.place(relx=0.5, y=300, anchor=ctk.CENTER)

        self.export_button = ctk.CTkButton(master=right_frame, text="Export path", command=self.export_path)
        self.export_button.place(relx=0.5, y=340, anchor=ctk.CENTER)

        self.start_button = ctk.CTkButton(master=right_frame, text="Start", command=self.start_backend_process)
        self.start_button.place(relx=0.5, y=400, anchor=ctk.CENTER)

    def import_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.filepath_entry.delete(0, ctk.END)
            self.filepath_entry.insert(0, filepath)
    
    def export_path(self):
        directory = filedialog.askdirectory()
        if directory:
            self.export_path_entry.delete(0, ctk.END)
            self.export_path_entry.insert(0, directory)

    def start_backend_process(self):
        # Placeholder for backend code
        print("Backend process started...")

if __name__ == "__main__":
    app = App()
    app.mainloop()
