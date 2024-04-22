import customtkinter as ctk
from tkinter import Canvas

# CustomTkinter settings
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "green" (default), "blue", "dark-blue"

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('Revenue Forecasting')
        self.geometry('800x600')

        # Header label
        header_label = ctk.CTkLabel(master=self, text="Revenue Forecasting", 
                                    font=("Roboto", 16), fg_color="#04B540", text_color="#FFFFFF")
        header_label.pack(fill='x', pady=10)

        # Button frame
        button_frame = ctk.CTkFrame(master=self)
        button_frame.pack(pady=20, padx=20)

        # Buttons for Prophet, ARIMA, and 3-Month Avg
        self.prophet_button = ctk.CTkButton(master=button_frame, text="Prophet", 
                                            command=lambda: self.switch_panel("prophet"))
        self.prophet_button.grid(row=0, column=0, padx=10)

        self.arima_button = ctk.CTkButton(master=button_frame, text="ARIMA", 
                                          command=lambda: self.switch_panel("arima"))
        self.arima_button.grid(row=0, column=1, padx=10)

        self.avg_button = ctk.CTkButton(master=button_frame, text="3-Month Avg.", 
                                        command=lambda: self.switch_panel("avg"))
        self.avg_button.grid(row=0, column=2, padx=10)

        # Graph panel (placeholder for your graphs)
        self.graph_panel = Canvas(master=self, bg="#f0f0f0", height=400, width=600)
        self.graph_panel.pack(pady=20, padx=20)

        # Initially set the Prophet panel
        self.switch_panel("prophet")

    def switch_panel(self, model):
        # This function will switch the graph panels
        print(f"Switched to {model} panel")  # Placeholder for actual functionality
        # Update buttons state based on current panel
        for button in [self.prophet_button, self.arima_button, self.avg_button]:
            button.configure(fg_color="#D5D5D5")  # Default color for inactive buttons

        if model == "prophet":
            self.prophet_button.configure(fg_color="#04B540")  # Active button color
        elif model == "arima":
            self.arima_button.configure(fg_color="#04B540")  # Active button color
        elif model == "avg":
            self.avg_button.configure(fg_color="#04B540")  # Active button color

        # Here you would also update the graph_panel to display the appropriate content.
        # For now, let's just change the canvas color.
        self.graph_panel.config(bg="#f0f0f0" if model != "avg" else "#d0d0d0")

if __name__ == "__main__":
    app = App()
    app.mainloop()
