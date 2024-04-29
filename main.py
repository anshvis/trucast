# MODEL CODE
import os
import argparse
import numpy as np
import time
from numpy.linalg import LinAlgError
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from prophet import Prophet


EXCEL_FILE_PATH = 'CBO Revenue Short.xlsx'
OUTPUT_FILE_PATH = 'TruCast Output 3 Month.xlsx'
MEDIUM_THRESHOLD = 200000 # in dollars
LARGE_THRESHOLD = 533000 # in dollars
NUMBER_OF_MONTHS = 12 # in months

def fixed(revenue_series):
    return [revenue_series.iloc[-1]]*NUMBER_OF_MONTHS

def three_month(revenue_series):
    return [(revenue_series.iloc[-3] + revenue_series.iloc[-2] + revenue_series.iloc[-1])/3]*NUMBER_OF_MONTHS

def ARIMA_rolling_forcast_origin(revenue_series, number_of_predicted_months, p, q, d):
    # Preforms a rolling forcast origin for an using an arima model on times series data for a set number of number_of_predicted_months < len(revenue_series) 
    arima_revenue_projection_list = revenue_series.tolist()
    total_revenue_by_month_list = revenue_series.tolist()
    for index, revenue in enumerate(arima_revenue_projection_list):
        if len(arima_revenue_projection_list) - index < number_of_predicted_months:
            try:
                model = ARIMA(total_revenue_by_month_list[0:index], order=(p, d, q))
                try:
                    results = model.fit()
                    forecast = results.forecast(steps=1)
                except LinAlgError as e:
                    forecast = [-10000000]
                arima_revenue_projection_list[index] = forecast[0]
            except LinAlgError as e:
                arima_revenue_projection_list[index] = np.nan
                continue
    arima_revenue_projection = pd.Series(data=arima_revenue_projection_list, index=revenue_series.index)
    
    return arima_revenue_projection

def arima(revenue_series):
    # Train on 1/3 of the data
    number_of_predicted_months = int(len(revenue_series)/3)

    # Choose the range of p and q that you want to optmize over
    p_range = 6
    q_range = 6
    d = 1

    # Create an ARIMA model on the data for each value of p and q forcast it forward using a rolling origin forcast, determine which pair of p and q works
    # best and output that
    arima_revenue_projection_list = revenue_series.tolist()
    projection_percent_difference = {}
    for i in range(1, p_range):
        for ii in range(1, q_range):
            projection = ARIMA_rolling_forcast_origin(revenue_series, number_of_predicted_months, i, ii, d)

            percent_difference = abs(projection[-number_of_predicted_months:] - arima_revenue_projection_list[-number_of_predicted_months:]) / arima_revenue_projection_list[-number_of_predicted_months:] * 100
            projection_avg_percent_difference = percent_difference.mean()

            if projection_avg_percent_difference == np.nan:
                # If any errors occur set the percent error to really high so it doesn't get choosen
                projection_percent_difference[(i, ii)] = 1000
            else:
                projection_percent_difference[(i, ii)] = projection_avg_percent_difference

    min_error_key = min(projection_percent_difference, key=projection_percent_difference.get)
    p = min_error_key[0]
    q = min_error_key[1]

    # Use the ARIMA model that produced the minimum error and forcast 1 time step forward
    model = ARIMA(revenue_series, order=(p, d, q))
    results = model.fit()
    forecast = results.forecast(steps=NUMBER_OF_MONTHS)

    return forecast

def prophet(revenue_series):
    # for the model, make a dataframe with columns 'ds' and 'y' out of the hospital row, which was a Series
    curr_hospital = pd.DataFrame({'ds':revenue_series.index, 'y':revenue_series.values})

    # make the months datetime objects, and the revenues numbers
    curr_hospital['ds'] = pd.to_datetime(curr_hospital['ds'])
    curr_hospital['y'] = pd.to_numeric(curr_hospital['y'])

    # training on all revenues up to the last year (CAN BE MODIFIED TO BE THE LAST MONTH, TWO MONTHS, ETC.)
    # MUDIT COMMENT: JUST GOT RID OF THE "-NUMBER_OF_MONTHS" HERE, NOW TRAINING ON ALL DATA IN REVENUE_SERIES
    train = curr_hospital.iloc[:len(curr_hospital)]

    # testing how accurately we predict the last year (CAN BE MODIFIED TO BE THE LAST MONTH, TWO MONTHS, ETC.)
    test = curr_hospital.iloc[len(curr_hospital) - NUMBER_OF_MONTHS:]

    # if this row of the dataframe contains less than two non-nan values, we cannot predict on it — SKIP
    if train[train['y'].notnull()].shape[0] < 2:
        return None # WE SHOULD REDIRECT HERE TO ANOTHER MODEL, IF THERE IS NOT ENOUGH TRAINING DATA FOR PROPHET
    
    # fit the model, generate the forecast
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods = NUMBER_OF_MONTHS, freq='MS')
    forecast = m.predict(future)

    # generate predictions (contains all from 2004 through 2023)
    predictions = forecast[['ds', 'yhat']].set_index('ds')['yhat'].iloc[-NUMBER_OF_MONTHS:]
    predictions_lower = forecast[['ds', 'yhat_lower']].set_index('ds')['yhat_lower'].iloc[-NUMBER_OF_MONTHS:]
    predictions_upper = forecast[['ds', 'yhat_upper']].set_index('ds')['yhat_upper'].iloc[-NUMBER_OF_MONTHS:]
    
    output_columns = predictions.index

    # return a tuple of the lower, point estimate, and upper bound for the last month
    return (predictions.values, predictions_lower.values, predictions_upper.values)

def determine_projection(revenue_series, medium_threshold, large_threshold, fixed_rate_projection, three_month_projection, arima_projection, prophet_projection):
    # if it's fixed, return fixed rate
    if (fixed_rate_projection != -1):
        return fixed_rate_projection
    
    if revenue_series.iloc[-NUMBER_OF_MONTHS:].count() == NUMBER_OF_MONTHS:
        yearly_revenue = sum(revenue_series.iloc[-NUMBER_OF_MONTHS:])
    else:
        yearly_revenue = (revenue_series.mean())*NUMBER_OF_MONTHS
    
    if yearly_revenue < medium_threshold:
        return arima_projection
    else:
        return prophet_projection
    
def determine_projection_type(revenue_series, medium_threshold, large_threshold):
    if (revenue_series.iloc[-2] == revenue_series.iloc[-1]) or (np.isnan(revenue_series.iloc[-1])):
        return 'fixed'
    elif revenue_series.iloc[-NUMBER_OF_MONTHS:].count() == NUMBER_OF_MONTHS:
        yearly_revenue = sum(revenue_series.iloc[-NUMBER_OF_MONTHS:])
    else:
        # if there isn't a years worth of data just return a 3 month average
        return 'three_month'
    
    if yearly_revenue < medium_threshold:
        return 'arima'
    else:
        return 'prophet'
    
def rename_duplicates(names):
    name_count = {}
    modified_names = []

    for name in names:
        if name in name_count:
            name_count[name] += 1
        else:
            name_count[name] = 1

        if name_count[name] == 1:
            modified_names.append(name)
        else:
            modified_names.append(f"{name}_{name_count[name]-1}")

    return modified_names
    
def process(input_path, export_path):
    if input_path.endswith('.xls') or input_path.endswith('.xlsx'):
        #revenue_data = pd.read_excel(EXCEL_FILE_PATH, index_col=['Site', 'Customer Code'])
        revenue_data = pd.read_excel(input_path, index_col='site')
    else:
        print("Error: Input file must be in Excel format.")

    # TODO Clean data


    # Create dataframe for the output date
    new_months = list(pd.date_range(revenue_data.columns[-1], periods=13, freq='M').strftime('%Y-%m'))[1:13]
    revenue_data.columns = [col.strftime('%Y-%m') if isinstance(col, pd.Timestamp) else col for col in revenue_data.columns]
    column_names = list(revenue_data.columns) + new_months
    revenue_data.index = rename_duplicates(revenue_data.index)
    output_data = pd.DataFrame(index=revenue_data.index, columns=column_names)

    #Loop through hospitals
    for hospital, revenue_series in revenue_data.iterrows():
        projection_type = determine_projection_type(revenue_series, MEDIUM_THRESHOLD, LARGE_THRESHOLD)
        print(f'{hospital}: {projection_type}')
        if projection_type == 'fixed':
            projection = fixed(revenue_series)
        elif projection_type == 'three_month':
            projection = three_month(revenue_series)
        elif projection_type == 'arima':
            try:
                projection = arima(revenue_series)
            except:
                projection = three_month(revenue_series)
            #projection = [0]*12
        elif projection_type == 'prophet':
            try:
                (projection, prophet_low, prophet_high) = prophet(revenue_series)
            except:
                projection = three_month(revenue_series)

        projection = [rev if rev >= 0 or np.isnan(rev) else 0 for rev in projection] # Sets a lower bound of 0 for any projection

        output_data.loc[hospital] = pd.concat([revenue_series, pd.Series(data=projection, index=new_months)])

    output_data.dropna(inplace=True, how='all')
    output_data.to_excel(export_path, index_label='site')
    return output_data



# GUI CODE
import customtkinter as ctk
from tkinter import ttk, filedialog
import threading
from tkinter import filedialog, Canvas

ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "green" (default), "blue", "dark-blue"

class MainApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('TruCast')
        self.geometry('700x500')

        # Left side panel
        self.left_frame = ctk.CTkFrame(master=self, width=430, corner_radius=0)
        self.left_frame.pack(side='left', fill='y')

        welcome_label = ctk.CTkLabel(master=self.left_frame, text="Welcome to TruCast!", font=("Roboto Medium", -35))
        welcome_label.pack(pady=80, padx=20)

        description_label = ctk.CTkLabel(master=self.left_frame,
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
        import_label.pack(pady=60)

        self.filepath_entry = ctk.CTkEntry(master=self.right_frame, placeholder_text="Choose file...")
        self.filepath_entry.pack(pady=10, padx=50)

        import_button = ctk.CTkButton(master=self.right_frame, text="Import file", command=self.import_file)
        import_button.pack(pady=10)

        self.export_path_entry = ctk.CTkEntry(master=self.right_frame, placeholder_text="Export path")
        self.export_path_entry.pack(pady=10, padx=50)

        export_button = ctk.CTkButton(master=self.right_frame, text="Export path", command=self.export_path)
        export_button.pack(pady=10)

        start_button = ctk.CTkButton(master=self.right_frame, text="Start", command=self.start_backend_process)
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
        if self.filepath_entry.get() and self.export_path_entry.get() and \
        (self.filepath_entry.get().endswith('.xls') or self.filepath_entry.get().endswith('.xlsx')):
            new_export_path = self.export_path_entry.get() + "/TruCast_Output.xlsx"
            
            # Initialize the progress bar
            self.progress_bar = ttk.Progressbar(self.right_frame, orient='horizontal', 
                                                mode='determinate', length=280)
            self.progress_bar.place(relx=0.5, rely=0.95, anchor='center', relwidth=0.95)
            self.progress_bar['maximum'] = 300  # Assuming the model takes approximately 5 minutes
            self.progress_bar['value'] = 0  # Initialize the progress value
            
            # Start the process in a new thread
            self.process_thread = threading.Thread(target=self.run_process, 
                                                args=(self.filepath_entry.get(), new_export_path))
            self.process_thread.start()

            # Start updating the progress bar in a new thread
            self.update_progress_bar()

        else:
            ctk.CTkLabel(self.right_frame, text="Please select both an input file (.xls or .xlsx) and an output directory.").pack()

    def update_progress_bar(self):
        # Update the progress bar every second
        for i in range(300):
            if not self.process_thread.is_alive():
                # If the process is done, break out of the loop
                self.close_progress(None)
                return
            time.sleep(1)  # Wait for one second
            self.progress_bar['value'] += 1  # Update the progress bar's value
            self.update()  # Refresh the progress bar's visual representation

    def run_process(self, filepath, new_export_path):
        # Your processing function
        output = process(filepath, new_export_path)
        # Update the GUI after the processing is done
        self.after(100, self.close_progress, output)

    def close_progress(self, output):
        if self.progress_bar:  # Check if the progress bar exists
            self.progress_bar['value'] = self.progress_bar['maximum']  # Set progress to full
            self.progress_bar.stop()
            self.progress_bar.pack_forget()  # Hide the progress bar
        if output:  # If there's output from the processing function, start the graphs interface
            app = GraphsApp(output)
            app.mainloop()

class GraphsApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('Revenue Forecasting')
        self.geometry('800x600')

        # Header label
        header_label = ctk.CTkLabel(master=self, text="Revenue Forecasting", font=("Roboto", 16), fg_color="#04B540", text_color="#FFFFFF")
        header_label.pack(fill='x', pady=10)

        # Button frame
        self.button_frame = ctk.CTkFrame(master=self)
        self.button_frame.pack(pady=20, padx=20)

    def initialize_graph(self):
        # This method will handle the plotting of the graph directly when the window is initialized.
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes.
        past_revenue = self.output_data.sum().values[0:-NUMBER_OF_MONTHS] / 1e6
        forecasted_revenue = self.output_data.sum().values[-NUMBER_OF_MONTHS:] / 1e6

        arima_button = ctk.CTkButton(master=self.button_frame, text="ARIMA", command=lambda: self.switch_panel("arima"))
        arima_button.grid(row=0, column=1, padx=10)

        avg_button = ctk.CTkButton(master=self.button_frame, text="3-Month Avg.", command=lambda: self.switch_panel("avg"))
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
