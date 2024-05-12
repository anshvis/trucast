# MODEL CODE
import time
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# path = C:\Users\chris\Documents\GitHub\trucast\prophet


EXCEL_FILE_PATH = 'CBO Revenue Short.xlsx' # pre-GUI testing
OUTPUT_FILE_PATH = 'TruCast Output 3 Month.xlsx' # pre-GUI testing
MEDIUM_THRESHOLD = 200000 # in dollars
LARGE_THRESHOLD = 533000 # in dollars
NUMBER_OF_MONTHS = 24 # in months

def fixed(revenue_series):
    return [revenue_series.iloc[-1]]*NUMBER_OF_MONTHS

def three_month(revenue_series):
    three_month_proj = pd.Series([revenue_series.iloc[-3], revenue_series.iloc[-2], revenue_series.iloc[-1]])
    for i in range(3, NUMBER_OF_MONTHS+3):
        number_of_non_nan_points = three_month_proj[i-3:].count()
        temp_series = three_month_proj.fillna(0)
        next_month = (temp_series.iloc[-3] + temp_series.iloc[-2] + temp_series.iloc[-1]) / number_of_non_nan_points
        three_month_proj = pd.concat([three_month_proj, pd.Series([next_month])], ignore_index=True)
    
    three_month_proj = three_month_proj[3:]
    return three_month_proj

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
    p_range = 4  #put 
    q_range = 4  #put 
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
    
def determine_projection_type(revenue_series, medium_threshold, large_threshold):
    THREE_MONTH_AVERAGE_DATA_CUTOFF = 6

    if (revenue_series.iloc[-2] == revenue_series.iloc[-1]) or (np.isnan(revenue_series.iloc[-1])):
        return 'fixed'
    elif revenue_series.iloc[-NUMBER_OF_MONTHS:].count() >= THREE_MONTH_AVERAGE_DATA_CUTOFF:
        # if there are less than 12 but more than 6 monthds of data, then find yearly revenue
        number_of_month_data_points = revenue_series.iloc[-NUMBER_OF_MONTHS:].count()
        yearly_revenue = (sum(revenue_series.iloc[-number_of_month_data_points:])/number_of_month_data_points)*12
    else:
        # if there isn't a years worth of data just return a 3 month average
        return 'three_month'
    #MODIFIED FOR TESTING
    if yearly_revenue > medium_threshold:
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
    
def process(input_path, export_path): # export_path is deprecate
    if input_path.endswith('.xls') or input_path.endswith('.xlsx'):
        #revenue_data = pd.read_excel(EXCEL_FILE_PATH, index_col=['Site', 'Customer Code'])
        revenue_data = pd.read_excel(input_path, index_col='site')
    else:
        print("Error: Input file must be in Excel format.")

    # Create dataframe for the output date
    new_months = list(pd.date_range(revenue_data.columns[-1], periods=NUMBER_OF_MONTHS+1, freq='M').strftime('%Y-%m'))[1:NUMBER_OF_MONTHS+1]
    revenue_data.columns = [col.strftime('%Y-%m') if isinstance(col, pd.Timestamp) else col for col in revenue_data.columns]
    column_names = list(revenue_data.columns) + new_months
    revenue_data.index = rename_duplicates(revenue_data.index)
    output_data = pd.DataFrame(index=revenue_data.index, columns=column_names)

    hospital_model_df = pd.DataFrame(columns=['Hospital', 'Model'])

    #Loop through hospitals
    for hospital, revenue_series in revenue_data.iterrows():
        projection_type = determine_projection_type(revenue_series, MEDIUM_THRESHOLD, LARGE_THRESHOLD)
        # append hospitl and projection type to df
        hospital_model_df = hospital_model_df.append({'Hospital': f'{hospital}', 'Model': f'{projection_type}'}, ignore_index=True)
        if projection_type == 'fixed':
            projection = fixed(revenue_series)
        elif projection_type == 'three_month':
            projection = three_month(revenue_series)
        elif projection_type == 'arima':
            try:
                projection = arima(revenue_series)
            except:
                projection = three_month(revenue_series)
        elif projection_type == 'prophet':
            try:
                (projection, prophet_low, prophet_high) = prophet(revenue_series)
            except:
                projection = three_month(revenue_series)

        projection = [rev if rev >= 0 or np.isnan(rev) else 0 for rev in projection] # Sets a lower bound of 0 for any projection
        
        output_data.loc[hospital] = pd.concat([revenue_series, pd.Series(data=projection, index=new_months)])
    
    output_data.dropna(inplace=True, how='all')
    hospital_model_df.dropna(inplace=True, how='all')
    # output_data.to_excel(export_path, index_label='site') # deprecated\
    input_file_name = input_path.split('/')[-1]
    hospital_model_output_path = export_path+ '/TruCast_Models_Used' + input_file_name
    hospital_model_df.to_excel(export_path, index_label='site')
    
    return output_data



# GUI CODE
import customtkinter as ctk
from tkinter import filedialog, ttk
import tkinter as tk
import threading
import time
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt 

ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "green" (default), "blue", "dark-blue"

class MainApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('TruCast')
        self.geometry('700x500')
        self.graph_window = None

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
            # new_export_path = self.export_path_entry.get() + "/TruCast_Output.xlsx"
            
            # Initialize the progress bar
            self.progress_bar = ttk.Progressbar(self.right_frame, orient='horizontal', 
                                                mode='determinate', length=280)
            self.progress_bar.place(relx=0.5, rely=0.95, anchor='center', relwidth=0.95)
            self.progress_bar['maximum'] = 500  # Assuming the model takes approximately 5 minutes
            self.progress_bar['value'] = 0  # Initialize the progress value
            
            # Start the process in a new thread
            self.process_thread = threading.Thread(target=self.run_process, 
                                                args=(self.filepath_entry.get(), self.export_path_entry.get()))
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
                self.close_progress(None, None)
                return
            time.sleep(1)  # Wait for one second
            self.progress_bar['value'] += 1  # Update the progress bar's value
            self.update()  # Refresh the progress bar's visual representation

    def run_process(self, filepath, export_path):
        # Your processing function
        output = process(filepath, export_path) # passing new_export_path to process is deprecated
        # Update the GUI after the processing is done
        self.after(100, self.close_progress, output, export_path)

    def close_progress(self, output, export_path):
        # Update GUI elements safely on the main thread
        self.after(0, self.update_gui_after_processing, output, export_path)

    def update_gui_after_processing(self, output, export_path):
        # Safe update of GUI elements after processing
        if self.progress_bar:
            self.progress_bar['value'] = self.progress_bar['maximum']
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

        if output is not None and not output.empty:
            self.graph_window = GraphsApp(output,self.filepath_entry.get(),export_path)
            self.graph_window.mainloop() 


class GraphsApp(ctk.CTk):
    def __init__(self, output_data,import_path, export_path): # output data always NUMBER_OF_MONTHS projected columns added on
        super().__init__()
        self.title('Revenue Forecasting')
        self.geometry('1200x600')  # Adjusted to fit both graph and side controls

        # Initialize important data
        self.output_data = output_data
        self.export_path = export_path
        self.import_path = import_path

        # Create a main frame to hold the graph
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(side='left', fill='both', expand=True)

        # Create a control frame to hold the date selection widgets
        control_frame = ctk.CTkFrame(self, width=300)  # Fixed width for control panel
        control_frame.pack(side='right', fill='y')

        # Initialize the matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas_widget.get_tk_widget().pack(side='top', fill='both', expand=True)

        # Initialize graph
        self.initialize_graph()

        # Add date selection widgets to the right side
        self.add_date_selection_widgets(control_frame)

        # intialize these to very first and last month
        self.start_date_var.set(self.output_data.columns[1]) 
        self.end_date_var.set(self.output_data.columns[-1])

    def initialize_graph(self, start_date=None, end_date=None):
        try:
            self.ax.clear()

            # Prepare data for plotting
            if 'site' in self.output_data.columns:
                data_for_plotting = self.output_data.drop(columns=['site'])  # Drop non-date column
            else:
                data_for_plotting = self.output_data.copy()

            # Convert column names to datetime
            data_for_plotting.columns = pd.to_datetime(data_for_plotting.columns)

            # Filter based on the provided date range
            if start_date and end_date:
                data_for_plotting = data_for_plotting.loc[:, start_date:end_date]

            # Sum up all rows for each date to get total revenue per date
            revenue_data = data_for_plotting.sum()

            # Define the range for projected revenue (adjust as needed)
            projected_start = pd.to_datetime(self.output_data.columns[-1 - NUMBER_OF_MONTHS + 1]) # projected start should be the last month - NUMBER OF MONTHS
            projected_end = pd.to_datetime(self.output_data.columns[-1]) # projected ends should be the last month
            projected_range = pd.date_range(start=projected_start, end=projected_end, freq='MS')

            # Identify which projected dates are within the current data
            included_projected_dates = [date for date in projected_range if date in revenue_data.index]

            # Separate actual and projected revenue data
            actual_revenue_data = revenue_data[~revenue_data.index.isin(included_projected_dates)]
            projected_revenue_data = revenue_data[revenue_data.index.isin(included_projected_dates)]

            # Plot the actual and projected revenue
            if not actual_revenue_data.empty:
                self.ax.plot(actual_revenue_data.index, actual_revenue_data.values / 1e6, 'green', label='Actual Revenue')

            if not projected_revenue_data.empty:
                self.ax.plot(projected_revenue_data.index, projected_revenue_data.values / 1e6, 'limegreen', linestyle='dotted', label='Projected Revenue')

            # Add labels to every other point in the actual data with dynamic offset
            values = actual_revenue_data.values / 1e6
            for index, (date, value) in enumerate(zip(actual_revenue_data.index, values)):
                if index % 2 == 0:  # Label every other point
                    offset = 0.1 if (values[index] - values[index - 1]) > 0 else -0.1 if index > 0 else 0.1
                    self.ax.text(
                        date, value + offset,
                        f"{value:.2f}",
                        fontsize=8,
                        ha='center',
                        va='bottom' if offset > 0 else 'top',
                        color='green'
                    )

            values_proj = projected_revenue_data.values / 1e6
            for index, (date, value) in enumerate(zip(projected_revenue_data.index, values_proj)):
                if index % 2 == 0:  # Label every other point
                    offset = 0.1 if (values[index] - values[index - 1]) > 0 else -0.1 if index > 0 else 0.1
                    self.ax.text(
                        date, value + offset,
                        f"{value:.2f}",
                        fontsize=8,
                        ha='center',
                        va='bottom' if offset > 0 else 'top',
                        color='limegreen'
                    )

            # Set chart title and labels
            self.ax.set(title='Total Revenue Over Selected Period', xlabel='Date', ylabel='Revenue (Millions)')
            self.ax.legend()

            # Adjust x-axis ticks for better readability
            self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            self.canvas_widget.draw()
        except Exception as e:
            print("Failed to initialize graph:", e)

    def add_date_selection_widgets(self, control_frame):
        # Create and pack date selection dropdowns and a regraph button within the control_frame
        self.start_date_var = ctk.StringVar()
        self.end_date_var = ctk.StringVar()

        if 'site' in self.output_data.columns:
            data_for_plotting = self.output_data.drop(columns=['site'])
        else:
            data_for_plotting = self.output_data.copy()
        data_for_plotting.columns = pd.to_datetime(data_for_plotting.columns)
        date_options = data_for_plotting.columns.strftime('%Y-%m-%d').tolist()

        # Label and dropdown for the start date
        start_date_label = ctk.CTkLabel(control_frame, text="Start Date:")
        start_date_label.pack(pady=(50, 0), padx=20, anchor='n')
        start_date_dropdown = ctk.CTkOptionMenu(control_frame, variable=self.start_date_var, values=date_options)
        start_date_dropdown.pack(pady=(0, 10), padx=20, anchor='n')
        self.start_date_var.set(date_options[0])  # Default to the first date

        # Label and dropdown for the end date
        end_date_label = ctk.CTkLabel(control_frame, text="End Date:")
        end_date_label.pack(pady=(10, 0), padx=20, anchor='n')
        end_date_dropdown = ctk.CTkOptionMenu(control_frame, variable=self.end_date_var, values=date_options)
        end_date_dropdown.pack(pady=(0, 10), padx=20, anchor='n')
        self.end_date_var.set(date_options[-1])  # Default to the last date

        # Button to regraph based on the selected dates
        regraph_button = ctk.CTkButton(control_frame, text='Regraph', command=self.regraph_based_on_selection)
        regraph_button.pack(pady=(20, 40), padx=20, anchor='n')

        # Button to export data to Excel
        export_button = ctk.CTkButton(control_frame, text='Export to Excel', command=self.export_to_excel)
        export_button.pack(pady=20, padx=20, anchor='n')

    def regraph_based_on_selection(self):
        start_date = pd.to_datetime(self.start_date_var.get())
        end_date = pd.to_datetime(self.end_date_var.get())
        self.ax.clear()
        self.initialize_graph(start_date=start_date, end_date=end_date)
        self.canvas_widget.draw()

    def export_to_excel(self):
        try:
            # Convert start and end dates from strings to datetime objects
            start_date_str = self.start_date_var.get()
            end_date_str = self.end_date_var.get()
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)

            # Prepare data for export, dropping 'site' only temporarily for date operations
            if 'site' in self.output_data.columns:
                data_for_export = self.output_data.drop(columns=['site'])
                site_column = self.output_data['site']  # Preserve the 'site' column separately
            else:
                data_for_export = self.output_data.copy()
                site_column = None

            # Ensure column names are datetime objects for comparison
            data_for_export.columns = pd.to_datetime(data_for_export.columns)

            # Filter columns within the date range
            filtered_data = data_for_export.loc[:, (data_for_export.columns >= start_date) & (data_for_export.columns <= end_date)]

            # Convert the filtered column names to the desired string format '%Y-%m'
            filtered_data.columns = [col.strftime('%Y-%m') for col in filtered_data.columns]

            # Re-integrate the 'site' column if it was originally present
            if site_column is not None:
                filtered_data.insert(0, 'site', site_column)  # Insert at position 0 to maintain order
            print(self.import_path)
            # Export the filtered data to Excel
            output_filename = f"{self.export_path}/TruCast_OUTPUT_{self.import_path.split('/')[-1].split('.')[0]}_({filtered_data.columns[0]}_{filtered_data.columns[-1]}).xlsx"
            filtered_data.to_excel(output_filename, index_label='site')
            print(f"Data exported successfully to {output_filename}")
        except Exception as e:
            print("Failed to export data:", e)




if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
