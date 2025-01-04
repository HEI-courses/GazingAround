import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
import os.path

class ReportGenerator:
    def __init__(self):
        pass
    # create report from a saved log file
    def from_file(self):
        df = pd.read_csv("log.csv")
        self.width = int(df.iloc[0]['gazex'])
        self.height = int(df.iloc[0]['gazey'])
        df = df.iloc[1:]
        df = df.set_index(['time'])
        df.index = pd.to_datetime(df.index)
        print(df)
        self.options = "nan"
        return df
    
    #used to create a report from a pandas dataframe
    def from_df(self, df, app = None, time_interval = None):
        self.width = int(df.iloc[0]['gazex'])
        self.height = int(df.iloc[0]['gazey'])
        df = df.iloc[1:]
        df = df.set_index(['time'])
        df.index = pd.to_datetime(df.index)
        self.filter_data(df, app, time_interval)

    def generate_heatmap(self, df:pd.DataFrame):
        size = (self.width, self.height)
        print(size)
        step_w = size[0]/10
        step_h = size[1]/10
        grid_values = []
        in_out = [0,0]
        #setup grid
        for i in range(10):
            grid_values.append([])
            for j in range(10):
                grid_values[i].append(0)
        
        for i in df.iloc[1:].iterrows():
            x_pos = int(i[1]['gazex'])
            y_pos = int(i[1]['gazey'])
            if x_pos > 0 and y_pos > 0 and x_pos < self.width and y_pos < self.height:
                try:
                    x = int(x_pos/step_w) 
                    y = int(y_pos/step_h)
                    grid_values[y][x] += 1
                except:
                    print("OH NO", x, y)
            else:
                in_out[1] += 1
            in_out[0] += 1

        for i in grid_values:
            print(i)

        return grid_values, in_out
    
    #creates the data for the pdf from the data from a df
    def write_report(self, df:pd.DataFrame):
        heatmap, in_out = self.generate_heatmap(df)
        if df.empty: return
        t_start:pd.Timestamp = df.index[0]
        t_end:pd.Timestamp = df.index[-1]
        timedelta:pd.Timedelta = t_end - t_start

        hours = timedelta.total_seconds()/3600
        minutes = (hours - int(hours)) * 60
        seconds = (minutes - int(minutes)) * 60

        seconds_out_tot = (timedelta.total_seconds())*(in_out[1]/in_out[0])

        hours_out = seconds_out_tot/3600
        minutes_out = (hours_out - int(hours_out)) * 60
        seconds_out = (minutes_out - int(minutes_out)) * 60

        time_str = f'Report duration: {int(hours)}h, {int(minutes)}m, {"%.2f" % seconds}s'
        time_str_out = f'Gaze time spend outside of screen: {int(hours_out)}h, {int(minutes_out)}m, {"%.2f" % seconds_out}s'
        start_str = f"Started at time {t_start.hour}:{t_start.minute}:{t_start.second}"
        date_str = f"{t_start.day_name()}, {t_start.month_name()} {t_start.day}th"
        app = df.iloc[0]['app']

        print(time_str, start_str, date_str)

        fig, ax = plt.subplots(figsize = (self.width/100,self.height/100))
        hm = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, cbar=False).get_figure()
        
        hm.savefig("heatmap.png")
        self.create_pdf([time_str, start_str, date_str, app, in_out, time_str_out])
    
    #creating the pdf
    def create_pdf(self, data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.set_font('arial', 'B', 30)
        pdf.cell(10)
        pdf.cell(75, 30, f"Gaze report for {data[3]}")
        pdf.cell(90, 20, " ", 0, 2, 'C')
        pdf.cell(-105)
        pdf.image("heatmap.png", x = None, y = None, w = 240, h = 120, type = '', link = '')
        pdf.set_font('arial', '', 12)
        pdf.cell(30)
        pdf.cell(10, 0, "This is the heatmap representing the gaze presence on different parts of the screen.")
        pdf.cell(-10)
        pdf.cell(10, 10, f"{data[0]}")
        pdf.cell(-10)
        pdf.cell(10, 20, f"{data[1]} on {data[2]}")
        pdf.cell(-10)
        ratio = (data[4][1]/data[4][0])*100
        pdf.cell(10, 30, f"Ratio of gazes out of the screen: {"%.2f" % ratio}%, ")
        pdf.cell(-10)
        pdf.cell(10, 40, data[5])
        pdf.cell(-10)
        pdf.cell(10, 50, f"Options selected: {self.options}")
        file_nbr = 0
        while(True):
            if os.path.exists(f"report{file_nbr}.pdf"):
                file_nbr+=1
            else:
                pdf.output(f"report{file_nbr}.pdf", 'F')
                break

    
    # Data filtering and creation of dataframes for specified settings
    def filter_data(self, df:pd.DataFrame, app:str = None, time_interval:int = None):
        if app is None:
            time_interval = time_interval+0.1
            groups = df.groupby('app')
            split_dfs = [data for group, data in groups]
            for i in split_dfs:
                if time_interval is None:
                    self.options = "All apps, no time interval"
                    self.write_report(i)
                else:
                    groups = i.resample(f'{time_interval}s')
                    self.options = "All apps with time interval"
                    split_dfs2 = [group for _, group in groups]
                    for j in split_dfs2:
                        self.write_report(j)
        else:
            if time_interval is None:
                app_df = df[df['app'] == app]
                self.options = "Specific app, no time interval"
                self.write_report(app_df)
            else:
                app_df = df[df['app'] == app]
                self.options = "Specific app with time interval"
                groups = app_df.resample(f'{time_interval}s')
                split_dfs = [group for _, group in groups]
                for i in split_dfs:
                    self.write_report(i)

if __name__ == '__main__':
    report = ReportGenerator()
    df = report.from_file()
    report.filter_data(df, app="firefox")