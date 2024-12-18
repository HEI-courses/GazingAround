import pandas as pd

class ReportGenerator:
    def __init__(self):
        pass
    def from_file(self):
        df = pd.read_csv("log1.csv")
        self.width = int(df.iloc[0]['gazex'])
        self.height = int(df.iloc[0]['gazey'])
        df = df.iloc[1:]
        df = df.set_index(['time'])
        df.index = pd.to_datetime(df.index)
        return df
    def generate_heatmap(self, df:pd.DataFrame):
        size = (self.height, self.width)
        print(size)
        step_w = size[0]/10
        step_h = size[1]/10
        grid_values = []
        #setup grid
        for i in range(10):
            grid_values.append([])
            for j in range(10):
                grid_values[i].append(0)
        
        for i in df.iloc[1:].iterrows():
            x_pos = int(i[1]['gazex'])
            y_pos = int(i[1]['gazey'])
            x = int(x_pos/step_w) if int(x_pos/step_w) < 10 else 9
            y = int(y_pos/step_h) if int(y_pos/step_h) < 10 else 9
            grid_values[x][y] += 1

        for i in grid_values:
            print(i)

        return grid_values
    
    def write_report(self, heatmap):
        pass

    def filter_data(self, df:pd.DataFrame, app:str = None, time_interval:int = None):
        if app is None:
            groups = df.groupby('app')
            split_dfs = [data for group, data in groups]
            for i in split_dfs:
                if time_interval is None:
                    self.generate_heatmap(i)
                else:
                    groups = i.resample(f'{time_interval}s')
                    split_dfs2 = [group for _, group in groups]
                    for j in split_dfs2:
                        self.generate_heatmap(j)
        else:
            if time_interval is None:
                app_df = df[df['app'] == app]
                hm = self.generate_heatmap(app_df)
                self.write_report(hm)
            else:
                app_df = df[df['app'] == app]
                 
                groups = app_df.resample(f'{time_interval}s')
                split_dfs = [group for _, group in groups]
                for i in split_dfs:
                    self.generate_heatmap(i)

if __name__ == '__main__':
    report = ReportGenerator()
    df = report.from_file()
    report.filter_data(df, "firefox", time_interval=2)