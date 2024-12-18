import pandas as pd

class ReportGenerator:
    def __init__(self):
        pass
    def from_file(self):
        df = pd.read_csv("log.csv")

        print(df)
        return df
    def generate_heatmap(self, df:pd.DataFrame):
        size = (int(df.iloc[0]['gazex']), int(df.iloc[0]['gazey']))
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

if __name__ == '__main__':
    report = ReportGenerator()
    df = report.from_file()
    report.generate_heatmap(df)