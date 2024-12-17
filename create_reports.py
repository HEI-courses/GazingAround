import pandas as pd

class ReportGenerator:
    def __init__(self):
        pass
    def from_file(self):
        df = pd.read_csv("log.csv")

        print(df)
        return df
    def generate_heatmap(df:pd.DataFrame):
        df.max()

if __name__ == '__main__':
    report = ReportGenerator()
    df = report.from_file()