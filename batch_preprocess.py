import pandas as pd

class Preprocessing:
    """Class for preparing welding sequence data for ML"""

    def __init__(self):
        self.time_col = 'Y/M/D hh:mm:ss'

    def _get_faulty_barcodes(self, df, lim1=448, lim2=448) -> list:
        tl = df.value_counts(["BarCode"]) < lim1
        ls_too_little_welds = tl[tl == True].index
        tm = df.value_counts(["BarCode"]) > lim2
        ls_too_many_welds = tm[tm == True].index
        bc_to_remove = list(ls_too_little_welds) + list(ls_too_many_welds)
        return [i[0] for i in bc_to_remove]

    def _filter_rows_by_values(self, df, col, values)-> pd.DataFrame:
        return df[~df[col].isin(values)]

    def apply_filter(self, df)->pd.DataFrame:
        bc_to_remove = self._get_faulty_barcodes(df)
        return  self._filter_rows_by_values(df, "BarCode", bc_to_remove)

    def _clean_weld_data(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        # strip barcodes
        df.BarCode = df.BarCode.apply(lambda x: x.strip() if isinstance(x, str) else x)
        # remove barcode (does not occur as already filtered in Excel)
        df = df[~(df.BarCode == 'sZGRZ50')]
        # reformat time column
        df['ts'] = pd.to_datetime(df[time_column])
        return df

    def _add_location(self, df: pd.DataFrame) -> pd.DataFrame:
        # get number
        for col in ["Face", "Cell", "Point"]:
            df[col] = df[col].str.slice(start=0, stop=-2)
        # create location string
        df["Face_Cell_Point"] = df["Face"].astype(str) + "_" + df["Cell"].astype(str) + "_" + df["Point"].astype(str)
        return df

    def _add_weekday(self, df: pd.DataFrame) -> pd.DataFrame:
        df['weekday'] = df.ts.dt.weekday
        return df

    def _get_good_bad(self, df: pd.DataFrame):
        df['anomaly'] = df.duplicated(subset=['BarCode','Face_Cell_Point'], keep='last')
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # df_out = self.apply_filter(df)
        df_out = self._clean_weld_data(df, self.time_col)
        df_out = self._add_location(df_out)
        df_out = self._add_weekday(df_out)
        df_out = self._get_good_bad(df_out)
        return df_out