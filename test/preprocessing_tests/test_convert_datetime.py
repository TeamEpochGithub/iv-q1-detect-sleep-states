from unittest import TestCase
import polars as pl
from src.preprocessing.convert_datetime import ConvertDatetime


class TestConvertDatetime(TestCase):
    def test_convert_datetime(self):
        data = pl.read_parquet("test/test_series.parquet")
        # apply datetime conversion
        converter = ConvertDatetime()
        converted_data = converter.preprocess(data)
        self.assertTrue(converted_data["datetime"].dtype == pl.Datetime)
        # convert to pandas
        converted_data = converted_data.to_pandas()
        print(converted_data["datetime"].dtype)
        self.assertTrue(converted_data["datetime"].dtype == 'datetime64[us]')
