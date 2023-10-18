from src.feature_engineering.feature_engineering import FE


class Time(FE):
    def __init__(self, config):
        super().__init__(config)
        self.day = self.config.get("day", False)
        self.hour = self.config.get("hour", False)
        self.minute = self.config.get("minute", False)
        self.second = self.config.get("second", False)

    def feature_engineering(self, data):
        # Group the data
        if self.day:
            data["f_day"] = data["timestamp"].dt.day
        if self.hour:
            data["f_hour"] = data["timestamp"].dt.hour
        if self.minute:
            data["f_minute"] = data["timestamp"].dt.minute
        if self.second:
            data["f_second"] = data["timestamp"].dt.second

        return data
