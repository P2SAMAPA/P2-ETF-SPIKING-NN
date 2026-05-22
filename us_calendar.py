import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

def next_trading_day():
    cal = USFederalHolidayCalendar()
    nyse = CustomBusinessDay(calendar=cal)
    today = pd.Timestamp.today().normalize()
    next_day = today + nyse
    if next_day == today:
        next_day = today + 2*nyse
    return next_day.strftime("%Y-%m-%d")
