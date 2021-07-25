import pandas as pd
from datetime import datetime
from loguru import logger


def get_peds_counts():
    # If link is broken, check https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-Monthly-counts-per-hour/b2ak-trbp
    COUNTS_URL = "https://data.melbourne.vic.gov.au/api/views/b2ak-trbp/rows.csv"

    # Hand-picked based on counts and timeframe of available data
    SENSORS_WHITELIST = [1, 2, 5, 6, 8, 9, 11, 12, 13, 15]

    df = pd.read_csv(COUNTS_URL, usecols=["Date_Time", "Sensor_ID", "Hourly_Counts"]).rename(
        columns={"Date_Time": "ds", "Sensor_ID": "sensor_id", "Hourly_Counts": "y"}
    )
    df = df.loc[df["sensor_id"].isin(SENSORS_WHITELIST)].sort_values(by=["sensor_id", "ds"])
    df["ds"] = df["ds"].apply(lambda x: datetime.strptime(x, "%m/%d/%Y %I:%M:%S %p"))
    return df


def get_peds_sensors():
    # If link is broken, check https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-Sensor-Locations/h57g-5234
    SENSORS_URL = "https://data.melbourne.vic.gov.au/api/views/h57g-5234/rows.csv"
    df = pd.read_csv(SENSORS_URL)
    df["installation_date"] = df["installation_date"].apply(
        lambda x: datetime.strptime(x, "%Y/%m/%d")
    )
    return df


def agg_peds_counts(df: pd.DataFrame, start: str, end: str):
    time_filter = (df["ds"] >= datetime.fromisoformat(start)) & (
        df["ds"] < datetime.fromisoformat(end)
    )
    df = df.loc[time_filter].copy()
    df["date"] = df["ds"].dt.date
    dfg = (
        df.groupby(["sensor_id", "date"])
        .agg({"y": "sum"})
        .reset_index()
        .rename(columns={"date": "ds"})
    )
    return dfg.sort_values(["sensor_id", "ds"])


def generate_peds_data():
    logger.info("Downloading pedestrians data")
    counts = get_peds_counts()
    logger.info("Aggregating pedestrians data")
    sensors = get_peds_sensors()
    peds = agg_peds_counts(counts, start="2010-01-01", end="2017-01-01")
    peds = pd.merge(peds, sensors[["sensor_id", "sensor_description"]], on="sensor_id", how="inner")
    peds.to_csv("example_pedestrians_multivariate.csv", index=False)
    return peds


def get_power_data():
    # If link is broken, check https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
    POWER_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    df = pd.read_csv(
        POWER_URL,
        delimiter=";",
        dtype={
            **{"Date": str, "Time": str},
            **{
                col: float
                for col in [
                    "Global_active_power",
                    "Global_reactive_power",
                    "Voltage",
                    "Global_intensity",
                    "Sub_metering_1",
                    "Sub_metering_2",
                    "Sub_metering_3",
                ]
            },
        },
        na_values="?",
    )
    return df


def agg_power_data(df: pd.DataFrame):
    dfg = (
        df.groupby("Date", as_index=False, sort=False)[
            ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        ]
        .agg("sum")
        .fillna(0)
    )
    dfg["ds"] = dfg["Date"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y"))
    pc = dfg.melt(
        id_vars="ds",
        value_vars=["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
        var_name="meter",
        value_name="y",
    ).sort_values(["meter", "ds"])
    pc["meter"] = pc["meter"].str.lower()
    return pc


def generate_power_data():
    logger.info("Downloading power consumption data")
    df = get_power_data()
    logger.info("Aggregating power consumption data")
    pc = agg_power_data(df)
    pc.to_csv("example_power_multivariate.csv", index=False)
    return pc


def get_retail_data():
    # If link is broken, check https://archive.ics.uci.edu/ml/datasets/Online+Retail#
    ONLINE_RETAIL_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    )
    df = pd.read_excel(ONLINE_RETAIL_URL, engine="openpyxl")
    df["ds"] = df["InvoiceDate"].dt.date
    return df


def agg_retail_data(df: pd.DataFrame):
    top_items = (
        df.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(n=30).index
    )
    dfg = (
        df.loc[df["StockCode"].isin(top_items) & (df["Quantity"] > 0)]
        .groupby(["StockCode", "ds"], as_index=False)["Quantity"]
        .agg("sum")
        .rename(columns={"StockCode": "item", "Quantity": "y"})
    )
    return dfg


def generate_retail_data():
    logger.info("Downloading online retail data")
    df = get_retail_data()
    logger.info("Aggregating online retail data")
    dfg = agg_retail_data(df)
    dfg.to_csv("example_online_retail_multivariate.csv", index=False)
    return dfg


if __name__ == "__main__":
    logger.info("### Melbourne Pedestrians Sensors ###")
    generate_peds_data()
    logger.info("### Household Electricity Power ###")
    generate_power_data()
    logger.info("### Online Retail Sales ###")
    generate_retail_data()
