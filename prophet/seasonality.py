import logging

def set_auto_seasonalities(df, yearly_seasonality):
    if yearly_seasonality and len(df) < 730:
        logging.warning('Yearly seasonality is forced on with insufficient history. This may lead to unstable forecasts.')
    # existing code...