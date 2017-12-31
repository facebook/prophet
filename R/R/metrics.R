## Copyright (c) 2017-present, Facebook, Inc.
## All rights reserved.

## This source code is licensed under the BSD-style license found in the
## LICENSE file in the root directory of this source tree. An additional grant
## of patent rights can be found in the PATENTS file in the same directory.

#' @title Metrics for Time Series Forecasts
#'
#' @description
#' A time-series forecast requires making a quantitative prediction of future values.
#' After forecast, we also have to provide accurracy of forecasts to check wether the forecast serves our need.
#' Metrics for time series forecasts are so useful in telling you how your model is good and helping you determine which particular forecasting models work best.
#'
#' @details
#' Here, as a notation, we assume that \eqn{y} is the actual value and \eqn{yhat} is the forecast value.
#'
#' Mean Error (ME, \code{me})
#'
#' The Mean Error (ME)  is defined by the formula:
#' \deqn{ \frac{1}{n} \sum_{t=1}^{n} y_{t}-yhat_{t} .}
#'
#' Mean Squared Error (MSE, \code{mse})
#'
#' The Mean Squared Error (MSE)  is defined by the formula:
#' \deqn{ \frac{1}{n} \sum_{t=1}^{n} (y_{t}-yhat_{t})^2 .}
#'
#' Root Mean Square Error (RMSE, \code{rmse})
#'
#' Root Mean Square Error (RMSE) is define by the formula:
#' \deqn{ \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_{t}-yhat_{t})^2} .}
#'
#' Mean Absolute Error (MAE, \code{mae})
#'
#' The Mean Absolute Error (MAE) is defined by the formula:
#' \deqn{ \frac{1}{n} \sum_{t=1}^{n} | y_{t}-yhat_{t} | .}
#'
#' Mean Percentage Error (MPE, \code{mpe})
#'
#' The Mean Percentage Error (MPE) is usually expressed as a percentage
#' and is defined by the formula:
#' \deqn{ \frac{100}{n} \sum_{t=1}^{n} \frac {y_{t}-yhat_{t}}{y_{t}} .}
#'
#' Mean Absolute Percentage Error (MAPE, \code{mape})
#'
#' The Mean absolute Percentage Error (MAPE), also known as Mean Absolute Percentage Deviation (MAPD), is usually expressed as a percentage,
#' and is defined by the formula:
#' \deqn{ \frac{100}{n} \sum_{t=1}^{n} | \frac {y_{t}-yhat_{t}}{y_{t}}| .}
#'
#' @param df A dataframe which is output of `predict`, `simulated_historical_forecasts ` or ``
#'
#' @return metrics value (numeric)
#'
#'@examples
#'\dontrun{
#' # Create example model
#' library(readr)
#' library(prophet)
#' df <- read_csv('../tests/testthat/data.csv')
#' m <- prophet(df)
#' future <- make_future_dataframe(m, periods = 365)
#' forecast <- predict(m, future)
#' all_metrics(forecast)
#' df.cv <- cross_validation(m, horizon = 100, units = 'days')
#' all_metrics(df.cv)
#' # You can check your models's accuracy using me, mse, rmse ...etc.
#' print(rmse(m))
#'}
#' @name metrics
NULL

#' @rdname metrics
#' @export
me <- function(obj)
{
  df <- create_metric_data(obj)
  mean(df$y-df$yhat)
}

#' @rdname metrics
#' @export
mse <- function(obj)
{
  df <- create_metric_data(obj)
  mean((df$y-df$yhat)^2)
}

#' @rdname metrics
#' @export
rmse <- function(obj)
{
  sqrt(mse(obj))
}

#' @rdname metrics
#' @export
mae <- function(obj)
{
  df <- create_metric_data(obj)
  mean(abs(df$y-df$yhat))
}

#' @rdname metrics
#' @export
mpe <- function(obj)
{
  df <- create_metric_data(obj)
  100*mean((df$y-df$yhat)/df$y)
}

#' @rdname metrics
#' @export
mape <- function(obj)
{
  df <- create_metric_data(obj)
  100*mean(abs(df$y-df$yhat)/df$y)
}

#' @rdname metrics
#' @export
all_metrics <- function(obj)
{
  # Define all metrics functions as a character
  metrics <- rlang::set_names(c("me", "mse", "rmse", "mae", "mpe", "mape"))
  # Convert character to function and evalate each metrics in invoke_map_df
  # The result is data.frame with each metrics name
  purrr::invoke_map_df(metrics, list(list(obj)))
}

#' Prepare dataframe for metrics calculation.
#'
#' @param obj a Prophet object or a data.frame resulting from simulated_historical_forecasts() or cross_validation()
#'
#' @return A dataframe only with y and yhat as a column.
#'
#' @keywords internal
create_metric_data <- function(obj)
{
  # Judge as a data.frame resulting from simulated_historical_forecasts() or cross_validation()
  data <- if(is.data.frame(obj) & all(c("y", "yhat") %in% names(obj))){
    obj
  } else if("prophet" %in% class(obj)) {
    forecast <- predict(obj, NULL)
    dplyr::inner_join(obj$history, forecast, by="ds")
  } else{
    stop("obj argument must be Prophet object or a data.frame resulting from simulated_historical_forecasts() or cross_validation()")
  }

  dplyr::select(data, y, yhat) %>%
    na.omit()
}
