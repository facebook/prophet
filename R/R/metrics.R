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
#' @param m Prophet object.
#' 
#' @return metrics value (numeric)
#' 
#'@examples
#'\dontrun{
#' # Create example model
#' library(readr)
#' df <- read_csv('../tests/testthat/data.csv')
#' m <- prophet(df)
#' # You can check your models's accuracy using me, mse, rmse ...etc.
#' print(rmse(m))
#'}
#' @name metrics
NULL

#' @rdname metrics
#' @export
me <- function(m)
{
  df <- create_metric_data(m)
  mean(df$y-df$yhat)
}

#' @rdname metrics
#' @export
mse <- function(m)
{
  df <- create_metric_data(m)
  mean((df$y-df$yhat)^2)
}

#' @rdname metrics
#' @export
rmse <- function(m)
{
  sqrt(mse(m))
}

#' @rdname metrics
#' @export
mae <- function(m)
{
  df <- create_metric_data(m)
  mean(abs(df$y-df$yhat))
}

#' @rdname metrics
#' @export
mpe <- function(m)
{
  df <- create_metric_data(m)
  100*mean((df$y-df$yhat)/df$y)
}

#' @rdname metrics
#' @export
mape <- function(m)
{
  df <- create_metric_data(m)
  100*mean(abs(df$y-df$yhat)/df$y)
}

#' Prepare dataframe for metrics calculation.
#'
#' @param m Prophet object.
#'
#' @return A dataframe only with y and yhat as a column.
#'
#' @keywords internal
create_metric_data <- function(m)
{
  forecast <- predict(m, NULL)
  dplyr::inner_join(m$history, forecast, by="ds") %>%
    dplyr::select(y, yhat) %>%
    na.omit()
}
