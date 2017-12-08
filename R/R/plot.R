#
#' Merge history and forecast for plotting.
#'
#' @param m Prophet object.
#' @param fcst Data frame returned by prophet predict.
#'
#' @importFrom dplyr "%>%"
#' @keywords internal
df_for_plotting <- function(m, fcst) {
  # Make sure there is no y in fcst
  fcst$y <- NULL
  df <- m$history %>%
    dplyr::select(ds, y) %>%
    dplyr::full_join(fcst, by = "ds") %>%
    dplyr::arrange(ds)
  return(df)
}

#' Plot the prophet forecast.
#'
#' @param x Prophet object.
#' @param fcst Data frame returned by predict(m, df).
#' @param uncertainty Boolean indicating if the uncertainty interval for yhat
#'  should be plotted. Must be present in fcst as yhat_lower and yhat_upper.
#' @param plot_cap Boolean indicating if the capacity should be shown in the
#'  figure, if available.
#' @param xlabel Optional label for x-axis
#' @param ylabel Optional label for y-axis
#' @param ... additional arguments
#'
#' @return A ggplot2 plot.
#'
#' @examples
#' \dontrun{
#' history <- data.frame(ds = seq(as.Date('2015-01-01'), as.Date('2016-01-01'), by = 'd'),
#'                       y = sin(1:366/200) + rnorm(366)/10)
#' m <- prophet(history)
#' future <- make_future_dataframe(m, periods = 365)
#' forecast <- predict(m, future)
#' plot(m, forecast)
#' }
#'
#' @export
plot.prophet <- function(x, fcst, uncertainty = TRUE, plot_cap = TRUE,
                         xlabel = 'ds', ylabel = 'y', ...) {
  df <- df_for_plotting(x, fcst)
  gg <- ggplot2::ggplot(df, ggplot2::aes(x = ds, y = y)) +
    ggplot2::labs(x = xlabel, y = ylabel)
  if (exists('cap', where = df) && plot_cap) {
    gg <- gg + ggplot2::geom_line(
      ggplot2::aes(y = cap), linetype = 'dashed', na.rm = TRUE)
  }
  if (x$logistic.floor && exists('floor', where = df) && plot_cap) {
    gg <- gg + ggplot2::geom_line(
      ggplot2::aes(y = floor), linetype = 'dashed', na.rm = TRUE)
  }
  if (uncertainty && exists('yhat_lower', where = df)) {
    gg <- gg +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = yhat_lower, ymax = yhat_upper),
                           alpha = 0.2,
                           fill = "#0072B2",
                           na.rm = TRUE)
  }
  gg <- gg +
    ggplot2::geom_point(na.rm=TRUE) +
    ggplot2::geom_line(ggplot2::aes(y = yhat), color = "#0072B2",
                       na.rm = TRUE) +
    ggplot2::theme(aspect.ratio = 3 / 5)
  return(gg)
}

#' Plot the components of a prophet forecast.
#' Prints a ggplot2 with panels for trend, weekly and yearly seasonalities if
#' present, and holidays if present.
#'
#' @param m Prophet object.
#' @param fcst Data frame returned by predict(m, df).
#' @param uncertainty Boolean indicating if the uncertainty interval should be
#'  plotted for the trend, from fcst columns trend_lower and trend_upper.
#' @param plot_cap Boolean indicating if the capacity should be shown in the
#'  figure, if available.
#' @param weekly_start Integer specifying the start day of the weekly
#'  seasonality plot. 0 (default) starts the week on Sunday. 1 shifts by 1 day
#'  to Monday, and so on.
#' @param yearly_start Integer specifying the start day of the yearly
#'  seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts by 1 day
#'  to Jan 2, and so on.
#'
#' @return Invisibly return a list containing the plotted ggplot objects
#'
#' @export
#' @importFrom dplyr "%>%"
prophet_plot_components <- function(
  m, fcst, uncertainty = TRUE, plot_cap = TRUE, weekly_start = 0,
  yearly_start = 0
) {
  # Plot the trend
  panels <- list(plot_forecast_component(fcst, 'trend', uncertainty, plot_cap))
  # Plot holiday components, if present.
  if (!is.null(m$holidays) & ('holidays' %in% colnames(fcst))) {
    panels[[length(panels) + 1]] <- plot_forecast_component(
      fcst, 'holidays', uncertainty, FALSE)
  }
  # Plot weekly seasonality, if present
  if ("weekly" %in% colnames(fcst)) {
    panels[[length(panels) + 1]] <- plot_weekly(m, uncertainty, weekly_start)
  }
  # Plot yearly seasonality, if present
  if ("yearly" %in% colnames(fcst)) {
    panels[[length(panels) + 1]] <- plot_yearly(m, uncertainty, yearly_start)
  }
  # Plot other seasonalities
  for (name in names(m$seasonalities)) {
    if (!(name %in% c('weekly', 'yearly')) &&
        (name %in% colnames(fcst))) {
      panels[[length(panels) + 1]] <- plot_seasonality(m, name, uncertainty)
    }
  }
  # Plot extra regressors
  if ((length(m$extra_regressors) > 0)
      & ('extra_regressors' %in% colnames(fcst))) {
    panels[[length(panels) + 1]] <- plot_forecast_component(
      fcst, 'extra_regressors', uncertainty, FALSE)
  }

  # Make the plot.
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(length(panels),
                                                               1)))
  for (i in 1:length(panels)) {
    print(panels[[i]], vp = grid::viewport(layout.pos.row = i,
                                           layout.pos.col = 1))
  }
  return(invisible(panels))
}

#' Plot a particular component of the forecast.
#'
#' @param fcst Dataframe output of `predict`.
#' @param name String name of the component to plot (column of fcst).
#' @param uncertainty Boolean to plot uncertainty intervals.
#' @param plot_cap Boolean indicating if the capacity should be shown in the
#'  figure, if available.
#'
#' @return A ggplot2 plot.
#'
#' @export
plot_forecast_component <- function(
  fcst, name, uncertainty = TRUE, plot_cap = FALSE
) {
  gg.comp <- ggplot2::ggplot(
      fcst, ggplot2::aes_string(x = 'ds', y = name, group = 1)) +
    ggplot2::geom_line(color = "#0072B2", na.rm = TRUE)
  if (exists('cap', where = fcst) && plot_cap) {
    gg.comp <- gg.comp + ggplot2::geom_line(
      ggplot2::aes(y = cap), linetype = 'dashed', na.rm = TRUE)
  }
  if (exists('floor', where = fcst) && plot_cap) {
    gg.comp <- gg.comp + ggplot2::geom_line(
      ggplot2::aes(y = floor), linetype = 'dashed', na.rm = TRUE)
  }
  if (uncertainty) {
    gg.comp <- gg.comp +
      ggplot2::geom_ribbon(
        ggplot2::aes_string(
          ymin = paste0(name, '_lower'), ymax = paste0(name, '_upper')
        ),
        alpha = 0.2,
        fill = "#0072B2",
        na.rm = TRUE)
  }
  return(gg.comp)
}

#' Prepare dataframe for plotting seasonal components.
#'
#' @param m Prophet object.
#' @param ds Array of dates for column ds.
#'
#' @return A dataframe with seasonal components on ds.
#'
#' @keywords internal
seasonality_plot_df <- function(m, ds) {
  df_list <- list(ds = ds, cap = 1)
  for (name in names(m$extra_regressors)) {
    df_list[[name]] <- 0
  }
  df <- as.data.frame(df_list)
  df <- setup_dataframe(m, df)$df
  return(df)
}

#' Plot the weekly component of the forecast.
#'
#' @param m Prophet model object
#' @param uncertainty Boolean to plot uncertainty intervals.
#' @param weekly_start Integer specifying the start day of the weekly
#'  seasonality plot. 0 (default) starts the week on Sunday. 1 shifts by 1 day
#'  to Monday, and so on.
#'
#' @return A ggplot2 plot.
#'
#' @keywords internal
plot_weekly <- function(m, uncertainty = TRUE, weekly_start = 0) {
  # Compute weekly seasonality for a Sun-Sat sequence of dates.
  days <- seq(set_date('2017-01-01'), by='d', length.out=7) + weekly_start
  df.w <- seasonality_plot_df(m, days)
  seas <- predict_seasonal_components(m, df.w)
  seas$dow <- factor(weekdays(df.w$ds), levels=weekdays(df.w$ds))

  gg.weekly <- ggplot2::ggplot(seas, ggplot2::aes(x = dow, y = weekly,
                                                  group = 1)) +
    ggplot2::geom_line(color = "#0072B2", na.rm = TRUE) +
    ggplot2::labs(x = "Day of week")
  if (uncertainty) {
    gg.weekly <- gg.weekly +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = weekly_lower,
                                        ymax = weekly_upper),
                           alpha = 0.2,
                           fill = "#0072B2",
                           na.rm = TRUE)
  }
  return(gg.weekly)
}

#' Plot the yearly component of the forecast.
#'
#' @param m Prophet model object.
#' @param uncertainty Boolean to plot uncertainty intervals.
#' @param yearly_start Integer specifying the start day of the yearly
#'  seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts by 1 day
#'  to Jan 2, and so on.
#'
#' @return A ggplot2 plot.
#'
#' @keywords internal
plot_yearly <- function(m, uncertainty = TRUE, yearly_start = 0) {
  # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
  days <- seq(set_date('2017-01-01'), by='d', length.out=365) + yearly_start
  df.y <- seasonality_plot_df(m, days)
  seas <- predict_seasonal_components(m, df.y)
  seas$ds <- df.y$ds

  gg.yearly <- ggplot2::ggplot(seas, ggplot2::aes(x = ds, y = yearly,
                                                  group = 1)) +
    ggplot2::geom_line(color = "#0072B2", na.rm = TRUE) +
    ggplot2::labs(x = "Day of year") +
    ggplot2::scale_x_datetime(labels = scales::date_format('%B %d'))
  if (uncertainty) {
    gg.yearly <- gg.yearly +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = yearly_lower,
                                        ymax = yearly_upper),
                           alpha = 0.2,
                           fill = "#0072B2",
                           na.rm = TRUE)
  }
  return(gg.yearly)
}

#' Plot a custom seasonal component.
#'
#' @param m Prophet model object.
#' @param name String name of the seasonality.
#' @param uncertainty Boolean to plot uncertainty intervals.
#'
#' @return A ggplot2 plot.
#'
#' @keywords internal
plot_seasonality <- function(m, name, uncertainty = TRUE) {
  # Compute seasonality from Jan 1 through a single period.
  start <- set_date('2017-01-01')
  period <- m$seasonalities[[name]]$period
  end <- start + period * 24 * 3600
  plot.points <- 200
  days <- seq(from=start, to=end, length.out=plot.points)
  df.y <- seasonality_plot_df(m, days)
  seas <- predict_seasonal_components(m, df.y)
  seas$ds <- df.y$ds
  gg.s <- ggplot2::ggplot(
      seas, ggplot2::aes_string(x = 'ds', y = name, group = 1)) +
    ggplot2::geom_line(color = "#0072B2", na.rm = TRUE)
  if (period <= 2) {
    fmt.str <- '%T'
  } else if (period < 14) {
    fmt.str <- '%m/%d %R'
  } else {
    fmt.str <- '%m/%d'
  }
  gg.s <- gg.s +
    ggplot2::scale_x_datetime(labels = scales::date_format(fmt.str))
  if (uncertainty) {
    gg.s <- gg.s +
    ggplot2::geom_ribbon(
      ggplot2::aes_string(
        ymin = paste0(name, '_lower'), ymax = paste0(name, '_upper')
      ),
      alpha = 0.2,
      fill = "#0072B2",
      na.rm = TRUE)
  }
  return(gg.s)
}

#' Plot the prophet forecast.
#'
#' @param x Prophet object.
#' @param fcst Data frame returned by predict(m, df).
#' @param uncertainty Boolean indicating if the uncertainty interval for yhat
#'  should be plotted. Must be present in fcst as yhat_lower and yhat_upper.
#' @param ... additional arguments
#' @importFrom magrittr "%>%"
#' @return A dygraph plot.
#'
#' @examples
#' \dontrun{
#' history <- data.frame(ds = seq(as.Date('2015-01-01'), as.Date('2016-01-01'), by = 'd'),
#'                       y = sin(1:366/200) + rnorm(366)/10)
#' m <- prophet(history)
#' future <- make_future_dataframe(m, periods = 365)
#' forecast <- predict(m, future)
#' dyplot.prophet(m, forecast)
#' }
#'
#' @export
dyplot.prophet <- function(x, fcst, uncertainty=TRUE, 
                           ...) 
{
  forecast.label='Predicted'
  actual.label='Actual'
  # create data.frame for plotting
  df <- df_for_plotting(x, fcst)
  
  # build variables to include, or not, the uncertainty data
  if(uncertainty && exists("yhat_lower", where = df))
  {
    colsToKeep <- c('y', 'yhat', 'yhat_lower', 'yhat_upper')
    forecastCols <- c('yhat_lower', 'yhat', 'yhat_upper')
  } else
  {
    colsToKeep <- c('y', 'yhat')
    forecastCols <- c('yhat')
  }
  # convert to xts for easier date handling by dygraph
  dfTS <- xts::xts(df %>% dplyr::select_(.dots=colsToKeep), order.by = df$ds)

  # base plot
  dyBase <- dygraphs::dygraph(dfTS)
  
  presAnnotation <- function(dygraph, x, text) {
    dygraph %>%
      dygraphs::dyAnnotation(x, text, text, attachAtBottom = TRUE)
  }
  
  dyBase <- dyBase %>% 
    dygraphs::dyOptions(colors = RColorBrewer::brewer.pal(3, "Set1")) %>%
    # plot actual values
    dygraphs::dySeries('y', label=actual.label) %>% 
    # plot forecast and ribbon
    dygraphs::dySeries(forecastCols, label=forecast.label) %>% 
    # allow zooming
    dygraphs::dyRangeSelector() %>% 
    # make unzoom button
    dygraphs::dyUnzoom()
  if (!is.null(x$holidays)) {
    for (i in 1:nrow(x$holidays)) {
      # make a gray line
      dyBase <- dyBase %>% dygraphs::dyEvent(x$holidays$ds[i],color = "rgb(200,200,200)", strokePattern = "solid")
      dyBase <- dyBase %>% dygraphs::dyAnnotation(x$holidays$ds[i], x$holidays$holiday[i], x$holidays$holiday[i], attachAtBottom = TRUE)
    }
  }
  return(dyBase)
}

