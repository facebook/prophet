#' Pick changepoints from prophet object
#'
#' @param model Prophet model object.
#' @param digits Integer, indicating the number of decimal places to be used. Default 2.
#'
#' @return A data frame consists of changepoints, growth rates and delta (changes in the growth rates).
#'
#' @examples
#' \dontrun{
#' m <- prophet(df)
#' prophet_pick_changepoints(m)
#' }
#'
#' @export
prophet_pick_changepoints <- function(model, digits = 2) {
  df <- data.frame(cp = model$changepoints,
                   cp.t = model$changepoints.t,
                   delta = as.vector(model$params$delta))
  while(nrow(df) > 1 && any(abs(df$delta) < 10^-digits)) {
    ind <- which.min(abs(df$delta))
    if (ind == 1) {
      pos <- 2
    } else if (ind == nrow(df) || 2 * df$cp.t[ind] < df$cp.t[ind-1] + df$cp.t[ind+1]) {
      pos <- ind - 1
    } else {
      pos <- ind + 1
    }
    df$delta[pos] <- df$delta[pos] + df$delta[ind]
    df <- df[-ind, , drop = FALSE]
  }
  if (nrow(df) == 1 && abs(df$delta) < 10^-digits) {
    df <- data.frame()
  }
  cp <- c(model$start, df$cp)
  delta <- c(0, df$delta)
  growth_rate <- model$params$k + cumsum(delta)
  result <- data.frame(changepoints = cp, growth_rate = growth_rate, delta = delta)
  class(result) <- c("prophet_changepoint", class(result))
  result
}
