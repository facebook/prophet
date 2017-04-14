.onLoad <- function(libname, pkgname) {
  .prophet.stan.models <- list(
    "linear"=get_prophet_stan_model("linear"),
    "logistic"=get_prophet_stan_model("logistic"))
  assign(".prophet.stan.models", .prophet.stan.models,
         envir=parent.env(environment()))
}
