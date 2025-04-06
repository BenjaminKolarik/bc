# install_packages.R

required_packages <- c("readxl", "dplyr", "tidyr", "ggplot2", "openxlsx", "car", "lmtest")

# Function to check if a package is installed
is_installed <- function(pkg) {
  is.element(pkg, installed.packages()[, "Package"])
}

# Install missing packages
for (pkg in required_packages) {
  if (!is_installed(pkg)) {
    install.packages(pkg, dependencies = TRUE)
  }
}