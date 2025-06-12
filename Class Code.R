# Define the sigmoid function
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# Generate x values
x <- seq(-10, 10, by = 0.1)

# Compute y values
y <- sigmoid(x)

# Create the plot
plot(x, y, type = "l", col = "blue", lwd = 2,
     xlab = "x", ylab = "f(x)",
     main = "Sigmoid Function: F(x) = 1/(1 + e^(-x))")

################################################################################

# Define the function
f <- function(x) {
  exp(x) / (1 + exp(x))
}

# Generate x values
x <- seq(-10, 10, by = 0.1)

# Compute y values
y <- f(x)

# Create the plot
plot(x, y, type = "l", col = "purple", lwd = 2,
     xlab = "x", ylab = "f(x)",
     main = "Function: F(x) = e^x / (1 + e^x)")

################################################################################
