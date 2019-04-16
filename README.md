# Input predictor variable matrix where each row of X corresponds to an observation, 
# observed response variables y, and the number of iterations of feedforward and backpropagation
Neural_Network <- function(X,y,n){
  
  # Generate a random number between 0 and 1 for each element in X to use as the initial weights for layer 1
  rand_vector <- runif(ncol(X) * nrow(X))
  
  # Convert this random vector into a matrix of the same dimensions as X
  rand_matrix <- matrix(
    rand_vector,
    nrow = ncol(X),
    ncol = nrow(X),
    byrow = TRUE
  )
  
  my_nn <- list(
    # predictor variables
    input = X,
    # weights for layer 1
    weights1 = rand_matrix,
    # weights for layer 2
    weights2 = matrix(runif(length(y)), ncol = 1),
    # observed outcome
    y = y,
    # predicted outcome
    output = matrix(
      rep(0, times = length(y)),
      ncol = 1
    )
  )
  
  # Activation function
  sigmoid <- function(x) {
    1.0 / (1.0 + exp(-x))
  }
  
  # Derivative of activation function
  sigmoid_derivative <- function(x) {
    x * (1.0 - x)
  }
  
  # Loss function using sum-of-squares error
  loss_function <- function(nn) {
    sum((nn$y - nn$output) ^ 2)
  }
  
  # Feedforward function
  feedforward <- function(nn) {
    nn$layer1 <- sigmoid(nn$input %*% nn$weights1)
    nn$output <- sigmoid(nn$layer1 %*% nn$weights2)
    nn
  }
  
  # Backpropagation function
  backprop <- function(nn) {
    
    # derivative of the loss function with respect to weights2 and weights1
    d_weights2 <- (
      t(nn$layer1) %*% (2 * (nn$y - nn$output) * sigmoid_derivative(nn$output))
    )
    
    d_weights1 <- ( 2 * (nn$y - nn$output) * sigmoid_derivative(nn$output)) %*% 
      t(nn$weights2)
    d_weights1 <- d_weights1 * sigmoid_derivative(nn$layer1)
    d_weights1 <- t(nn$input) %*% d_weights1
    
    # Update weights using the derivative of the loss function
    nn$weights1 <- nn$weights1 + d_weights1
    nn$weights2 <- nn$weights2 + d_weights2
    
    nn
  }
  
  # Data frame stores results of loss function
  loss_df <- data.frame(
    iteration = 1:n,
    loss = vector("numeric", length = n)
  )
  
  for (i in seq_len(n)) {
    my_nn <- feedforward(my_nn)
    my_nn <- backprop(my_nn)
    
    # Store result of the loss function
    loss_df$loss[i] <- loss_function(my_nn)
  }
  
  # Print predictions with actual observations
  D = data.frame(
    "Predicted" = round(my_nn$output, 3),
    "Actual" = y
  )
  
  # Plot the cost vs iterations
  library(ggplot2)
  plot <- ggplot(data = loss_df, aes(x = iteration, y = loss)) + geom_line()
  
  return(list(D,plot))
}

