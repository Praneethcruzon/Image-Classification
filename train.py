# Gets a batch of training data from the DataLoader
# Zeros the optimizer’s gradients
# Performs an inference - that is, gets predictions from the model for an input batch
# Calculates the loss for that set of predictions vs. the labels on the dataset
# Calculates the backward gradients over the learning weights
# Tells the optimizer to perform one learning step - that is, adjust the model’s learning weights based on the observed gradients for this batch, according to the optimization algorithm we chose
# It reports on the loss for every 1000 batches.
# Finally, it reports the average per-batch loss for the last 1000 batches, for comparison with a validation run

# Ref https://pytorch.org/tutorials/beginner/introyt/trainingyt.html