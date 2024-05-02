from pysr import PySRRegressor

# Suppose `X` is your input matrix (numpy array) and `y` is the output from the neural network
X = ...  # Your input grid
y = model.predict(X)  # Using your trained model to generate outputs

# Setup PySRRegressor with desired options
model_sr = PySRRegressor(
    niterations=5,
    binary_operators=["+", "*", "/", "-"],
    unary_operators=[
        "sin", "cos", "exp", "log"],  # Depending on the nature of your physical system
    model_selection="best",
    verbosity=1
)

# Fit model
model_sr.fit(X, y)

# Print the best equation found
print(model_sr)