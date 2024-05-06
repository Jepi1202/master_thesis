from pysr import PySRRegressor

NB_RUN = 50
BINARY_OP = ["+", "*", "/", "cond"]
UNARY_OP = None



def getPySrModel(nbRun = NB_RUN, binaryOp = BINARY_OP, unaryOp = UNARY_OP):

    mdoel = PySRRegressor(
        niterations=nbRun,
        binary_operators=binaryOp,
        unary_operators= unaryOp,
    )

    return mdoel


def fittingModel(model, X, y, verbose:bool = False):

    if verbose:
        print(">>>>> Fitting pySr")


    # Fit model
    model.fit(X, y)

    return model


