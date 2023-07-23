import os
import numpy as np
import pandas as pd
from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron

def main(data, modelName, plotname, eta, epochs):
    df_AND = pd.DataFrame(OR)
    
    X, y = prepare_data(df_AND)

    

    model_and = Perceptron(eta=eta, epochs=epochs)
    model_and.fit(X, y)

    _ = model_and.total_loss()
    
    model_and.save(filename=modelName, model_dir="model")
    save_plot(df_AND, model_and, filename=plotname)


if __name__ == "__main__"

    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,0,0,1]
    }
    
    ETA = 0.3
    EPOCHS = 10
    main(data=OR, modelName = "and.model", plotname ="and2.png", eta = ETA, epochs=EPOCHS)





