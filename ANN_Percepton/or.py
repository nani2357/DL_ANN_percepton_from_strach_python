import os
import numpy as np
import pandas as pd
from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import logging

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, "running_logs"),
    level=logging.INFO, 
    format='[%(asctime)s: %(levelname)s : %(module)s: %(message)s',
    filemode='a'
)


def main(data, modelName, plotname, eta, epochs):
    df_OR = pd.DataFrame(OR)
    logging.info(f"This is the raw dataset: {df}")
    
    X, y = prepare_data(df_OR)

    

    model_or = Perceptron(eta=eta, epochs=epochs)
    model_or.fit(X, y)

    _ = model_or.total_loss()
    
    model_or.save(filename=modelName, model_dir="model")
    save_plot(df_OR, model_or, filename=plotname)


if __name__ == "__main__"

    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,1,1,1]
    }
    
    ETA = 0.3
    EPOCHS = 10
    main(data=OR, modelName = "or.model", plotname ="or2.png", eta = ETA, epochs=EPOCHS)
    try:
        logging.info(f">>>>> starting training for {gate}>>>>>")
        main(data=OR, modelName="or.model", plotName="or.png", eta=ETA, epochs=EPOCHS)
        logging.info(f"<<<<< done training for {gate} <<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e




