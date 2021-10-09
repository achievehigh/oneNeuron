"""
author : Rhevathi
email :achievehigh.nevergiveup@gmail.com
"""

    
from utils.model import Perceptron
from utils.all_utils import prepare_data,save_plot,save_model
import pandas as pd
#import numpy as np
import logging
import os

logging_str="[%(asctime)s:%(levelname)s:%(module)s:%(message)s]"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(level=logging.INFO,format=logging_str)

def main(data,modelName,plotName,eta,epochs):
    df=pd.DataFrame(data)
    logging.info(f"This is actual dataframe{df}")
    X,y=prepare_data(df)
    model_OR=Perceptron(eta=eta,epochs=epochs)
    model_OR.fit(X,y)
    _=model_OR.total_loss()
    save_model(model_OR,filename=modelName)
    save_plot(df,plotName,model_OR)

if __name__== "__main__":
    OR ={
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,1]   
    }
#df=pd.DataFrame(OR)
#logging.info(f"This is actual dataframe{df}")

#X,y=prepare_data(df)
ETA=0.3 
EPOCHS=10
#model_OR =Perceptron(eta=ETA,epochs=EPOCHS)

#model_OR.fit(X,y)

#_=model_OR.total_loss()

#save_model(model_OR,filename="or.model")
#save_plot(df,"or.png",model_OR)
try :
    logging.info(">>>>Starting training<<<<<")
    main(data= OR,modelName="or.model",plotName="or.png",eta=ETA,epochs=EPOCHS)
    logging.info("<<<<training done sucessfully<<<<\n")
except Exception as e :
    logging.exception(e)
    #raise e
