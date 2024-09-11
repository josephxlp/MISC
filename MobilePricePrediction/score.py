import pandas as pd
import joblib
import logging
import time
import os 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error


os.getcwd()

def main():
  """
  Loads and preprocesses data, loads model, generates 
  predictions, and logs metrics to scores.log
  """

  # Load data to be scored
  
  X_val = pd.read_csv('./data/scoring_data.csv')

  # Preprocess data
  preprocessing_pipeline = joblib.load('./models/preprocessing_pipeline.joblib')
  X_val = preprocessing_pipeline.transform(X_val)    

  # Load model
  model = joblib.load('./models/model.joblib')
  
  # Generate scores and measure prediction time
  start_time = time.time()
  scores = model.predict(X_val)
  prediction_time = time.time() - start_time

  # Assuming y_true is available for calculating metrics
  y_true = pd.read_csv('./data/true_values.csv')  # Load true values for validation 
  y_true = y_true['Price'].values
  #print(y_true.shape)
  #print(y_true)

  # Calculate regression metrics
  mse = mean_squared_error(y_true, scores)
  mae = mean_absolute_error(y_true, scores)
  r2 = r2_score(y_true, scores)
  evs = explained_variance_score(y_true, scores)
  max_err = max_error(y_true, scores)

  # Calculate AIC and BIC
 # aic, bic = calculate_aic_bic(model, X_val, y_true)

  # Log metrics
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  f_handler = logging.FileHandler('scores.log')
  f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  f_handler.setFormatter(f_format)
  logger.addHandler(f_handler)
  logger.info(f'MSE: {mse}, MAE: {mae}, R2: {r2}, Explained Variance: {evs}, Max Error: {max_err}')
  logger.info(f'Prediction Time: {prediction_time} seconds')
  #logger.info(f'AIC: {aic}, BIC: {bic}')

if __name__ == '__main__':
  main()