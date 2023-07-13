# DeepAmes
#### DeepAmes provides deep learning-powered Ames test predictive models for predicting the results of Ames tests.

### Script
  - base_knn.py (develop KNN base classifiers)
  - base_lr.py (develop LR base classifiers)
  - base_svm.py (develop SVM base classifiers)
  - base_rf.py (develop RF base classifiers)
  - base_xgboost.py (develop XGBoost base classifiers)
  - select_base.py (select the base classifiers based on the base classifiers' MCC rankings)
  - validation_predictions_combine.py (combine predicted probabilities for development set)
  - test_predictions_combine.py (combine predicted probabilities for test set)  
  - deepames_plus.py (make DeepAmes predictions)
  - main.py (the main script need to modity to run your own data using the DeepAmes framework)

### Model development environment
- Python: 3.7.3
- Numpy: 1.16.4
- Pandas: 0.24.2
- Scikit-learn: 0.21.2
- Keras: 2.2.4




