Introduction: The objective of the project is to predict bike rental demand on an hourly basis using historical usage patterns and weather data. This dataset for this problem was provided by Hadi Fanaee Tork using data from Capital Bikeshare.

Importing the Libraries: We import Numpy, Pandas, Scikit Learn,  Matplotlib and os libraries and set the random seed to 42 to ensure repeatability of execution. 

Loading the data: We load the data using Pandas and explore the metadata using info and describe methods

Performing EDA on the datasets: We look for missing values, observe categorical and numerical features, and scale of the various features

Cleaning the dataset: We drop columns from the dataset which we find are unnecessary or irrelevant for the model

Dividing Dataset into Train and Test Set: We split the data into Training and Test data sets into 70:30 ratio using scikit-learn's train_test_split() function. Train_test_split() function uses 'Random Sampling' for dividing the data.

Feature Scaling the Dataset: The values of some of the features like temp, hum and wind speed are in different scales (ranges), hence, we apply scaling to these features values for our ML algorithms to work fine on them. We use the Standard Scaler class of the Scikit-learn library for the same

Training various models on the Dataset: We import DecisionTreeRegressor, RandomForestRegressor, and LinearRegression from Scikit learn. We train those models on the training data set using cross-validation and calculate mean absolute error and root mean squared error (RMSE) for all. The Random Forest Regressor performs best (with a root mean squared error of 67.23) and is selected for further fine tuning.

Fine Tuning the Selected Model: We import GridSearchCV from Scikit Learn and choose the set of hyperparameter combinations in the form of a parameter grid. We then apply Grid Search on the Random Forest Regressor model to fine-tune the model and find the best hyperparameters

Knowing Feature Importances: We get the relative importance of each feature of the training data set by using feature_importances_ variable of the grid_search object.

Evaluating the Model: Since we got the best final model using Grid Search for this problem, we use the same on the test data set to predict the bike hourly demand values and then compare the predicted values to the actual values. We get a final RMSE of 39.48. We also plot the predicted values against the actual values for better visual representation of the model’s performance
