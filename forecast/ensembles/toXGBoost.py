import pickle
import warnings

import xgboost as xgb
from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore")


class XGBoost:
    """
	Input a dataframe 
	"""

    def __init__(self):
        """
			the given values for this must be a range for optimization
			where gamma and colsample range from 0 to 1 and max_depth must be integer
		"""

    def fit_model(
        self, X_train, y_train, max_depth_range, gamma_range, colsample_bytree_range
    ):
        """
		function to optimize parameter
		"""
        self.dtrain = xgb.DMatrix(X_train, label=y_train)

        def xgb_evaluate(max_depth, gamma, colsample_bytree):
            params = {
                "eval_metric": "rmse",
                "max_depth": int(max_depth),
                "subsample": 0.8,
                "eta": 0.1,
                "gamma": gamma,
                "colsample_bytree": colsample_bytree,
            }
            cv_result = xgb.cv(params, self.dtrain, num_boost_round=100, nfold=3)
            return -1.0 * cv_result["test-rmse-mean"].iloc[-1]

        # Use the expected improvement acquisition function to handle negative numbers
        # Optimally needs quite a few more initiation points and number of iterations
        self.xgb_bo = BayesianOptimization(
            xgb_evaluate,
            {
                "max_depth": max_depth_range,
                "gamma": gamma_range,
                "colsample_bytree": colsample_bytree_range,
            },
        )

        self.xgb_bo.maximize(init_points=3, n_iter=20, acq="ei")
        print(self.xgb_bo.max["params"])
        self.params = self.xgb_bo.max["params"]
        print(self.params)
        self.params["max_depth"] = int(self.params["max_depth"])
        # Train a new model with the best parameters from the search
        self.model = xgb.train(self.params, self.dtrain, num_boost_round=250)

    def predict(self, X):
        """
		final prediction with the params
		"""
        # Predict on testing and training set
        y = self.model.predict(xgb.DMatrix(X))
        return y

    def getModel(self):
        """
		   Method to return the model
		"""
        return self.model

    def save_model(self, weights_path):
        """
		save the model to a pickle file 
		"""
        pickle.dump(self.model, open(weights_path, "wb"))

    def load_model(self, weights_path):
        """
		load the model from a pickle file
		"""
        loaded_model = pickle.load(open(weights_path, "rb"))
        self.model = loaded_model
