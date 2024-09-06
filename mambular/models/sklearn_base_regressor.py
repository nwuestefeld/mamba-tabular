import lightning as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import warnings
from ..base_models.lightning_wrapper import TaskModel
from ..data_utils.datamodule import MambularDataModule
from ..preprocessing import Preprocessor
from lightning.pytorch.callbacks import ModelSummary
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import warnings
from ..utils.config_mapper import (
    get_search_space,
    activation_mapper,
    round_to_nearest_16,
)


class SklearnBaseRegressor(BaseEstimator):
    def __init__(self, model, config, **kwargs):
        self.preprocessor_arg_names = [
            "n_bins",
            "numerical_preprocessing",
            "use_decision_tree_bins",
            "binning_strategy",
            "task",
            "cat_cutoff",
            "treat_all_integers_as_numerical",
            "knots",
            "degree",
        ]

        self.config_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.preprocessor_arg_names
        }
        self.config = config(**self.config_kwargs)

        preprocessor_kwargs = {
            k: v for k, v in kwargs.items() if k in self.preprocessor_arg_names
        }

        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.base_model = model
        self.task_model = None
        self.built = False

        # Raise a warning if task is set to 'classification'
        if preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. The Regressor is designed for regression tasks.",
                UserWarning,
            )

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        params.update(self.config_kwargs)

        if deep:
            preprocessor_params = {
                "preprocessor__" + key: value
                for key, value in self.preprocessor.get_params().items()
            }
            params.update(preprocessor_params)

        return params

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **parameters : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        config_params = {
            k: v for k, v in parameters.items() if not k.startswith("preprocessor__")
        }
        preprocessor_params = {
            k.split("__")[1]: v
            for k, v in parameters.items()
            if k.startswith("preprocessor__")
        }

        if config_params:
            self.config_kwargs.update(config_params)
            if self.config is not None:
                for key, value in config_params.items():
                    setattr(self.config, key, value)
            else:
                self.config = self.config_class(**self.config_kwargs)

        if preprocessor_params:
            self.preprocessor.set_params(**preprocessor_params)

        return self

    def build_model(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        dataloader_kwargs={},
    ):
        """
        Builds the model using the provided training data.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None. Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The validation target values. Required if `X_val` is provided.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=0.025
            Weight decay (L2 penalty) coefficient.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.



        Returns
        -------
        self : object
            The built regressor.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        if X_val:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

        self.data_module = MambularDataModule(
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            regression=True,
            **dataloader_kwargs,
        )

        self.data_module.preprocess_data(
            X, y, X_val, y_val, val_size=val_size, random_state=random_state
        )

        self.task_model = TaskModel(
            model_class=self.base_model,
            config=self.config,
            cat_feature_info=self.data_module.cat_feature_info,
            num_feature_info=self.data_module.num_feature_info,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=factor,
            weight_decay=weight_decay,
        )

        self.built = True

        return self

    def get_number_of_params(self, requires_grad=True):
        """
        Calculate the number of parameters in the model.

        Parameters
        ----------
        requires_grad : bool, optional
            If True, only count the parameters that require gradients (trainable parameters).
            If False, count all parameters. Default is True.

        Returns
        -------
        int
            The total number of parameters in the model.

        Raises
        ------
        ValueError
            If the model has not been built prior to calling this method.
        """
        if not self.built:
            raise ValueError(
                "The model must be built before the number of parameters can be estimated"
            )
        else:
            if requires_grad:
                return sum(
                    p.numel() for p in self.task_model.parameters() if p.requires_grad
                )
            else:
                return sum(p.numel() for p in self.task_model.parameters())

    def fit(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs={},
        rebuild=True,
        **trainer_kwargs,
    ):
        """
        Trains the regression model using the provided training data. Optionally, a separate validation set can be used.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None. Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The validation target values. Required if `X_val` is provided.
        max_epochs : int, default=100
            Maximum number of epochs for training.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before early stopping.
        monitor : str, default="val_loss"
            The metric to monitor for early stopping.
        mode : str, default="min"
            Whether the monitored metric should be minimized (`min`) or maximized (`max`).
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=0.025
            Weight decay (L2 penalty) coefficient.
        checkpoint_path : str, default="model_checkpoints"
            Path where the checkpoints are being saved.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted regressor.
        """
        if rebuild:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            if isinstance(y, pd.Series):
                y = y.values
            if X_val:
                if not isinstance(X_val, pd.DataFrame):
                    X_val = pd.DataFrame(X_val)
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values

            self.data_module = MambularDataModule(
                preprocessor=self.preprocessor,
                batch_size=batch_size,
                shuffle=shuffle,
                X_val=X_val,
                y_val=y_val,
                val_size=val_size,
                random_state=random_state,
                regression=True,
                **dataloader_kwargs,
            )

            self.data_module.preprocess_data(
                X, y, X_val, y_val, val_size=val_size, random_state=random_state
            )

            self.task_model = TaskModel(
                model_class=self.base_model,
                config=self.config,
                cat_feature_info=self.data_module.cat_feature_info,
                num_feature_info=self.data_module.num_feature_info,
                lr=lr,
                lr_patience=lr_patience,
                lr_factor=factor,
                weight_decay=weight_decay,
            )

        else:
            assert self.built, "The model must be built before calling the fit method."

        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # Adjust according to your validation metric
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,  # Specify the directory to save checkpoints
            filename="best_model",
        )

        # Initialize the trainer and train the model
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                early_stop_callback,
                checkpoint_callback,
                ModelSummary(max_depth=2),
            ],
            **trainer_kwargs,
        )
        self.trainer.fit(self.task_model, self.data_module)

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            self.task_model.load_state_dict(checkpoint["state_dict"])

        return self

    def predict(self, X):
        """
        Predicts target values for the given input samples.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.


        Returns
        -------
        predictions : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values.
        """
        # Ensure model and data module are initialized
        if self.task_model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")

        # Preprocess the data using the data module
        cat_tensors, num_tensors = self.data_module.preprocess_test_data(X)

        # Move tensors to appropriate device
        device = next(self.task_model.parameters()).device
        if isinstance(cat_tensors, list):
            cat_tensors = [tensor.to(device) for tensor in cat_tensors]
        else:
            cat_tensors = cat_tensors.to(device)

        if isinstance(num_tensors, list):
            num_tensors = [tensor.to(device) for tensor in num_tensors]
        else:
            num_tensors = num_tensors.to(device)

        # Set model to evaluation mode
        self.task_model.eval()

        # Perform inference
        with torch.no_grad():
            predictions = self.task_model(
                num_features=num_tensors, cat_features=cat_tensors
            )

        # Convert predictions to NumPy array and return
        return predictions.cpu().numpy()

    def evaluate(self, X, y_true, metrics=None):
        """
        Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The true target values against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are the metric functions.


        Notes
        -----
        This method uses the `predict` method to generate predictions and computes each metric.


        Examples
        --------
        >>> from sklearn.metrics import mean_squared_error, r2_score
        >>> from sklearn.model_selection import train_test_split
        >>> from mambular.models import MambularRegressor
        >>> metrics = {
        ...     'Mean Squared Error': mean_squared_error,
        ...     'R2 Score': r2_score
        ... }
        >>> # Assuming 'X_test' and 'y_test' are your test dataset and labels
        >>> # Evaluate using the specified metrics
        >>> results = regressor.evaluate(X_test, y_test, metrics=metrics)


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.
        """
        if metrics is None:
            metrics = {"Mean Squared Error": mean_squared_error}

        # Generate predictions using the trained model
        predictions = self.predict(X)

        # Initialize dictionary to store results
        scores = {}

        # Compute each metric
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = metric_func(y_true, predictions)

        return scores

    def score(self, X, y, metric=mean_squared_error):
        """
        Calculate the score of the model using the specified metric.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The true target values against which to evaluate the predictions.
        metric : callable, default=mean_squared_error
            The metric function to use for evaluation. Must be a callable with the signature `metric(y_true, y_pred)`.

        Returns
        -------
        score : float
            The score calculated using the specified metric.
        """
        predictions = self.predict(X)
        return metric(y, predictions)

    def optimize_hparams(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        time=100,
        max_epochs=200,
        prune_by_epoch=True,
        prune_epoch=5,
        **optimize_kwargs,
    ):
        """
        Optimizes hyperparameters using Bayesian optimization with optional pruning.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Training labels.
        X_val, y_val : array-like, optional
            Validation data and labels.
        time : int
            The number of optimization trials to run.
        max_epochs : int
            Maximum number of epochs for training.
        prune_by_epoch : bool
            Whether to prune based on a specific epoch (True) or the best validation loss (False).
        prune_epoch : int
            The specific epoch to prune by when prune_by_epoch is True.
        **optimize_kwargs : dict
            Additional keyword arguments passed to the fit method.

        Returns
        -------
        best_hparams : list
            Best hyperparameters found during optimization.
        """

        # Define the hyperparameter search space from the model config
        param_names, param_space = get_search_space(self.config)

        # Initial model fitting to get the baseline validation loss
        self.fit(X, y, X_val=X_val, y_val=y_val, max_epochs=max_epochs)
        best_val_loss = float("inf")

        if X_val is not None and y_val is not None:
            val_loss = self.evaluate(
                X_val, y_val, metrics={"Mean Squared Error": mean_squared_error}
            )["Mean Squared Error"]
        else:
            val_loss = self.trainer.validate(self.task_model, self.data_module)[0][
                "val_loss"
            ]

        best_val_loss = val_loss
        best_epoch_val_loss = self.task_model.epoch_val_loss_at(prune_epoch)

        def _objective(hyperparams):
            nonlocal best_val_loss, best_epoch_val_loss  # Access across trials

            head_layer_sizes = []

            for key, param_value in zip(param_names, hyperparams):
                if key == "head_layer_size_length":
                    head_layer_size_length = param_value
                elif key.startswith("head_layer_size_"):
                    head_layer_sizes.append(round_to_nearest_16(param_value))
                else:
                    field_type = self.config.__dataclass_fields__[key].type

                    # Check if the field is a callable (e.g., activation function)
                    if field_type == callable and isinstance(param_value, str):
                        if param_value in activation_mapper:
                            setattr(self.config, key, activation_mapper[param_value])
                        else:
                            raise ValueError(
                                f"Unknown activation function: {param_value}"
                            )
                    else:
                        setattr(self.config, key, param_value)

            # Truncate or use part of head_layer_sizes based on the optimized length
            if head_layer_size_length is not None:
                setattr(
                    self.config,
                    "head_layer_sizes",
                    head_layer_sizes[:head_layer_size_length],
                )

                print(head_layer_sizes)

            # Build the model with updated hyperparameters
            self.build_model(
                X, y, X_val=X_val, y_val=y_val, lr=self.config.lr, **optimize_kwargs
            )

            # Dynamically set the early pruning threshold
            if prune_by_epoch:
                early_pruning_threshold = (
                    best_epoch_val_loss * 1.5
                )  # Prune based on specific epoch loss
            else:
                early_pruning_threshold = (
                    best_val_loss * 1.5
                )  # Prune based on the best overall validation loss

            # Initialize the model with pruning
            self.task_model.early_pruning_threshold = early_pruning_threshold
            self.task_model.pruning_epoch = prune_epoch

            # Fit the model (limit epochs for faster optimization)
            self.fit(
                X, y, X_val=X_val, y_val=y_val, max_epochs=max_epochs, rebuild=False
            )

            # Retrieve the current validation loss
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(
                    X_val, y_val, metrics={"Mean Squared Error": mean_squared_error}
                )["Mean Squared Error"]
            else:
                val_loss = self.trainer.validate(self.task_model, self.data_module)[0][
                    "val_loss"
                ]

            # Retrieve validation loss at the specified epoch (e.g., epoch 5)
            epoch_val_loss = self.task_model.epoch_val_loss_at(prune_epoch)

            # Update the best validation loss at the specified epoch
            if prune_by_epoch and epoch_val_loss < best_epoch_val_loss:
                best_epoch_val_loss = epoch_val_loss

            # Update the best overall validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            return val_loss

        # Perform Bayesian optimization using scikit-optimize
        result = gp_minimize(_objective, param_space, n_calls=time, random_state=42)

        # Update the model with the best-found hyperparameters
        best_hparams = result.x
        if "head_layer_sizes" in self.config.__dataclass_fields__:
            head_layer_sizes = []

        # Iterate over the best hyperparameters found by optimization
        for key, param_value in zip(param_names, best_hparams):
            if key.startswith("head_layer_size_"):
                # These are the individual head layer sizes
                head_layer_sizes.append(round_to_nearest_16(param_value))
            else:
                # For all other config values, update normally
                field_type = self.config.__dataclass_fields__[key].type
                if field_type == callable and isinstance(param_value, str):
                    setattr(self.config, key, activation_mapper[param_value])
                else:
                    setattr(self.config, key, param_value)

        # After the loop, set head_layer_sizes in the config
        if head_layer_sizes:
            setattr(self.config, "head_layer_sizes", head_layer_sizes)

        print("Best hyperparameters found:", best_hparams)

        return best_hparams
