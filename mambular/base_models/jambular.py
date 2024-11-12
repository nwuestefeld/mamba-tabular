import torch
import torch.nn as nn
from ..arch_utils.jamba_arch import Jamba
from ..arch_utils.mlp_utils import MLP
from ..arch_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
from ..configs.mambular_config import DefaultMambularConfig
from .basemodel import BaseModel
from ..arch_utils.embedding_layer import EmbeddingLayer


class Jambular(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultMambularConfig = DefaultMambularConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        
        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.pooling_method = self.hparams.get("pooling_method", config.pooling_method)
        self.shuffle_embeddings = self.hparams.get(
            "shuffle_embeddings", config.shuffle_embeddings
        )
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        self.jamba = Jamba()
        ##hier muss noch was hin



    def forward(self, num_features, cat_features):
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        num_features : Tensor
            Tensor containing the numerical features.
        cat_features : Tensor
            Tensor containing the categorical features.

        Returns
        -------
        Tensor
            The output predictions of the model.
        """
        x = self.embedding_layer(num_features, cat_features)
        x = self.jamba(x)
        x = torch.mean(x, dim=1)
        x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
