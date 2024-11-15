import torch
from ..arch_utils.mamba_utils.mamba_arch import Mamba
from ..arch_utils.mlp_utils import MLPhead
from ..configs.mambular_config import DefaultMambularConfig
from .basemodel import BaseModel
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.mamba_utils.mamba_original import MambaOriginal


class Jambular(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultJambularConfig = DefaultJambularConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.pooling_method = self.hparams.get("pooling_method", config.pooling_method)
        self.shuffle_embeddings = self.hparams.get(
            "shuffle_embeddings", config.shuffle_embeddings
        )


        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )
        ## in Jamba arch anpassen 
        self.jamba = Jamba(config)

        self.tabular_head = MLPhead(
            input_dim=self.hparams.get("d_model", config.d_model),
            config=config,
            output_dim=num_classes,
        )

