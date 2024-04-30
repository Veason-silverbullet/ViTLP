from transformers.configuration_utils import PretrainedConfig


class ViTLPConfig(PretrainedConfig):
    model_type = 'ViTLP'
    def __init__(
        self,
        vocab_size=50267,
        hidden_size=1024,
        num_hidden_layers=18,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1280,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs
        )
