from typing import TYPE_CHECKING
from transformers.file_utils import _LazyModule, is_torch_available


_import_structure = {'configuration_ViTLP': ['ViTLPConfig']}
if is_torch_available():
    _import_structure['modeling_ViTLP'] = [
        'ViTLPLayer',
        'ViTLPModel',
        'ViTLPPreTrainedModel',
        'ViTLPForPreTraining',
        'ViTLPForQuestionAnswering',
        'ViTLPForSequenceClassification',
        'ViTLPForTokenClassification'
    ]


if TYPE_CHECKING:
    from .configuration_ViTLP import ViTLPConfig
    if is_torch_available():
        from .modeling_ViTLP import (
            ViTLPLayer,
            ViTLPModel,
            ViTLPPreTrainedModel,
            ViTLPForPreTraining,
            ViTLPForQuestionAnswering,
            ViTLPForSequenceClassification,
            ViTLPForTokenClassification
        )
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()['__file__'], _import_structure)
