import os
import math
import torch
import torch.nn.functional as F


def load_pretrained_bart(model, path, decoder_layers=6, hidden_dim=1024, bart_vocab_size=50265):
    print('Loading checkpoint', path)
    pretrain_state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
    def getattr_recursive(pointer, attrs):
        for attr in attrs.split('.'):
            pointer = getattr(pointer, attr)
        return pointer
    parameter_num = 0
    parameter_size = 0
    loaded_weights = set()
    for name, array in pretrain_state_dict.items():
        if 'lm_head' in name or '.k_proj.bias' in name or 'coder.embed_tokens.weight' in name or 'encoder.' in name:
            continue
        if name.startswith('decoder.layers.') and int(name.split('.')[2]) >= decoder_layers:
            continue
        array = array.to(torch.float32)
        if name == 'shared.weight':
            pointer = getattr_recursive(model, 'decoder.word_embeddings.weight')
            loaded_weights.add('decoder.word_embeddings.weight')
        else:
            pointer = getattr_recursive(model, name)
            loaded_weights.add(name)
        if array.shape == torch.Size([bart_vocab_size, hidden_dim]) and len(pointer.shape) == 2 and pointer.shape[1] == hidden_dim and pointer.shape[0] > bart_vocab_size: # for vocabulary loading
            pointer.data[:array.shape[0]] = array
            word_embedding_mean = pointer.mean(dim=0)
            word_embedding_std = pointer.std(dim=0)
            for i in range(bart_vocab_size, pointer.shape[0]):
                pointer.data[i] = torch.normal(mean=word_embedding_mean, std=word_embedding_std)
            print('Initialize ViTLP word embeddings of %s from pretrained BART word embeddings of %s' % (str(pointer.shape), str(array.shape)))
        else:
            if name != 'decoder.embed_positions.weight':
                assert pointer.shape == array.shape, 'Loading weight error of %s : %s vs. %s' % (name, str(pointer.shape), str(array.shape))
                pointer.data = array
            else:
                assert pointer.shape[1] == array.shape[1], 'Loading weight error of decoder.embed_positions.weight : word embedding size %d vs. %d' % (pointer.shape[1], array.shape[1])
                vitlp_max_seq = pointer.shape[0]
                bart_max_seq = array.shape[0]
                if vitlp_max_seq <= bart_max_seq - 2: # offset = 2, https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L128
                    pointer.data = array[2: vitlp_max_seq + 2, :].contiguous()
                else:
                    pointer.data = F.interpolate(array[2:,].t().unsqueeze(0), size=vitlp_max_seq, mode='linear', align_corners=False).squeeze(0).t().contiguous()
        parameter_num += 1
        parameter_size += array.numel()
    print('Loaded parameter number : %d\nLoaded parameter size : %.1fMB' % (parameter_num, parameter_size / 1024 / 1024 * 4))
    return model, loaded_weights


def load_pretrained_vit(model, path, layer_num=12, h0=60, w0=50):
    print('Loading checkpoint', path)
    pretrain_state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
    weight_map = {
        'embeddings.cls_token': 'encoder.cls_token',
        'embeddings.position_embeddings': 'encoder.position_embeddings',
        'embeddings.patch_embeddings.projection.weight': 'encoder.patch_embeddings.projection.weight',
        'embeddings.patch_embeddings.projection.bias': 'encoder.patch_embeddings.projection.bias',
        'layernorm.weight': 'encoder.layernorm.weight',
        'layernorm.bias': 'encoder.layernorm.bias'
    }
    for i in range(layer_num):
        # Q/K/V weights
        weight_map['encoder.layer.%d.attention.attention.key.weight' % i] = 'encoder.layers.%d.self_attn.k_proj.weight' % i
        weight_map['encoder.layer.%d.attention.attention.query.weight' % i] = 'encoder.layers.%d.self_attn.q_proj.weight' % i
        weight_map['encoder.layer.%d.attention.attention.query.bias' % i] = 'encoder.layers.%d.self_attn.q_proj.bias' % i
        weight_map['encoder.layer.%d.attention.attention.value.weight' % i] = 'encoder.layers.%d.self_attn.v_proj.weight' % i
        weight_map['encoder.layer.%d.attention.attention.value.bias' % i] = 'encoder.layers.%d.self_attn.v_proj.bias' % i
        # projection weights
        weight_map['encoder.layer.%d.attention.output.dense.weight' % i] = 'encoder.layers.%d.self_attn.out_proj.weight' % i
        weight_map['encoder.layer.%d.attention.output.dense.bias' % i] = 'encoder.layers.%d.self_attn.out_proj.bias' % i
        # intermediate and output weights
        weight_map['encoder.layer.%d.intermediate.dense.weight' % i] = 'encoder.layers.%d.intermediate.weight' % i
        weight_map['encoder.layer.%d.intermediate.dense.bias' % i] = 'encoder.layers.%d.intermediate.bias' % i
        weight_map['encoder.layer.%d.output.dense.weight' % i] = 'encoder.layers.%d.output.weight' % i
        weight_map['encoder.layer.%d.output.dense.bias' % i] = 'encoder.layers.%d.output.bias' % i
        # layernorm weights
        weight_map['encoder.layer.%d.layernorm_before.weight' % i] = 'encoder.layers.%d.layernorm_before.weight' % i
        weight_map['encoder.layer.%d.layernorm_before.bias' % i] = 'encoder.layers.%d.layernorm_before.bias' % i
        weight_map['encoder.layer.%d.layernorm_after.weight' % i] = 'encoder.layers.%d.layernorm_after.weight' % i
        weight_map['encoder.layer.%d.layernorm_after.bias' % i] = 'encoder.layers.%d.layernorm_after.bias' % i
    def getattr_recursive(pointer, attrs):
        for attr in attrs.split('.'):
            pointer = getattr(pointer, attr)
        return pointer
    parameter_num = 0
    parameter_size = 0
    for name, array in pretrain_state_dict.items():
        pointer = model
        if name[:4] == 'vit.':
            name = name[4:]
        if name in weight_map:
            pointer = getattr_recursive(pointer, weight_map[name])
            if name not in ['embeddings.cls_token', 'embeddings.position_embeddings']:
                array = array.squeeze(dim=0)
            if name != 'embeddings.position_embeddings':
                assert pointer.shape == array.shape, 'Loading weight error of %s : %s vs. %s' % (name, str(pointer.shape), str(array.shape))
                pointer.data = array
            elif pointer.shape == array.shape:
                pointer.data = array
            else:
                assert pointer.shape[0] == array.shape[0] and pointer.shape[2] == array.shape[2], 'VIT position embedding dimensions mismatch : %s vs. %s' % (str(pointer.shape), str(array.shape))
                pointer.data[:, 0, :] = array[:, 0, :]
                N = array.shape[1] - 1
                patch_pos_embed = array[:, 1:]
                dim = array.shape[-1]
                h0, w0 = h0 + 0.1, w0 + 0.1
                patch_pos_embed = F.interpolate(
                    patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                    scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
                    mode='bicubic',
                    align_corners=False
                )
                assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
                pointer.data[:, 1:, :] = patch_pos_embed
                print('Initialize ViTLP image position embeddings of %s from pretrained VIT image position embeddings of %s' % (str(pointer.shape), str(array.shape)))
            parameter_num += 1
            parameter_size += array.numel()
    assert len(weight_map) == parameter_num, 'VIT loading weight error: %d vs. %d' % (len(weight_map), parameter_num)
    print('Loaded parameter number : %d\nLoaded parameter size : %.1fMB' % (parameter_num, parameter_size / 1024 / 1024 * 4))
    return model, set(weight_map.values())
