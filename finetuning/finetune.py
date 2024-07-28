import sys
sys.path.append('..')
import os
import torch
from argparse import ArgumentParser
from transformers import get_scheduler, set_seed
from torch.utils.tensorboard import SummaryWriter
from models.ViTLP.configuration_ViTLP import ViTLPConfig
from models.ViTLP.modeling_ViTLP import ViTLPForPreTraining
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataset.pretrain import PretrainDataset
import deepspeed
import shutil


def save_checkpoint(model: deepspeed.DeepSpeedEngine, checkpoint_dir: str, is_main_process: bool, config_file: str):
    model.save_checkpoint(checkpoint_dir)
    dist.barrier()
    if is_main_process:
        shutil.copy(config_file, checkpoint_dir)
        state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        save_dir = os.path.join(checkpoint_dir, 'ViTLP')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(state_dict, os.path.join(save_dir, 'pytorch_model.bin'))
        shutil.copy(config_file, save_dir)


def train(args):
    # Step1: Initialize ViTLP checkpoint
    config = ViTLPConfig.from_pretrained(args.checkpoint)
    config.gradient_checkpointing = bool(args.gradient_checkpointing)
    config.load_vit = config.load_bart = False
    model = ViTLPForPreTraining.from_pretrained(args.checkpoint, config=config)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = model.cuda()
    model.train()

    # Step2: Prepare training data
    train_dataset = PretrainDataset(dataset_path=args.train_data_path, image_dir=args.image_dir, config=config, mode='train')
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)

    # Step3: Prepare optimizer
    no_decay = ['bias', 'layernorm', 'layer_norm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in filter(lambda x: x[1].requires_grad, model.named_parameters()) if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in filter(lambda x: x[1].requires_grad, model.named_parameters()) if any(nd in n for nd in no_decay)], 'weight_decay': 0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-7, betas=(0.9, 0.999))
    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * 0.025)
    scheduler = get_scheduler(name='linear', optimizer=optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    if args.is_main_process:
        print('Training steps:', num_training_steps)

    # Step4: Training
    model, _, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        dist_init_required=True,
        config=args.deepspeed_config
    )
    iteration = 0
    if args.is_main_process:
        writer = SummaryWriter(log_dir=args.output_dir, filename_suffix='.training-log')
        iteration_lm_loss, iteration_locate_loss, iteration_loss = 0, 0, 0

    for epoch_index in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch_index)
        for batch in train_dataloader:
            n1 = batch['n1']
            n2 = batch['n2']
            alpha = (n1.sum() / n2.sum())
            del batch['n1']
            del batch['n2']
            batch = tuple(t.cuda(non_blocking=True) for t in batch.values())
            inputs = {
                'image': batch[0],
                'decoder_input_ids': batch[1],
                'decoder_input_bboxes': batch[2],
                'labels': batch[3],
                'bboxes': batch[4]
            }
            with autocast():
                lm_loss, locate_loss = model(**inputs)
                loss = lm_loss + locate_loss * alpha
            model.backward(loss)
            model.step()

            iteration += 1
            if args.is_main_process:
                iteration_lm_loss += lm_loss.item()
                iteration_locate_loss += locate_loss.item()
                iteration_loss += loss.item()
                if args.log_interval > 0 and iteration % args.log_interval == 0:
                    iteration_loss /= args.log_interval
                    iteration_lm_loss /= args.log_interval
                    iteration_locate_loss /= args.log_interval
                    print('Iteration %d: lr = %.6f\tloss = %.3f\tlm_loss = %.3f\tlocate_loss = %.3f' % (iteration, lr_scheduler._last_lr[0], iteration_loss, iteration_lm_loss, iteration_locate_loss))
                    writer.add_scalar('Iteration loss', iteration_loss, iteration)
                    writer.add_scalar('Iteration lm_loss', iteration_lm_loss, iteration)
                    writer.add_scalar('Iteration locate_loss', iteration_locate_loss, iteration)
                    iteration_lm_loss, iteration_locate_loss, iteration_loss = 0, 0, 0
            if args.save_iteration > 0 and iteration % args.save_iteration == 0:
                save_checkpoint(model, os.path.join(args.output_dir, 'iteration-' + str(iteration)), args.is_main_process, os.path.join(args.checkpoint, 'config.json'))
        save_checkpoint(model, os.path.join(args.output_dir, 'epoch-' + str(epoch_index)), args.is_main_process, os.path.join(args.checkpoint, 'config.json'))
    if args.is_main_process:
        writer.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='ViTLP fintuning')
    parser.add_argument('--checkpoint', default='../ckpts/ViTLP-medium', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--train_data_path', default='text_bbox_data', type=str)
    parser.add_argument('--image_dir', default='text_bbox_data/images', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='outputs', type=str)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--save_iteration', default=1000, type=int)
    parser.add_argument('--deepspeed_config', default='zero1_fp16.json', type=str)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--gradient_checkpointing', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()
    if args.local_rank != -1:
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.local_rank in [-1, 0]:
        if args.local_rank == -1:
            args.is_main_process = True
            args.rank = 0
        elif args.rank == 0:
            args.is_main_process = True
        else:
            args.is_main_process = False
        print('******************************** Config ********************************')
        for k, v in dict(vars(args)).items():
            print('%s : %s' % (k, str(v)))
        print('******************************** Config ********************************')
    else:
        args.is_main_process = False
    if args.local_rank != -1:
        dist.init_process_group()
    set_seed(args.seed)
    train(args)
