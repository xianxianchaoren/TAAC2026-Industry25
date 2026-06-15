"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from item_oof_feature_engineering import (
    _DEFAULT_WINDOWS as ITEM_OOF_DEFAULT_WINDOWS,
    build_item_oof_artifacts,
    list_row_groups,
    split_train_valid_row_groups,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def build_aligned_user_pair_specs(
    user_int_schema: FeatureSchema,
    user_int_vocab_sizes: List[int],
    user_dense_schema: FeatureSchema,
    aligned_fids: List[int],
) -> List[dict]:
    """Build aligned ``user_int`` / ``user_dense`` specs for paired pooling."""
    dense_entries = {
        fid: (offset, length)
        for fid, offset, length in user_dense_schema.entries
    }

    pair_specs = []
    aligned_fid_set = set(aligned_fids)
    for fid, int_offset, int_length in user_int_schema.entries:
        if fid not in aligned_fid_set or fid not in dense_entries:
            continue
        dense_offset, dense_length = dense_entries[fid]
        if int_length != dense_length:
            logging.warning(
                "Skipping aligned fid=%s because int_length=%s != dense_length=%s",
                fid, int_length, dense_length)
            continue
        vocab_size = max(user_int_vocab_sizes[int_offset:int_offset + int_length])
        pair_specs.append({
            "fid": int(fid),
            "vocab_size": int(vocab_size),
            "int_offset": int(int_offset),
            "int_length": int(int_length),
            "dense_offset": int(dense_offset),
            "dense_length": int(dense_length),
        })
    return pair_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')
    parser.add_argument('--use_abs_pos', action='store_true', default=False,
                        help='Enable learned absolute positional encoding for sequences (0, 1, 2...)')
    parser.add_argument('--max_abs_pos', type=int, default=1024,
                        help='Maximum length for absolute positional encoding')
    parser.add_argument('--use_domain_emb', action='store_true', default=False,
                        help='Enable learned domain ID embedding to differentiate between sequence types')

    # Independent item OOF dense feature chain.
    parser.add_argument('--use_item_oof_dense', action='store_true', default=False,
                        help='Enable the independent item_oof_dense feature chain')
    parser.add_argument('--item_oof_num_folds', type=int, default=5,
                        help='Number of contiguous train row-group folds used for OOF item stats')
    parser.add_argument('--item_oof_hour_windows', type=str,
                        default=",".join(str(x) for x in ITEM_OOF_DEFAULT_WINDOWS),
                        help='Comma-separated hour windows for item OOF stats; current layout expects 1,3,6,24')
    parser.add_argument('--item_oof_smoothing_alpha', type=float, default=100.0,
                        help='Smoothing alpha used by item OOF CVR features')
    parser.add_argument('--item_oof_artifact_dir', type=str, default=None,
                        help='Directory to store generated item OOF artifacts (defaults to <ckpt_dir>/item_oof_artifacts)')
    parser.add_argument('--item_oof_dropout', type=float, default=0.35,
                        help='Training-only probability of replacing matched item OOF features with the default vector')
    parser.add_argument('--use_item_oof_gate', dest='use_item_oof_gate', action='store_true', default=True,
                        help='Gate item OOF token toward the stable item_dense token (default on)')
    parser.add_argument('--no_item_oof_gate', dest='use_item_oof_gate', action='store_false',
                        help='Disable item OOF gating')
    parser.add_argument('--item_oof_gate_hidden_dim', type=int, default=16,
                        help='Hidden dimension of the item OOF gate MLP')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--use_dense_guided_user_pooling', action='store_true', default=True,
                        help='Fuse an aligned user token built from paired '
                             'user_int/user_dense features via weighted pooling')
    parser.add_argument('--dense_guided_user_fids', type=str,
                        default='62,63,64,65,66,89,90,91',
                        help='Comma-separated user fids whose int/dense arrays '
                             'are aligned and participate in dense-guided pooling')
    parser.add_argument('--exclude_aligned_dense_feats', action='store_true', default=False,
                        help='Exclude the aligned user dense features from the general user_dense_proj')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Compile the model using torch.compile')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable bf16 automatic mixed precision')
    parser.add_argument('--use_ema', dest='use_ema', action='store_true', default=True,
                        help='Enable dense-only EMA for validation and checkpoint export (default on)')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false',
                        help='Disable dense-only EMA for validation and checkpoint export')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay used when --use_ema is enabled')
    parser.add_argument('--ema_warmup_steps', type=int, default=1000,
                        help='Number of optimizer steps to skip before EMA updates begin')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_item_oof_feature_table_paths_by_fold = None
    train_item_oof_fold_by_row_group = None
    valid_item_oof_feature_table_path = None
    item_oof_artifact_path = None
    if args.use_item_oof_dense:
        item_oof_artifact_dir = args.item_oof_artifact_dir or os.path.join(
            args.ckpt_dir, "item_oof_artifacts")
        hour_windows = tuple(
            int(part.strip()) for part in args.item_oof_hour_windows.split(",") if part.strip()
        )
        all_row_groups = list_row_groups(args.data_dir)
        train_row_groups, _ = split_train_valid_row_groups(
            all_row_groups,
            valid_ratio=args.valid_ratio,
            train_ratio=args.train_ratio,
        )
        logging.info(
            "Building item OOF artifacts: train_row_groups=%s, num_folds=%s, hour_windows=%s",
            len(train_row_groups), args.item_oof_num_folds, hour_windows)
        item_oof_artifact_path, train_item_oof_feature_table_paths_by_fold, train_item_oof_fold_by_row_group = (
            build_item_oof_artifacts(
                train_row_groups=train_row_groups,
                artifact_dir=item_oof_artifact_dir,
                num_folds=args.item_oof_num_folds,
                hour_windows=hour_windows,
                smoothing_alpha=args.item_oof_smoothing_alpha,
            )
        )
        valid_item_oof_feature_table_path = item_oof_artifact_path
        logging.info("Generated item OOF full artifact at %s", item_oof_artifact_path)

    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        train_item_oof_feature_table_paths_by_fold=train_item_oof_feature_table_paths_by_fold,
        train_item_oof_fold_by_row_group=train_item_oof_fold_by_row_group,
        valid_item_oof_feature_table_path=valid_item_oof_feature_table_path,
        item_oof_dropout=args.item_oof_dropout if args.use_item_oof_dense else 0.0,
        item_oof_smoothing_alpha=args.item_oof_smoothing_alpha,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)
    aligned_fids = [
        int(fid.strip()) for fid in args.dense_guided_user_fids.split(',')
        if fid.strip()
    ]
    aligned_user_pair_specs = build_aligned_user_pair_specs(
        pcvr_dataset.user_int_schema,
        pcvr_dataset.user_int_vocab_sizes,
        pcvr_dataset.user_dense_schema,
        aligned_fids,
    )
    if args.use_dense_guided_user_pooling:
        logging.info(
            "Dense-guided user pooling enabled for fids=%s, matched_pairs=%s",
            aligned_fids, [spec["fid"] for spec in aligned_user_pair_specs])

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "item_oof_dense_dim": pcvr_dataset.item_oof_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "use_abs_pos": args.use_abs_pos,
        "max_abs_pos": args.max_abs_pos,
        "use_domain_emb": args.use_domain_emb,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_dense_guided_user_pooling": args.use_dense_guided_user_pooling,
        "aligned_user_pair_specs": aligned_user_pair_specs,
        "exclude_aligned_dense_feats": args.exclude_aligned_dense_feats,
        "use_item_oof_gate": args.use_item_oof_gate,
        "item_oof_gate_hidden_dim": args.item_oof_gate_hidden_dim,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    if args.compile:
        model = torch.compile(model, mode='default')
    
    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        item_oof_artifact_path=item_oof_artifact_path,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        amp=args.amp,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_warmup_steps=args.ema_warmup_steps,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
