import ml_collections
import os
from transformers import PretrainedConfig, AutoConfig


def create_config(args):
    config = ml_collections.ConfigDict()

    config.work_dir = os.getcwd()
    
    training = config.training = ml_collections.ConfigDict()
    training.accum_batch_steps = 1
    training.training_iters = 200_000 * training.accum_batch_steps
    training.training_iters = training.training_iters
    training.checkpoint_freq = 25_000 * training.accum_batch_steps
    training.eval_freq = 25_000 * training.accum_batch_steps
    training.batch_size = 512 // training.accum_batch_steps
    training.ode_sampling = False
    training.checkpoints_folder = f"{config.work_dir}/checkpoints/"
    training.checkpoint_name = ""
    
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5000 * training.accum_batch_steps
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1000
    validation.num_gen_texts = 5000
    validation.texts_path = f"{config.work_dir}/generated_texts"
    validation.cfg_coef = 0.

    dynamic = config.dynamic = ml_collections.ConfigDict()
    dynamic.solver = 'euler'
    dynamic.scheduler = args.scheduler
    dynamic.N = 50
    dynamic.beta_min = 0.1
    dynamic.beta_max = 20
    dynamic.ode_sampling = False
    dynamic.coef_d = args.coef_d

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.downstream_task = ""
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.encoder_name = args.encoder_name

    if "bert" in model.encoder_name.lower():
        model.encoder_link = 'google-bert/bert-base-cased'
    elif "roberta" in model.encoder_name.lower():
        model.encoder_link = 'FacebookAI/roberta-base'
    elif "t5" in model.encoder_name.lower():
        model.encoder_link = 'google-t5/t5-base'
    elif "bart" in model.encoder_name.lower():
        model.encoder_link = 'facebook/bart-base'

    model.conditional_encoder_name = model.encoder_name
    model.encoder_name_hash = model.encoder_name.replace("/", "-")
    model.conditional_encoder_name_hash = model.conditional_encoder_name.replace("/", "-")

    data = config.data = ml_collections.ConfigDict()
    data.datasets = create_datasets_config(args)
    data.base_path = f"{config.work_dir}/datasets"
    data.max_sequence_len = get_sequence_len(data.datasets.datasets_list[0])
    data.max_context_len = get_context_len(data.datasets.datasets_list[0])
    data.path = ""
    data.swap_cfg_coef = args.swap_cfg_coef
    data.enc_gen_mean = f"{data.base_path}/{data.datasets.datasets_list[0]}/statistics/encodings-{model.encoder_name_hash}-mean.pt"
    data.enc_gen_std = f"{data.base_path}/{data.datasets.datasets_list[0]}/statistics/encodings-{model.encoder_name_hash}-std.pt"

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.use_self_cond = True
    config.is_conditional = False if 'rocstories' in data.datasets.datasets_list or 'wikipedia' in data.datasets.datasets_list else True
    config.emb = args.emb

    decoder = config.decoder = create_decoder_config() 
    decoder.dataset = data.datasets.datasets_list[0]
    decoder.name = f"decoder-{model.encoder_name_hash}-128-transformer"
    decoder.name += decoder.suffix
    decoder.is_conditional = config.is_conditional
    if decoder.is_conditional:
        decoder.name += "-conditional"
    if config.emb:
        decoder.name += "-emb"
    decoder.decoder_path = f"{data.base_path}/{data.datasets.datasets_list[0]}/{decoder.name}.pth"
    if decoder.max_sequence_len < data.max_sequence_len:
        raise Exception("Decoder max_sequence_len is less than required")

    config.se_config = create_se_config()
    config.se_config.is_conditional = config.is_conditional
    config.se_config.vocab_size = AutoConfig.from_pretrained(model.encoder_link).vocab_size
    config.se_config.use_self_cond = config.use_self_cond

    config.project_name = args.project_name
    config.timesteps = "linear"
    pref = "emb" if config.emb else "tencdm"
    training.checkpoints_prefix = f"{pref}-{model.encoder_name_hash}-{training.batch_size}-{optim.lr}-{data.datasets.datasets_list[0]}-cfg={data.swap_cfg_coef}"
    config.eval = False
    
    config.tracked_dataset = data.datasets.datasets_list[0]
    config.tracked_metric = data.datasets.metrics[config.tracked_dataset]["tracked_metric"]
    config.higher_better = True
    config.save_top_k = 2
    return config


def create_se_config():
    se_config = AutoConfig.from_pretrained("bert-base-cased")
    se_config.attention_head_size = se_config.hidden_size / se_config.num_attention_heads
    return se_config


def create_datasets_config(args):
    config = ml_collections.ConfigDict()
    config.downstream_tasks = ["qqp", "xsum", "paradetox", "wiki_auto"]
    if args.dataset_name is None:
        config.datasets_list = ["rocstories"]
    else:
        config.datasets_list = [args.dataset_name]
    config.metrics = {
        "rocstories": {"metrics": ["mauve", "div", "ppl"],
                       "tracked_metric": "mauve"},
        "wikipedia": {"metrics": ["mauve", "div", "ppl"],
                       "tracked_metric": "mauve"},
        "qqp": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        },
        "xsum": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        },
        "wiki_auto": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        },
    }
    return config


def create_decoder_config():
    config = ml_collections.ConfigDict()

    config.max_sequence_len = 128
    config.noise_sigma = 0.2
    config.lr = 1e-4
    config.betas = (0.9, 0.98)
    config.weight_decay = 0.001
    config.batch_size = 64
    config.epochs = 1
    config.max_norm = 1.0
    config.is_conditional = False
    config.dataset = ""
    config.T = 0.15
    config.eps = 0.001
    config.diffusion_forward = True
    config.suffix = ""
    config.num_hidden_layers = 3
    
    return config


def get_sequence_len(dataset_name):
    data = {
        "wikipedia": 128,
        "rocstories": 80,
        "qqp": 50,
        "xsum": 64,
        "wiki_auto": 100,
    }
    return data[dataset_name]


def get_context_len(dataset_name):
    data = {
        "wikipedia": 128,
        "rocstories": 80,
        "qqp": 50,
        "xsum": 512,
        "wiki_auto": 100,
    }
    return data[dataset_name]

    
