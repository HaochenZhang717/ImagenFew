from data_provider.datasets import ETTh, ETTm, Custom, UEA, GLUONTS, Sine, Stock, Energy, Mujoco, PSM, MSL, SMAP, SMD, SWAT, AirQuality, AIREADIGlucose
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset, TensorDataset
from .multi_dataloader_iter import MultiDataloaderIter
import functools
import torch

data_dict = {
    'ETTh1': ETTh,
    'ETTh2': ETTh,
    'ETTm1': ETTm,
    'ETTm2': ETTm,
    'custom': Custom,
    'PSM': PSM,
    'MSL': MSL,
    'SMAP': SMAP,
    'SMD': SMD,
    'SWAT': SWAT,
    'UEA': UEA,
    'gluonts': GLUONTS,
    'sine': Sine,
    'stock': Stock,
    'energy': Energy,
    'mujoco': Mujoco,
    'AirQuality': AirQuality,
    'AIREADIGlucose': AIREADIGlucose,
}

def random_permute(trainset, testset):
    perm_train = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(0)).numpy()
    perm_test = torch.randperm(len(testset), generator=torch.Generator().manual_seed(0)).numpy()

    return Subset(trainset, perm_train), Subset(testset, perm_test)

def random_subset(dataset, subset_p=None, subset_n=None):
    num_samples = len(dataset)
    if subset_n is not None:
        num_samples = subset_n
    if subset_p is not None:
        num_samples = int(num_samples * subset_p)
    indices = torch.arange(0, num_samples, dtype=int) % len(dataset)
    return Subset(dataset, indices)

def data_provider(args):

    trainsets = {}
    testsets  = {}
    trainloaders = {}
    testloaders  = {}
    samplers = {}
    metadatas  = {}

    datasets = [dataset for dataset in args.datasets if dataset['name'] in args.train_on_datasets]

    for config in datasets:
        metadata = {}
        config['seq_len'] = args.seq_len
        config['datasets_dir'] = args.datasets_dir
        trainset, testset = get_train(config), get_train(config)
        subset_p = getattr(args,'subset_p', None)
        subset_n = getattr(args,'subset_n', None)

        # Randomly permute train/testsets
        trainset, testset = random_permute(trainset, testset)
        if (subset_n is not None or subset_p is not None) and (not 'subset_n' in config.keys()):
            trainset, testset = random_subset(trainset, subset_p, subset_n), trainset
        trainset, testset = dataset_to_tensor(trainset, args), dataset_to_tensor(testset, args)

        caption_embeddings_path = getattr(args, "caption_embeddings_path", None)
        if caption_embeddings_path and config["name"] in args.train_on_datasets:
            caption_embeddings = torch.load(caption_embeddings_path, map_location="cpu", weights_only=False)
            if not torch.is_tensor(caption_embeddings):
                raise TypeError(
                    f"caption_embeddings_path must point to a tensor, got {type(caption_embeddings)}"
                )
            if caption_embeddings.ndim != 3:
                raise ValueError(
                    f"Expected caption embeddings with shape (N, C, D), got {tuple(caption_embeddings.shape)}"
                )
            if caption_embeddings.shape[0] != trainset.shape[0]:
                raise ValueError(
                    f"Caption embedding count ({caption_embeddings.shape[0]}) does not match "
                    f"train set size ({trainset.shape[0]}) for dataset {config['name']}."
                )
            trainset = TensorDataset(trainset, caption_embeddings.to(torch.float32))

        if args.finetune:
            train_seq_tensor = trainset.tensors[0] if isinstance(trainset, TensorDataset) else trainset
            assert train_seq_tensor.size(1) == args.seq_len, f"{config['name']} Does not output proper sequence length"
        else:
            train_seq_tensor = trainset.tensors[0] if isinstance(trainset, TensorDataset) else trainset
            if train_seq_tensor.size(1) != args.seq_len:
                train_seq_tensor = torch.nn.functional.pad(train_seq_tensor, (0, 0, 0, args.seq_len - train_seq_tensor.size(1)))
                if isinstance(trainset, TensorDataset):
                    trainset = TensorDataset(train_seq_tensor, trainset.tensors[1])
                else:
                    trainset = train_seq_tensor
        print(f"{config['name']} Contains: {len(trainset)} train datapoints; {len(testset)} test datapoints;")

        metadata['name'] = config['name']
        metadata['channels'] = (trainset.tensors[0] if isinstance(trainset, TensorDataset) else trainset).size(-1)

        if args.input_channels is not None:
            trainset = torch.nn.functional.pad(trainset, (0, args.input_channels - trainset.size(2), 0, 0))

        if config['name'] in args.train_on_datasets:
            trainsets[config['name']] = trainset
            if getattr(args, 'ddp', False):
                samplers[config['name']] = DistributedSampler(trainsets[config['name']])
            else:
                samplers[config['name']] = None
            trainloaders[config['name']] = (DataLoader(dataset=trainsets[config['name']], batch_size=args.batch_size, num_workers=args.num_workers, sampler=samplers[config['name']]), metadata)
        testsets[config['name']] = testset
        metadatas[config['name']] = metadata

    args.input_channels = functools.reduce(lambda acc, metadata: max(acc, metadata['channels']), metadatas.values(), args.input_channels if args.input_channels is not None else 1)
    dataset_loader = MultiDataloaderIter(trainloaders, testsets)
    return dataset_loader, samplers, trainsets, metadatas


def get_train(config):
    Data = data_dict[config['data']]
    config['flag'] = 'train'
    if 'subset_n' in config.keys():
        return Subset(Data(**config), torch.arange(config['subset_n']))
    return Data(**config)

def get_test(config):
    Data = data_dict[config['data']]
    config['flag'] = 'test'
    return Data(**config)

def dataset_to_tensor(dataset, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    tensor = []
    for i, item in enumerate(loader):
        if type(item) is list:
            adjusted_item = item[0][:, :args.seq_len]
            tensor.append(adjusted_item)
        else:
            adjusted_item = item[:, :args.seq_len]
            tensor.append(adjusted_item)
    dataset = torch.concat(tensor, dim=0)
    return dataset
