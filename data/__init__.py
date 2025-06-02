import data.img_transforms as T
import data.spatial_transforms as ST
import data.temporal_transforms as TT
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, TestImageDataset, VideoDataset
from data.samplers import RandomIdentitySampler, InferenceSampler
from data.datasets.celeb import Celeb
from data.datasets.syn import Syn

__factory = {
    'celeb': Celeb,
    'celeb_light': Celeb,
    'celeb_blur': Celeb,
    'celeb_light_blur': Celeb,
}

VID_DATASET = ['ccvid']


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))

    syn_dataset = Syn(root=config.DATA.ROOT)
    if config.DATA.DATASET in VID_DATASET:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT,
                                                 sampling_step=config.DATA.SAMPLING_STEP,
                                                 seq_len=config.AUG.SEQ_LEN,
                                                 stride=config.AUG.SAMPLING_STRIDE)
    else:
        if config.DATA.DATASET == 'celeb' or config.DATA.DATASET == 'celeb_light' or config.DATA.DATASET == 'celeb_blur' or config.DATA.DATASET == 'celeb_light_blur':
            dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, ds_name=config.DATA.DATASET)
        else:
            dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT)

    return syn_dataset, dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    return transform_train, transform_test


def build_vid_transforms(config):
    spatial_transform_train = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ST.RandomErasing(height=config.DATA.HEIGHT, width=config.DATA.WIDTH, probability=config.AUG.RE_PROB)
    ])
    spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
        temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
    elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
        temporal_transform_train = TT.TemporalRandomCrop(size=config.AUG.SEQ_LEN,
                                                         stride=config.AUG.SAMPLING_STRIDE)
    else:
        raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))

    temporal_transform_test = None

    return spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test


def build_dataloader(config):
    syn_dataset, dataset = build_dataset(config)
    # video dataset
    if config.DATA.DATASET in VID_DATASET:
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = build_vid_transforms(
            config)

        if config.DATA.DENSE_SAMPLING:
            train_sampler = RandomIdentitySampler(dataset.train_dense, num_instances=config.DATA.NUM_INSTANCES)
            # split each original training video into a series of short videos and sample one clip for each short video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train_dense, spatial_transform_train, temporal_transform_train),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        else:
            train_sampler = RandomIdentitySampler(dataset.train, num_instances=config.DATA.NUM_INSTANCES)
            # sample one clip for each original training video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train, spatial_transform_train, temporal_transform_train),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)

        _, transform_test = build_img_transforms(config)
        synloader = DataLoaderX(dataset=ImageDataset(syn_dataset.train, transform=transform_test),
                                    sampler=InferenceSampler(syn_dataset.train),
                                    batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False)

        # split each original test video into a series of clips and use the averaged feature of all clips as its representation
        queryloader = DataLoaderX(
            dataset=VideoDataset(dataset.recombined_query, spatial_transform_test, temporal_transform_test),
            sampler=InferenceSampler(dataset.recombined_query),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False)
        galleryloader = DataLoaderX(
            dataset=VideoDataset(dataset.recombined_gallery, spatial_transform_test, temporal_transform_test),
            sampler=InferenceSampler(dataset.recombined_gallery),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False)

        return trainloader, queryloader, galleryloader, dataset, train_sampler
    # image dataset
    else:
        transform_train, transform_test = build_img_transforms(config)
        train_sampler = RandomIdentitySampler(dataset.train, num_instances=config.DATA.NUM_INSTANCES)
        trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train),
                                  sampler=train_sampler,
                                  batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                  pin_memory=True, drop_last=False)

        synloader = DataLoaderX(dataset=ImageDataset(syn_dataset.train, transform=transform_test),
                                    sampler=InferenceSampler(syn_dataset.train),
                                    batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False)

        galleryloader = DataLoaderX(dataset=TestImageDataset(dataset.gallery, transform=transform_test),
                                    sampler=InferenceSampler(dataset.gallery),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)

        if config.DATA.DATASET == 'prcc':
            queryloader_same = DataLoaderX(dataset=TestImageDataset(dataset.query_same, transform=transform_test),
                                           sampler=InferenceSampler(dataset.query_same),
                                           batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                           pin_memory=True, drop_last=False, shuffle=False)
            queryloader_diff = DataLoaderX(dataset=TestImageDataset(dataset.query_diff, transform=transform_test),
                                           sampler=InferenceSampler(dataset.query_diff),
                                           batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                           pin_memory=True, drop_last=False, shuffle=False)

            return trainloader, synloader, queryloader_same, queryloader_diff, galleryloader, dataset
        else:
            queryloader = DataLoaderX(dataset=TestImageDataset(dataset.query, transform=transform_test),
                                      sampler=InferenceSampler(dataset.query),
                                      batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                      pin_memory=True, drop_last=False, shuffle=False)

            return trainloader, synloader, queryloader, galleryloader, dataset




