"""create dataset and dataloader"""
import logging
import torch.utils.data

def create_dataloader(dataset, dataset_opt, phase):
    """create dataloader"""
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    """create dataset"""
    mode = dataset_opt['mode']
    test_split = dataset_opt.get('test_split', 0.2)
    random_seed = dataset_opt.get('random_seed', 42)
    
    from data.LRHR import LRHRDataset as D
        
    # Create dataset with auto-split
    dataset = D(dataset_opt, phase, test_split, random_seed)
    
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(
        dataset.__class__.__name__,
        dataset_opt['name']))
    return dataset


# Original function
# def create_dataset(dataset_opt, phase):
#     """create dataset"""
#     mode = dataset_opt['mode']
#     from data.LRHR import LRHRDataset as D
#     # dataset = D(dataroot=dataset_opt['dataroot'],
#     #             datatype=dataset_opt['datatype'],
#     #             l_resolution=dataset_opt['l_resolution'],
#     #             r_resolution=dataset_opt['r_resolution'],
#     #             split=phase,
#     #             data_len=dataset_opt['data_len'],
#     #             need_LR=(mode == 'LRHR')
#     #             )
#     dataset = D(dataset_opt, phase)
#     logger = logging.getLogger('base')
#     logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
#                                                            dataset_opt['name']))
#     return dataset