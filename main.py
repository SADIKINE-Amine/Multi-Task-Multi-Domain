import  os
import  shutil
import  tempfile
import  matplotlib.pyplot       as plt
import  numpy                   as np
from    model                   import Model
from    monai.data              import list_data_collate
from    torch.utils.data        import DataLoader
from    ipdb                    import set_trace
from    transforms              import Deftransforms
import  torch
from    arguments               import get_args
from    monai.apps              import CrossValidation
import  logging
from    Inference               import prediction
from    BatchSchedulerSampler   import BatchSchedulerSampler
from    utils                   import (Extract_Ids_From_Dic, 
                                        Save_Experement_Info, 
                                        Genarate_LBBV_Dataset, print_ods,
                                        Add_Path_2_DataName, LossFunction, 
                                        Train, make_dir, load_checkpoint)

from    datasets                import MultiSourceDataset

args    = get_args()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

Dataset = MultiSourceDataset(Datsets=args.DatasetName, dataset_dir=args.DataPath, section="training", anatomy=args.anatomy_target, transform=Deftransforms(args.DatasetName))
if __name__ == "__main__":
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('Using : {}'.format(device))
    set_trace()

    make_dir(args.output_2_save)
    Save_Experement_Info(args)
    
    if args.generate_lbb:
        make_dir(args.DataPath)
        Genarate_LBBV_Dataset(DatasetName= args.DatasetName, key_target=args.anatomy_target, size=args.spatial_size, Path2Save=args.DataPath)
    train_transforms, val_transforms = Deftransforms(Dataset=args.DatasetName)
    loss_function                    = LossFunction(args.loss_name)
    if args.cross_val:
        print("cross_val")

    else:
        # To launch: train/ validation (default 20%)/ test (default 20%)
        #======================================| Training and validation stage |================================

        train_ds        = MultiSourceDataset(Datsets=args.DatasetName, dataset_dir=args.DataPath, section="training", anatomy=args.anatomy_target, transform=Deftransforms(args.DatasetName))
        train_loader    = DataLoader(Dataset, sampler=BatchSchedulerSampler(dataset=Dataset, batch_size=args.batch_size), batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
        val_ds          = MultiSourceDataset(Datsets=args.DatasetName, dataset_dir=args.DataPath, section="validation", anatomy=args.anatomy_target, transform=Deftransforms(args.DatasetName))
        val_loader      = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        model           = Model(args.ModelName, device, args.spatial_size, args.norm, args.multi_domain)
        set_trace()
        Train(train_loader, train_ds, val_loader, val_ds, model, loss_function, args.lr, args.epochs, device, args.spatial_size, args.output_2_save)
        del train_loader, val_loader, train_ds, val_ds

