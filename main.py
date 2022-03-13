import  os
import  shutil
import  tempfile
import  matplotlib.pyplot       as plt
import  numpy                   as np
from    model                   import Model
from    monai.data              import Dataset, list_data_collate
from    torch.utils.data        import DataLoader
from    ipdb                    import set_trace
from    transforms              import Deftransforms
import  torch
from    arguments               import get_args
from    datasets                import VEELA_Dataset
from    monai.apps              import CrossValidation
import  logging
from    Inference               import prediction
from    utils                   import (Extract_Ids_From_Dic, 
                                        Save_Experement_Info, 
                                        Genarate_LBBV_Dataset, print_ods,
                                        Add_Path_2_DataName, LossFunction, 
                                        Train, make_dir, load_checkpoint)

args    = get_args()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if args.DatasetName     =="VEELA":
    from datasets import VEELA_Dataset as Dataset
elif args.DatasetName   =="IRCAD":
    from datasets import IRCAD_Dataset as Dataset
else:
    raise ValueError("Check the name of datset ;)")

if __name__ == "__main__":
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('Using : {}'.format(device))
    make_dir(args.output_2_save)
    Save_Experement_Info(args)
    if args.generate_lbb:
        make_dir(args.DataPath)
        Genarate_LBBV_Dataset(DatasetName= args.DatasetName, key_target=args.anatomy_target, size=args.spatial_size, Path2Save=args.DataPath)
    train_transforms, val_transforms = Deftransforms(Dataset=args.DatasetName)
    loss_function                    = LossFunction(args.loss_name)
    if args.cross_val:
        # To launch CV
        folds     = list(range(args.NBFolds))
        scores    = np.zeros((args.NBFolds,7), dtype = np.float)
        tr_folds  = [folds[0: i] + folds[i+1:] for i in folds]
        cvdataset = CrossValidation(
                                    dataset_cls=Dataset,
                                    nfolds=5,
                                    seed=0,
                                    dataset_dir=args.DataPath,
                                    section="training",
                                    transform=train_transforms)

        logging.info('Cross validation of {}-folds ...'.format(args.NBFolds))
        for itr in folds:
            logging.info(f'Split {itr+ 1}/{args.NBFolds}')
            Output2SaveFold = args.output_2_save +f"/Split_{itr+1}"
            best_model_path = args.output_2_save +f"/Split_{itr+1}/" +"best_metric_model_dict.pth"
            make_dir(Output2SaveFold)
            
            #======================================| Training stage |==========================================
            
            train_ds        = cvdataset.get_dataset(folds=tr_folds[itr])
            val_ds          = cvdataset.get_dataset(folds=itr, transform=val_transforms)
            train_loader    = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
            val_loader      = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4, collate_fn=list_data_collate)
            model           = Model(args.ModelName, device, args.spatial_size)
            Train(train_loader, train_ds, val_loader, val_ds, model, loss_function, args.lr, args.epochs, device, args.spatial_size, Output2SaveFold)
            del train_loader, val_loader, train_ds

            #======================================| Test stage |===============================================

            best_model      = load_checkpoint(best_model_path, model, device)
            val_loader      = DataLoader(val_ds, batch_size=1, shuffle=False,num_workers=4, collate_fn=list_data_collate)
            ids             = Extract_Ids_From_Dic(val_ds.data)
            Means           = prediction(best_model, args.DatasetName, ids ,val_loader, args.anatomy_target, Output2SaveFold, device, True)
            del best_model, val_loader, ids
            scores[itr,:]   = Means
            if itr == args.NBFolds-1: # We save results after the last split
                print_ods(scores, np.arange(args.NBFolds)+1, args.output_2_save, '/Cross_Validation_Result.ods')
    else:
        # To launch: train/ validation (default 20%)/ test (default 20%)
        #======================================| Training and validation stage |================================

        train_ds        = Dataset(dataset_dir=args.DataPath, section="training",transform=train_transforms)
        train_loader    = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
        val_ds          = Dataset(dataset_dir=args.DataPath, section="validation",transform=val_transforms)
        val_loader      = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        model           = Model(args.ModelName, device, args.spatial_size)
        Train(train_loader, train_ds, val_loader, val_ds, model, loss_function, args.lr, args.epochs, device, args.spatial_size, args.output_2_save)
        del train_loader, val_loader, train_ds, val_ds

        #======================================| Test stage |===================================================

        best_model_path = args.output_2_save +"/best_metric_model_dict.pth"
        best_model      = load_checkpoint(best_model_path, model, device)
        test_ds         = Dataset(dataset_dir=args.DataPath, section="test",transform=val_transforms)

        test_loader     = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        ids             = Extract_Ids_From_Dic(test_ds.data)
        prediction(best_model, args.DatasetName, ids ,test_loader, args.anatomy_target,args.output_2_save, device)
        del best_model, test_loader, ids

