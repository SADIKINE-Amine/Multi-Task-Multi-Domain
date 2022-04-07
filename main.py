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
from    torch.cuda              import device, is_available
from    arguments               import get_args
from    monai.apps              import CrossValidation
import  logging
from    Inference               import prediction
from    BatchSchedulerSampler   import BatchSchedulerSampler
from    utils                   import (Extract_Ids_From_DS, 
                                        Save_Experement_Info, 
                                        Genarate_LBBV_Dataset, print_ods,
                                        Add_Path_2_DataName, LossFunction, 
                                        Train, make_dir, load_checkpoint)

from    MultiDataLoader         import MultiSourceLoader
import  torch

args    = get_args()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if __name__ == "__main__":
    
    device  = torch.device("cuda" if is_available() else "cpu")
    logging.info('Using : {}'.format(device))

    make_dir(args.output_2_save)

    Save_Experement_Info(args)
    
    if args.generate_lbb:
        for dataset_name in args.DatasetName:
            make_dir(args.DataPath[dataset_name])
            Genarate_LBBV_Dataset(DatasetName= dataset_name, key_target=args.anatomy_target[dataset_name], size=args.spatial_size, Path2Save=args.DataPath[dataset_name])

    loss_function                    = LossFunction(args.loss_name)
    if args.cross_val:
        from datasets import MultiSourceCVDataset, GetMultiSourceCVDataset
        # To launch CV
        folds     = list(range(args.NBFolds))
        scores_dic= dict()
        for data_name in args.DatasetName:
            scores_dic[data_name]=np.zeros((args.NBFolds,7), dtype = np.float)

        tr_folds  = [folds[0: i] + folds[i+1:] for i in folds]

        MultiSourceCvDataset = MultiSourceCVDataset(Datsets=args.DatasetName, dataset_dir=args.DataPath, anatomy=args.anatomy_target, transform=Deftransforms(args.DatasetName), nfolds=args.NBFolds)
        logging.info('Cross validation of {}-folds ...'.format(args.NBFolds))
        for itr in folds:
            logging.info(f'Split {itr+ 1}/{args.NBFolds}')
            Output2SaveFold = args.output_2_save +f"/Split_{itr+1}"
            best_model_path = args.output_2_save +f"/Split_{itr+1}/" +"best_metric_model_dict.pth"
            make_dir(Output2SaveFold)
            
            #======================================| Training stage |==========================================

            train_ds        = GetMultiSourceCVDataset(Datsets=args.DatasetName, MultiSourceCVDataset=MultiSourceCvDataset, folds=folds, tr_folds=tr_folds, itr=itr, section="training")
            val_ds          = GetMultiSourceCVDataset(Datsets=args.DatasetName, MultiSourceCVDataset=MultiSourceCvDataset, folds=folds, tr_folds=tr_folds,itr=itr, section="validation", transform=Deftransforms(args.DatasetName))
            train_loader    = DataLoader(train_ds, sampler=BatchSchedulerSampler(dataset=train_ds, batch_size=args.batch_size), batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=list_data_collate, pin_memory=is_available())
            val_loader      = MultiSourceLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=list_data_collate)
            model           = Model(args.ModelName, device, args.spatial_size, args.norm, args.multi_domain)
            Train(train_loader, train_ds, val_loader, val_ds, model, loss_function, args.lr, args.epochs, device, args.spatial_size, Output2SaveFold, args.batch_size, args.Reg, args.Lambda)
            del train_loader, val_loader, train_ds

            #======================================| Test stage |===============================================

            best_model      = load_checkpoint(best_model_path, model, device)
            val_loader      = MultiSourceLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
            ids             = Extract_Ids_From_DS(Datsets=args.DatasetName, Ds=val_ds)# dictionnary of ids
            Means           = prediction(best_model, args.DatasetName, ids ,val_loader, args.anatomy_target, Output2SaveFold, device, True)
            del best_model, val_loader, ids

            for data_name in args.DatasetName:
                scores_dic[data_name][itr,:]   = Means[data_name]
            if itr == args.NBFolds-1: # We save results after the last split
                for data_name in args.DatasetName:
                    print_ods(scores_dic[data_name], np.arange(args.NBFolds)+1, args.output_2_save, '/Cross_Validation_Result'+data_name+'.ods')

    else:
        from    datasets                import MultiSourceDataset

        # To launch: train/ validation (default 20%)/ test (default 20%)
        #======================================| Training and validation stage |================================

        train_ds        = MultiSourceDataset(Datsets=args.DatasetName, dataset_dir=args.DataPath, section="training", anatomy=args.anatomy_target, transform=Deftransforms(args.DatasetName))
        train_loader    = DataLoader(train_ds, sampler=BatchSchedulerSampler(dataset=train_ds, batch_size=args.batch_size), batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=list_data_collate, pin_memory=is_available())
        val_ds          = MultiSourceDataset(Datsets=args.DatasetName, dataset_dir=args.DataPath, section="validation", anatomy=args.anatomy_target, transform=Deftransforms(args.DatasetName))
        val_loader      = MultiSourceLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        model           = Model(args.ModelName, device, args.spatial_size, args.norm, args.multi_domain)
        Train(train_loader, train_ds, val_loader, val_ds, model, loss_function, args.lr, args.epochs, device, args.spatial_size, args.output_2_save, args.batch_size, args.Reg, args.Lambda)
        del train_loader, val_loader, train_ds, val_ds

        #======================================| Test stage |===================================================

        best_model_path = args.output_2_save +"/best_metric_model_dict.pth"
        best_model      = load_checkpoint(best_model_path, model, device)
        test_ds         = MultiSourceDataset(Datsets=args.DatasetName, dataset_dir=args.DataPath, section="test", anatomy=args.anatomy_target, transform=Deftransforms(args.DatasetName))
        test_loader     = MultiSourceLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        ids             = Extract_Ids_From_DS(Datsets=args.DatasetName, Ds=test_ds)# dictionnary of ids
        prediction(best_model, args.DatasetName, ids ,test_loader, args.anatomy_target, args.output_2_save, device)
        del best_model, test_loader, ids

