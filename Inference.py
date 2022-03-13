import  logging
import  os
import  numpy                   as np
import  torch
import  tqdm
from    skimage                 import img_as_ubyte
import  nibabel
from    skimage.transform       import resize, rotate
from    skimage.exposure        import rescale_intensity
from    skimage.segmentation    import mark_boundaries
import  xlrd
import  medpy.metric.binary
import  cv2
from    ipdb                    import set_trace
import  matplotlib.pyplot       as plt
import  nibabel                 as nib
from    monai.transforms        import AsDiscrete, Activations, EnsureType, Compose
from    utils                   import make_dir, assessment, print_ods
from    monai.data              import decollate_batch
from    utils                   import Extract_Ids_From_Dic
from    ipdb                    import set_trace
import  skimage.transform       as skTrans
from    scipy.ndimage           import zoom


post_trans  = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def prediction(model, datset_name, ids, test_loader, anatomy_target, output, device, cv = False):
    
    if datset_name      =="VEELA":
        from data_dic import VEELA_DIC
        info_dict = VEELA_DIC(ids=ids).data_dict
    elif datset_name    =="IRCAD":
        from data_dic import IRCAD_DIC
        info_dict = IRCAD_DIC(ids=ids).data_dict
    else:
        raise ValueError("Check the name of datset ;)")

    make_dir(output+'/prediction')
    make_dir(output+'/prediction/nii_visualization')
    scores    = np.zeros((len(ids),7), dtype = np.float)
    itr       = iter(test_loader)
    model.eval()
    for index, id_ in enumerate(tqdm.tqdm(ids)):
        
        logging.info('Exam N°: {}'.format(id_))
        with torch.no_grad():
            test_data = next(itr)
            val_images, val_labels = test_data["image"].to(device), test_data["label"].to(device)
            pred    = model(val_images)
            pred    = post_trans(decollate_batch(pred))
        pred        = pred[0].squeeze().cpu().numpy()
        
        Gt_Vol, pred, spacing   = Reshape_Pred_2_Original_Volume_VEELA(info_dict, index, anatomy_target, pred, output)
        logging.info('Calculating metrics for exam N°: {}'.format(id_))
        scores[index,:]         = assessment(pred, Gt_Vol, spacing)
        del Gt_Vol, pred
        logging.info(f'''dice for exam {id_}: dice: {scores[index,0]}''')
        
    print_ods(scores, ids, output, '/prediction/overview-results-all.ods')
    if cv:
        return np.mean(scores,0)
    else:
        return None

def Reshape_Pred_2_Original_Volume_VEELA(info_dict, idx, anatomy_target, pred, output):
    output+= '/prediction/nii_visualization'
    anatomy_target.append('VE')
    for Volume_of_interest in anatomy_target:
        path    = info_dict[Volume_of_interest][idx]
        nii_obj = nib.as_closest_canonical(nib.load(path))
        #print(nib.aff2axcodes(nii_obj.affine))
        spacing = [abs(nii_obj.get_qform()[0,0]), abs(nii_obj.get_qform()[1,1]),
                    abs(nii_obj.get_qform()[2,2])]
        Vol     = nii_obj.get_fdata()

        Vol = Vol[
            info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1],
            info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3],
            info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5]
        ]
        if Volume_of_interest=='VE':
            nii_Vol       = nib.Nifti1Image(Vol, nii_obj.affine, nii_obj.header)
            nib.save(nii_Vol, output+'/'+ os.path.basename(path))
        else:
            Vol[np.where(Vol != 0)]   = 1
            nii_Vol         = nib.Nifti1Image(Vol, nii_obj.affine, nii_obj.header)
            nib.save(nii_Vol, output+'/'+ os.path.basename(path))
            pred_name       = os.path.basename(path)
            Gt_Vol          = Vol

            Orig_size     = (info_dict['Liver coordinates'][idx][1]-info_dict['Liver coordinates'][idx][0],
                            info_dict['Liver coordinates'][idx][3]-info_dict['Liver coordinates'][idx][2], 
                            info_dict['Liver coordinates'][idx][5]-info_dict['Liver coordinates'][idx][4])

            if Orig_size[2] < pred.shape[2]:
                pred      = pred[:,:,:Orig_size[2]]
                scale     = (Orig_size[0]/pred.shape[0], Orig_size[1]/pred.shape[1], 1.)
            else:
                scale     = (Orig_size[0]/pred.shape[0], Orig_size[1]/pred.shape[1], Orig_size[2]/pred.shape[2])
            pred          = zoom(pred, scale, order=0)
            nii_Vol       = nib.Nifti1Image(pred, nii_obj.affine, nii_obj.header)
            nib.save(nii_Vol, output+'/'+ 'Pred-'+pred_name)

    return Gt_Vol.astype(np.uint8), pred.astype(np.uint8), spacing


# if shape[2]>z:
#     scale       = (self.size/shape[0], self.size/shape[1], z/shape[2])
# else:
#     scale       = (self.size/shape[0], self.size/shape[1], 1.)
#     arr_Volume      = np.zeros((shape[0],shape[1],z), dtype = Volume.dtype)
#     arr_Volume[:,:,:shape[2]]    = Volume
#     Volume          = arr_Volume
#     del arr_Volume
#     arr_GT_Volume   = np.zeros((shape[0],shape[1],z), dtype = GT_Volume.dtype)
#     arr_GT_Volume[:,:,:shape[2]] = GT_Volume
#     GT_Volume       = arr_GT_Volume
#     del arr_GT_Volume
# Volume      = ndimage.zoom(Volume, scale, order=3)
# GT_Volume   = ndimage.zoom(GT_Volume, scale, order=0)