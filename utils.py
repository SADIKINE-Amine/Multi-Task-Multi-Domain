import  torch
import  os
from    monai.losses        import  DiceLoss, GeneralizedDiceLoss, DiceCELoss
from    monai.metrics       import  DiceMetric
from    monai.transforms    import  AsDiscrete
import  medpy.metric.binary
import  numpy               as      np
import  nibabel             as      nib
from    ipdb                import  set_trace
from    monai.data          import  decollate_batch
from    monai.metrics       import  DiceMetric
from    monai.transforms    import  AsDiscrete, Activations, EnsureType, Compose
import  matplotlib.pyplot   as      plt
from    scipy.ndimage       import  zoom
from    tqdm                import  tqdm
import  logging
from    clDice.cldice       import soft_dice_cldice

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans  = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def Train(train_loader, train_ds, val_loader, val_ds, model , loss_function, lr, epochs, device, spatial_size, output_2_save):
    optimizer         = torch.optim.Adam(model.parameters(), lr)
    val_interval      =  1
    best_metric       = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values     = list()
    epoch_val_loss_values    = list()
    for epoch in range(epochs):
        logging.info('='*20)
        logging.info(f'epoch {epoch + 1}/{epochs}')
        model.train()
        model.multi_domain_par['state']=True
        epoch_loss  = 0
        step        = 0
        for batch_data in train_loader:
            step           += 1
            inputs, labels  = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs         = model(inputs)
            loss            = MultiDomainLossF(loss_function, outputs, labels)
            loss.backward()
            optimizer.step()
            # set_trace()
            epoch_loss      += loss.item()
            epoch_len        = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if (epoch + 1) % val_interval == 0:
            eval_metrics        = eval_net(model, val_loader, loss_function, device)
            metric              = eval_metrics['metric']
            epoch_val_loss      = eval_metrics['loss']
            epoch_val_loss_values.append(epoch_val_loss)
            metric_values.append(metric)
            if metric > best_metric:
                best_metric         = metric
                best_metric_epoch   = epoch + 1
                torch.save(model.state_dict(), output_2_save+'/'+"best_metric_model_dict.pth")
                logging.info("saved new best metric model")
            logging.info(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch))

    np.save(output_2_save+'/Train_loss.npy', np.array(epoch_loss_values))
    np.save(output_2_save+'/Val_dices.npy', np.array(metric_values))
    Plot_Curves(epoch_loss_values, epoch_val_loss_values, metric_values, output_2_save)

def MultiDomainLossF(loss_function, outputs, labels):
    Loss = 0
    bs      = labels.shape[0]
    mini_bs = int(bs/len(outputs))
    a       = 0
    b       = mini_bs
    for i, output in enumerate(outputs):
        if i!=0:
            a=b
            b=(i+1)*b 
        Loss+=loss_function(output, labels[a:b,:,:,:,:])
    return Loss/len(outputs)

def eval_net(model,loader, loss_function, device):
    model.eval()
    model.multi_domain_par['state']=False
    epoch_loss  = 0
    step        = 0
    T_metric    = 0
    T_epoch_loss= 0
    with torch.no_grad():
        for idx, Signle_Source_loader in enumerate(loader):
            model.multi_domain_par["domain_id"]=idx
            for data in Signle_Source_loader:
                step           += 1
                images, labels  = data["image"].to(device), data["label"].to(device)
                outputs         = model(images)
                loss            = loss_function(outputs, labels)
                outputs         = [post_trans(i) for i in decollate_batch(outputs)]
                epoch_loss     += loss.item()
                # compute metric for current iteration
                dice_metric(y_pred=outputs, y=labels)
            epoch_loss /= step
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            T_metric        += metric
            T_epoch_loss    += epoch_loss
    return {'metric':T_metric/len(loader), 'loss':T_epoch_loss/len(loader)}

def Plot_Curves(epoch_loss_values, epoch_val_loss_values, metric_values, output_2_save):
    
    plt.title("Loss")
    x   = [(i) for i in range(len(epoch_loss_values))]
    y1 = epoch_loss_values
    y2 = epoch_loss_values
    plt.plot(x, epoch_loss_values)
    plt.plot(x, epoch_val_loss_values)
    plt.legend(['train', 'val'], loc='upper right')
    plt.xlabel("Epochs")
    plt.grid() 
    plt.savefig(output_2_save+'/'+'loss.png')

    plt.close()
    plt.title("Val Dice")
    x = [(i) for i in range(len(metric_values))]
    y = metric_values
    plt.plot(x, y)
    plt.xlabel("Epochs")
    plt.ylim([0,1])
    plt.grid() 
    plt.savefig(output_2_save+'/'+'Val_dices.png')
    plt.close()

def Genarate_LBBV_Dataset(DatasetName= "VEELA", key_target='por', size=(96,96,96), Path2Save="/home/sadikine/data/PreVEELA"):
    """
    LBBV:Liver Bounding Box Volume
    Read volumes of interest after applying a bounding box around the liver as nii file.
    Resizing with a specific size 
    Save the vollume as nii file
    ===================================================================================
    DatasetName     : VEELA or IRCAD
    key_target      : keys of segementation target (Por,Hep)
    """
    if DatasetName      =="VEELA":
        from data_dic import VEELA_DIC
        info_dict = VEELA_DIC().data_dict
    elif DatasetName    =="IRCAD":
        from data_dic import IRCAD_DIC
        info_dict = IRCAD_DIC().data_dict
    else:
        raise ValueError("Check the name of datset ;)")
    key_target=[key_target]
    key_target.append('VE')
    make_dir(Path2Save)
    logging.info("Genarating Liver Bounding Box Volume for {} dataset...".format(DatasetName))
    for Volume_of_interest in key_target:
        logging.info("Genarating {} volumes".format(Volume_of_interest))
        for idx, path in tqdm(enumerate(info_dict[Volume_of_interest])):
            nii_obj = nib.as_closest_canonical(nib.load(path))
            Vol     = nii_obj.get_fdata()
            # 3D indexing volume_images
            Vol = Vol[
                info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1],
                info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3],
                info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5]
            ]
            if Vol.shape[2]<size[2]:
                arr   = np.zeros((Vol.shape[0],Vol.shape[1],size[2]), dtype = Vol.dtype)
                arr[:,:,:Vol.shape[2]] = Vol
                Vol       = arr
                del arr
                scale     = (size[0]/Vol.shape[0], size[1]/Vol.shape[1], 1.)
            else:
                scale     = (size[0]/Vol.shape[0], size[1]/Vol.shape[1], size[2]/Vol.shape[2])
            #scale             = (size[0]/Vol.shape[0], size[1]/Vol.shape[1], size[2]/Vol.shape[2])
            if Volume_of_interest=='VE':
                Vol           = zoom(Vol, scale, order=3)
                nii_Vol       = nib.Nifti1Image(Vol, nii_obj.affine, nii_obj.header)
                nib.save(nii_Vol, Path2Save+'/'+ os.path.basename(path))
            else:
                Vol[np.where(Vol != 0)]   = 1
                Vol           = zoom(Vol, scale, order=0)
                nii_Vol       = nib.Nifti1Image(Vol, nii_obj.affine, nii_obj.header)
                nib.save(nii_Vol, Path2Save+'/'+ os.path.basename(path))
    logging.info("End of generation =D")

def Add_Path_2_DataName(dict_, mode, data_path):
    dictionary = dict_[mode]
    for idx in np.arange(len(dictionary)):
        dictionary[idx]['image']=data_path+'/'+dictionary[idx]['image']
        dictionary[idx]['label']=data_path+'/'+dictionary[idx]['label']
    return dictionary

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory")
    else:
        print ("Successfully created the directorie")

def Extract_Ids_From_Dic(dic):
    ids = []
    for i in np.arange(len(dic)):
        id_= int(dic[i]['image'].split('/')[-1].split('-')[0])
        ids.append(id_)
    return ids

def printF(P):
    M=[]
    print('-------------')
    for j in np.arange(len(P)):
        M.append(P[j]['image'])
    print(sorted(M))
    print('-------------')

def assessment(result, groundtruth, out_spacing):
    ''' complete assessment '''
    dice = 100.*medpy.metric.binary.dc(result.astype(np.bool), groundtruth.astype(np.bool))
    sens = 100.*medpy.metric.binary.sensitivity(result.astype(np.bool), groundtruth.astype(np.bool))
    spec = 100.*medpy.metric.binary.specificity(result.astype(np.bool), groundtruth.astype(np.bool))
    jacc = 100.*medpy.metric.binary.jc(result.astype(np.bool), groundtruth.astype(np.bool))
    
    xspacing = out_spacing[0]
    yspacing = out_spacing[1]    
    zspacing = out_spacing[2] 

    avd = np.abs(medpy.metric.binary.ravd(result.astype(np.bool), groundtruth.astype(np.bool)))
    connectivity = 2
    if len(np.unique(result.astype(np.bool))) > 1:
        assd = medpy.metric.binary.assd(result.astype(np.bool), groundtruth.astype(np.bool), (xspacing, yspacing, zspacing), connectivity)
        mssd = medpy.metric.binary.hd(result.astype(np.bool), groundtruth.astype(np.bool), (xspacing, yspacing, zspacing), connectivity)
    else:
        assd = -1
        mssd = -1
    return np.array([dice, sens, spec, jacc, avd, assd, mssd])

def print_ods(scores, test_ids, output, name):

    resfile = open(output + name, "a")
    resfile.write('exam\t' + 'dice\t' + 'sens\t' + 'spec\t' + 'jacc\t' + 'avd\t' + 'assd\t' + 'mssd\t\n')
    
    for index, id_ in enumerate(test_ids):
        resfile.write('%0*d' % (3, id_) + '\t')
        for idx in range(scores.shape[1]):
            resfile.write(str('%.3f' % scores[index, idx]).replace(".", ",") + '\t')
        resfile.write('\n')
    resfile.write('mean\t')

    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f' % np.mean(scores[:, idx])).replace(".", ",") + '\t')
    
    resfile.write('\nstd\t')
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f' % np.std(scores[:, idx])).replace(".", ",") + '\t')
    resfile.write('\n')
    resfile.close()

def load_checkpoint(filepath, net, device):
    checkpoint = torch.load(filepath, map_location= device)
    net.load_state_dict(checkpoint)
    return net

def Save_Experement_Info(args):
    file = open(args.output_2_save+'/Experement_Info.txt','w')
    file.write('Network name                    : %s\n'%(args.ModelName))
    file.write('Data name                       : %s\n'%(args.DatasetName))
    file.write('Spatial size                    : %s\n'%(str(args.spatial_size)))
    file.write('Epochs                          : %d\n'%(args.epochs))
    file.write('Batch size                      : %d\n'%(args.batch_size))
    file.write('Learning rate                   : %f\n'%(args.lr))
    file.write('Loss                            : %s\n'%(args.loss_name))
    file.close()

def LossFunction(loss_name="DiceLoss"):
    if loss_name=="DiceLoss":
        return DiceLoss(sigmoid=True)

    elif loss_name=="GeneralizedDiceLoss":
        return GeneralizedDiceLoss(sigmoid=True)
    
    elif loss_name=="DiceCELoss":
        return DiceCELoss(sigmoid=True)

    elif loss_name=="DiceCldiceLoss":
        return soft_dice_cldice(iter_=5, alpha=0.3, sigmoid=True)
    
    else:
        raise ValueError("Check the name of the loss function ;)")