import  argparse


def get_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o',       '--output_2_save',      type=str,       default= "./test",          dest='output_2_save')

    parser.add_argument('-dn',      '--DatasetName',        type=str,       default= "IRCAD",           dest='DatasetName') #VEELA, IRCAD 

    parser.add_argument('-dp',      '--DataPath',           type=str,       default= "/home/sadikine/data/PreVEELA",          dest='DataPath')

    parser.add_argument('-k',       '--anatomy_target',     type=list,      default=['Por'],            dest='anatomy_target')

    parser.add_argument('-lbb',     '--generate_lbb',       type=bool,      default=False,              dest='generate_lbb')

    parser.add_argument('-n',       '--ModelName',          type=str,       default="Unet",             dest='ModelName')

    parser.add_argument('-ls',      '--loss_name',          type=str,       default="DiceCldiceLoss",        dest='loss_name')# GeneralizedDiceLoss, DiceLoss, DiceCELoss, DiceCldiceLoss
       
    parser.add_argument('-cv',      '--cross_val',          type=bool,      default=False,              dest='cross_val')

    parser.add_argument('-nf',      '--NBFolds',            type=int,       default=5,                  dest='NBFolds')

    parser.add_argument('-e',       '--epochs',             type=int,       default=50,                 dest='epochs')
                
    parser.add_argument('-b',       '--batch_size',         type=int,       default=1,                  dest='batch_size')
    
    parser.add_argument('-l',       '--learning',           type=float,     default=1e-3,               dest='lr')
        
    parser.add_argument('-de',      '--decay',              type=float,     default=0.,                 dest='decay')
    
    parser.add_argument('-s',       '--spatial_size',       type=int,       default=(256, 256, 128),    dest='spatial_size')

    parser.add_argument('-sp',      '--spacing',            type=list,      default=None,               dest='spacing')
    
    return parser.parse_args()