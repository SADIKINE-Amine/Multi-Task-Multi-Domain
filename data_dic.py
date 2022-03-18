import  nibabel             as      nib
import  skimage.transform   as      skTrans
import  numpy               as      np               
from    ipdb                import  set_trace
import  os

VEELA_DATSET_PATH = "/home/sadikine/data/VEELA"
IRCAD_DATSET_PATH = "/home/sadikine/data/IRCAD/data"
VEELA_IDS         = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40]
IRCAD_IDS         = list(np.arange(1,21))

class VEELA_DIC:
    def __init__(self, ids=VEELA_IDS, dataset_path=VEELA_DATSET_PATH, load_VE_liv=True, load_VE_por=True, load_VE_hep=False):
        self.data_path      = dataset_path
        self.ids            = ids
        self.load_VE_liv    = load_VE_liv
        self.load_VE_por    = load_VE_por
        self.load_VE_hep    = load_VE_hep
        self.data_dict      = {}

        if not os.path.isdir(self.data_path):
            raise ValueError("data_path must (VEELA_Dataset) be a directory.")

        for id_ in self.ids:
            VE_path                = self.data_path+ '/'+ '%0*d'%(3,id_)+'-VE.nii.gz'
            self.append_value_to_dict(data_dict=self.data_dict, key='VE', value=VE_path)
            if self.load_VE_liv:
                VE_liv_path        = self.data_path+ '/'+ '%0*d'%(3,id_)+'-VE-liv.nii.gz'
                self.append_value_to_dict(data_dict=self.data_dict, key='Liv', value=VE_liv_path)
                self.append_value_to_dict(data_dict=self.data_dict, key='Liver coordinates', value=self.liver_bounding_box(nib.as_closest_canonical(nib.load(VE_liv_path)).get_fdata()))
            if self.load_VE_por:
                VE_por_path        = self.data_path+ '/'+ '%0*d'%(3,id_)+'-VE-por.nii.gz'
                self.append_value_to_dict(data_dict=self.data_dict, key='por', value=VE_por_path)
            if self.load_VE_hep:
                VE_hep_path        = self.data_path+ '/'+ '%0*d'%(3,id_)+'-VE-hep.nii.gz'
                self.append_value_to_dict(data_dict=self.data_dict, key='hep', value=VE_hep_path)

    def append_value_to_dict(self,data_dict, key, value):
        # Check if key exist in dict or not
        if key in self.data_dict:
            # Key exist in dict.
            # Check if type of value of key is list or not
            if not isinstance(self.data_dict[key], list):
                # If type is not list then make it list
                self.data_dict[key] = [self.data_dict[key]]
            # Append the value in list
            self.data_dict[key].append(value)
        else:
            # As key is not in dict,
            # so, add key-value pair
            self.data_dict[key] = value
    def liver_bounding_box(self, liver):
        X, Y, Z                             = np.where(liver > 0.)
        return np.min(X), np.max(X)+1, np.min(Y), np.max(Y)+1, np.min(Z), np.max(Z)+1

class IRCAD_DIC:
    def __init__(self, ids=IRCAD_IDS, dataset_path=IRCAD_DATSET_PATH, load_VE_liv=True, load_VE_por=True, load_VE_hep=False):
        self.data_path      = dataset_path
        self.ids            = ids
        self.load_VE_liv    = load_VE_liv
        self.load_VE_por    = load_VE_por
        self.load_VE_hep    = load_VE_hep
        self.data_dict      = {}

        if not os.path.isdir(self.data_path):
            raise ValueError("data_path must (IRCAD_Dataset) be a directory.")

        for id_ in self.ids:
            VE_path                = self.data_path+ '/'+ '%0*d'%(2,id_)+'-VE.nii.gz'
            self.append_value_to_dict(data_dict=self.data_dict, key='VE', value=VE_path)
            if self.load_VE_liv:
                VE_liv_path        = self.data_path+ '/'+ '%0*d'%(2,id_)+'-VE-liv.nii.gz'
                self.append_value_to_dict(data_dict=self.data_dict, key='Liv', value=VE_liv_path)
                self.append_value_to_dict(data_dict=self.data_dict, key='Liver coordinates', value=self.liver_bounding_box(nib.as_closest_canonical(nib.load(VE_liv_path)).get_fdata()))
            if self.load_VE_por:
                VE_por_path        = self.data_path+ '/'+ '%0*d'%(2,id_)+'-VE-por.nii.gz'
                self.append_value_to_dict(data_dict=self.data_dict, key='por', value=VE_por_path)
            if self.load_VE_hep:
                VE_hep_path        = self.data_path+ '/'+ '%0*d'%(2,id_)+'-VE-hep.nii.gz'
                self.append_value_to_dict(data_dict=self.data_dict, key='hep', value=VE_hep_path)

    def append_value_to_dict(self,data_dict, key, value):
        # Check if key exist in dict or not
        if key in self.data_dict:
            # Key exist in dict.
            # Check if type of value of key is list or not
            if not isinstance(self.data_dict[key], list):
                # If type is not list then make it list
                self.data_dict[key] = [self.data_dict[key]]
            # Append the value in list
            self.data_dict[key].append(value)
        else:
            # As key is not in dict,
            # so, add key-value pair
            self.data_dict[key] = value
    def liver_bounding_box(self, liver):
        X, Y, Z                             = np.where(liver > 0.)
        return np.min(X), np.max(X)+1, np.min(Y), np.max(Y)+1, np.min(Z), np.max(Z)+1


