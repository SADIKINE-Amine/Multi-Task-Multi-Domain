from    monai.config.type_definitions import PathLike
import  numpy               as      np
from    abc                 import  ABC, abstractmethod
from    monai.transforms    import  Randomizable, LoadImaged
from    monai.data          import  CacheDataset
from    typing              import  Callable, Dict, List, Optional, Sequence, Union
from    ipdb                import  set_trace
import  sys, os

class VEELA_Dataset(Randomizable, CacheDataset):
    def __init__(
        self,
        dataset_dir: PathLike,
        section: str,
        anatomy: str = "por",
        transform: Union[Sequence[Callable], Callable] = (),
        seed: int = 0,
        val_frac: float = 0.2,
        test_frac: float = 0.2,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 4):

        if not os.path.isdir(dataset_dir):
            raise ValueError("Root directory dataset_dir must be a directory.")
        self.section    = section
        self.val_frac   = val_frac
        self.test_frac  = test_frac
        self.anatomy    = anatomy
        self.set_random_state(seed=seed)

        dataset_dir     = dataset_dir+"/"

        self.datalist = [
        {"image": os.path.join(dataset_dir,"001-VE.nii.gz"), "label": os.path.join(dataset_dir,"001-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"002-VE.nii.gz"), "label": os.path.join(dataset_dir,"002-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"003-VE.nii.gz"), "label": os.path.join(dataset_dir,"003-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"004-VE.nii.gz"), "label": os.path.join(dataset_dir,"004-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"005-VE.nii.gz"), "label": os.path.join(dataset_dir,"005-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"006-VE.nii.gz"), "label": os.path.join(dataset_dir,"006-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"007-VE.nii.gz"), "label": os.path.join(dataset_dir,"007-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"008-VE.nii.gz"), "label": os.path.join(dataset_dir,"008-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"009-VE.nii.gz"), "label": os.path.join(dataset_dir,"009-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"010-VE.nii.gz"), "label": os.path.join(dataset_dir,"010-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"011-VE.nii.gz"), "label": os.path.join(dataset_dir,"011-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"012-VE.nii.gz"), "label": os.path.join(dataset_dir,"012-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"013-VE.nii.gz"), "label": os.path.join(dataset_dir,"013-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"014-VE.nii.gz"), "label": os.path.join(dataset_dir,"014-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"015-VE.nii.gz"), "label": os.path.join(dataset_dir,"015-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"016-VE.nii.gz"), "label": os.path.join(dataset_dir,"016-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"017-VE.nii.gz"), "label": os.path.join(dataset_dir,"017-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"018-VE.nii.gz"), "label": os.path.join(dataset_dir,"018-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"019-VE.nii.gz"), "label": os.path.join(dataset_dir,"019-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"020-VE.nii.gz"), "label": os.path.join(dataset_dir,"020-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"022-VE.nii.gz"), "label": os.path.join(dataset_dir,"022-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"023-VE.nii.gz"), "label": os.path.join(dataset_dir,"023-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"025-VE.nii.gz"), "label": os.path.join(dataset_dir,"025-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"027-VE.nii.gz"), "label": os.path.join(dataset_dir,"027-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"028-VE.nii.gz"), "label": os.path.join(dataset_dir,"028-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"030-VE.nii.gz"), "label": os.path.join(dataset_dir,"030-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"031-VE.nii.gz"), "label": os.path.join(dataset_dir,"031-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"032-VE.nii.gz"), "label": os.path.join(dataset_dir,"032-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"033-VE.nii.gz"), "label": os.path.join(dataset_dir,"033-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"034-VE.nii.gz"), "label": os.path.join(dataset_dir,"034-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"035-VE.nii.gz"), "label": os.path.join(dataset_dir,"035-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"036-VE.nii.gz"), "label": os.path.join(dataset_dir,"036-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"037-VE.nii.gz"), "label": os.path.join(dataset_dir,"037-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"039-VE.nii.gz"), "label": os.path.join(dataset_dir,"039-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"040-VE.nii.gz"), "label": os.path.join(dataset_dir,"040-VE-"+str(self.anatomy)+".nii.gz")}
        ]

        self.indices: np.ndarray = np.array([])
        data = self._split_datalist(self.datalist)

        if transform == ():
            transform = LoadImaged(["image", "label"])

        CacheDataset.__init__(self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def _generate_data_list(self, datalist: List[Dict]) -> List[Dict]:
        # the types of the item in data list should be compatible with the dataloader
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        
        length  = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)

        train_length    = int(length*(1-self.val_frac-self.test_frac))
        val_length      = int(length*self.val_frac)
        test_length     = int(length*self.test_frac)

        if self.section     == "training":
            self.indices    = indices[:train_length]
        elif self.section   == "validation":
            self.indices    = indices[train_length:train_length+val_length]
        elif self.section   == "test":
            self.indices    = indices[train_length+val_length::]
        else:
            raise ValueError(f"Unsupported section: {self.section}, ""available options are ['training', 'validation', 'test'].")
        self.indices.sort()
        return [datalist[i] for i in self.indices]

class IRCAD_Dataset(Randomizable, CacheDataset):
    def __init__(
        self,
        dataset_dir: PathLike,
        section: str,
        anatomy: str,
        transform: Union[Sequence[Callable], Callable] = (),
        seed: int           = 0,
        val_frac: float     = 0.2,
        test_frac: float    = 0.2,
        cache_num: int      = sys.maxsize,
        cache_rate: float   = 1.0,
        num_workers: int    = 4):

        if not os.path.isdir(dataset_dir):
            raise ValueError("Root directory dataset_dir must be a directory.")
        self.section    = section
        self.val_frac   = val_frac
        self.test_frac  = test_frac
        self.anatomy    = anatomy
        self.set_random_state(seed=seed)

        dataset_dir     = dataset_dir+"/"

        self.datalist = [
        {"image": os.path.join(dataset_dir,"01-VE.nii.gz"), "label": os.path.join(dataset_dir,"01-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"02-VE.nii.gz"), "label": os.path.join(dataset_dir,"02-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"03-VE.nii.gz"), "label": os.path.join(dataset_dir,"03-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"04-VE.nii.gz"), "label": os.path.join(dataset_dir,"04-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"05-VE.nii.gz"), "label": os.path.join(dataset_dir,"05-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"06-VE.nii.gz"), "label": os.path.join(dataset_dir,"06-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"07-VE.nii.gz"), "label": os.path.join(dataset_dir,"07-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"08-VE.nii.gz"), "label": os.path.join(dataset_dir,"08-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"09-VE.nii.gz"), "label": os.path.join(dataset_dir,"09-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"10-VE.nii.gz"), "label": os.path.join(dataset_dir,"10-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"11-VE.nii.gz"), "label": os.path.join(dataset_dir,"11-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"12-VE.nii.gz"), "label": os.path.join(dataset_dir,"12-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"13-VE.nii.gz"), "label": os.path.join(dataset_dir,"13-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"14-VE.nii.gz"), "label": os.path.join(dataset_dir,"14-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"15-VE.nii.gz"), "label": os.path.join(dataset_dir,"15-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"16-VE.nii.gz"), "label": os.path.join(dataset_dir,"16-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"17-VE.nii.gz"), "label": os.path.join(dataset_dir,"17-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"18-VE.nii.gz"), "label": os.path.join(dataset_dir,"18-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"19-VE.nii.gz"), "label": os.path.join(dataset_dir,"19-VE-"+str(self.anatomy)+".nii.gz")},
        {"image": os.path.join(dataset_dir,"20-VE.nii.gz"), "label": os.path.join(dataset_dir,"20-VE-"+str(self.anatomy)+".nii.gz")}
        ]

        self.indices: np.ndarray = np.array([])
        data = self._split_datalist(self.datalist)

        if transform == ():
            transform = LoadImaged(["image", "label"])

        CacheDataset.__init__(self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def _generate_data_list(self, datalist: List[Dict]) -> List[Dict]:
        # the types of the item in data list should be compatible with the dataloader
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        
        length  = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)

        train_length    = int(length*(1-self.val_frac-self.test_frac))
        val_length      = int(length*self.val_frac)
        test_length     = int(length*self.test_frac)

        if self.section     == "training":
            self.indices    = indices[:train_length]
        elif self.section   == "validation":
            self.indices    = indices[train_length:train_length+val_length]
        elif self.section   == "test":
            self.indices    = indices[train_length+val_length::]
        else:
            raise ValueError(f"Unsupported section: {self.section}, ""available options are ['training', 'validation', 'test'].")
        self.indices.sort()
        return [self.datalist[i] for i in self.indices]

def MultiSourceDataset(Datsets: List, dataset_dir: PathLike, section: str, anatomy: Dict, transform: Dict):
    
    from torch.utils.data.dataset import ConcatDataset
    
    DataDict    =  {}

    if "VEELA" in Datsets:
        DataDict["VEELA"]=VEELA_Dataset(dataset_dir=dataset_dir["VEELA"], section=section, anatomy=anatomy["VEELA"], transform=transform["validation" if section in ["validation", "test"]  else section]["VEELA"])
    else:
        raise ValueError("VEELA don't exist in datsets list")
    if "IRCAD" in Datsets:
        DataDict["IRCAD"]=IRCAD_Dataset(dataset_dir=dataset_dir["IRCAD"], section=section, anatomy=anatomy["IRCAD"], transform=transform["validation" if section in ["validation", "test"]  else section]["IRCAD"])
    else:
        raise ValueError("IRCAD don't exist in datsets list")

    return ConcatDataset([DataDict[Datsets_name] for Datsets_name in Datsets])







