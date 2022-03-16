from monai.transforms 		import (
								    Activations,
								    AsChannelFirstd,
								    AsDiscrete,
								    Compose,
								    LoadImaged,
								    RandCropByPosNegLabeld,
								    RandRotate90d,
								    ScaleIntensityd,
								    EnsureTyped,
								    EnsureType,
								    AddChanneld,
								    Orientationd,
								    NormalizeIntensityd,
								    RandFlipd,
								    RandRotated,
								    RandAffined,
								    RandRotate90d,OneOf,
								    RandShiftIntensityd,
								    ToTensord)

def Deftransforms(Dataset='VEELA'):

	if Dataset=='VEELA':
	    train_transforms = Compose(
							[
					        LoadImaged(keys=["image", "label"]),
					        AddChanneld(keys=["image", "label"]),
					        Orientationd(keys=["image", "label"], axcodes="RAS"),
					        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
					        # RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10), 
        					# RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
					        # RandFlipd(
					        #     keys=["image", "label"],
					        #     spatial_axis=[2],
					        #     prob=0.10,
					        # ),
					        # RandShiftIntensityd(
					        #     keys=["image"],
					        #     offsets=0.10,
					        #     prob=0.50,
					        # ),
					        ToTensord(keys=["image", "label"]),
					    	]
						    )
	    val_transforms = Compose(
				    		[
					        LoadImaged(keys=["image", "label"]),
					        AddChanneld(keys=["image", "label"]),
					        Orientationd(keys=["image", "label"], axcodes="RAS"),
					        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
					        ToTensord(keys=["image", "label"])
							]
							)
	elif Dataset=='IRCAD':
	    train_transforms = Compose(
							[
					        LoadImaged(keys=["image", "label"]),
					        AddChanneld(keys=["image", "label"]),
					        Orientationd(keys=["image", "label"], axcodes="RAS"),
					        # OneOf([
					        # RandAffined(keys=['image', 'label'], prob=0.5, translate_range=(-10,10), mode=['bilinear','nearest'],padding_mode='zeros'), 
        					# RandAffined(keys=['image', 'label'], prob=0.5, rotate_range=(-10,10), mode=['bilinear','nearest'],padding_mode='zeros'),
        					# #RandAffined(keys=['image', 'label'], prob=0.5, scale_range=(0.7,0.9), mode=['bilinear','nearest'],padding_mode='zeros'), 
					        # #RandAffined(keys=['image', 'label'], prob=0.5, shear_range=(-10,10), mode=['bilinear','nearest'],padding_mode='zeros'),
					        # RandFlipd(
					        #     keys=["image", "label"],
					        #     spatial_axis=[2],
					        #     prob=0.10,
					        # )]),
					        # RandShiftIntensityd(
					        #     keys=["image"],
					        #     offsets=(10,20),
					        #     prob=0.50,
					        # ),
					        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
					        ToTensord(keys=["image", "label"]),
					    	]
						    )
	    val_transforms = Compose(
				    		[
					        LoadImaged(keys=["image", "label"]),
					        AddChanneld(keys=["image", "label"]),
					        Orientationd(keys=["image", "label"], axcodes="RAS"),
					        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
					        ToTensord(keys=["image", "label"])
							]
							)
	else:
		raise ValueError("Check the name of datset ;)")

	return train_transforms, val_transforms