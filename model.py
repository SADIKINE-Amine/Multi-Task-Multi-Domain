from Mymonai.UNet.UNet 		import UNet
from Mymonai.UNet.UNetV2 	import BasicUNet
from monai.networks.layers.factories import Norm
from ipdb import set_trace

def Model(model_name, device, size, norm, multi_domain_par):
#model_name: - Unet / - Unetr
	
	if norm=="IN": # instance normalization
		NORM=Norm.INSTANCE
	elif norm=="BN": # batch normalization
		NORM=Norm.BATCH
	else:
		raise ValueError("Check the name of Norm")

	if model_name=="Unet":	
		model = UNet(
		    spatial_dims=3,
		    in_channels=1,
		    out_channels=[1,1],
		    channels=(16,32,64,128,256),
		    strides = (2,2,2,2),
		    norm= NORM,
		    multi_domain_par=multi_domain_par,
		).to(device)
	elif model_name=="ResUnet":	
		model = UNet(
		    spatial_dims=3,
		    in_channels=1,
		    out_channels=1,
		    channels=(16,32,64,128,256),
		    strides = (2,2,2,2),
		    num_res_units=4,
		    norm= NORM,
		    multi_domain_par=multi_domain_par,
		).to(device)
	elif model_name=="BasicUNet": #support contrastive loss
		model = BasicUNet(
            spatial_dims=3,
            in_channels =1,
            out_channels=[1,1],
            features    = (32, 32, 64, 128, 256, 32),
            norm        = ("instance", {"affine": True}),
            multi_domain_par=multi_domain_par,
            latent_reduction="GM",
            upsample    = "deconv"
            ).to(device)
	else:
		raise ValueError("Check the name of the model ;)")

	return model