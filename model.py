from monai.networks.nets import UNETR, UNet


def Model(model_name, device, size):
#model_name: - Unet / - Unetr
	if model_name=="Unet":	
		model = UNet(
		    spatial_dims=3,
		    in_channels=1,
		    out_channels=1,
		    channels=(16,32,64,128,256),
		    strides = (2,2,2,2),
		).to(device)
	elif model_name=="ResUnet":	
		model = UNet(
		    spatial_dims=3,
		    in_channels=1,
		    out_channels=1,
		    channels=(16,32,64,128,256),
		    strides = (2,2,2,2),
		    num_res_units=4,
		).to(device)
	elif model_name=="UnetrV0":
		model = UNETR(
		    in_channels=1,
		    out_channels=1,
		    img_size=size,
		    feature_size=8,
		    hidden_size=768,
		    mlp_dim=3072, 
		    num_heads=16,
		    pos_embed="perceptron",
		    norm_name="instance",
		    res_block=True,
		    dropout_rate=0.0,
		).to(device)
	elif model_name=="Unetr":
		model = UNETR(
		    in_channels=1,
		    out_channels=1,
		    img_size=size,
		    feature_size=16,
		    hidden_size=768,
		    mlp_dim=3072, 
		    num_heads=12,
		    pos_embed="perceptron",
		    norm_name='batch',
		    res_block=False,
		    dropout_rate=0.0,
		).to(device)
	else:
		raise ValueError("Check the name of the model ;)")

	return model