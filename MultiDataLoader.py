from torch.utils.data        import DataLoader

def MultiSourceLoader(Multi_Source_Dataset, batch_size, shuffle, num_workers, collate_fn):
	dataloader=[]
	for ds in Multi_Source_Dataset.datasets:
		loader      = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
		dataloader.append(loader)
	return dataloader