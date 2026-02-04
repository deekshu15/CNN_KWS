from datasets.kws_dataset import KWSDataset

ds = KWSDataset("data/metadata_fixed.csv", folder_id=1)
print("Samples:", len(ds))

mel, kw, y = ds[0]
print("Mel shape:", mel.shape)
print("Keyword length:", len(kw))
print("Positive frames:", y.sum().item())
