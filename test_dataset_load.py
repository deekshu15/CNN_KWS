from CNN_KWS.datasets.kws_dataset import KWSDataset
from CNN_KWS.utils.keyword_encoder import char2idx

ds = KWSDataset(
    metadata_csv="data/metadata_sample.csv",
    char2idx=char2idx
)

print("Samples:", len(ds))

mel, kw, y = ds[0]
print("Mel shape:", mel.shape)
print("Keyword length:", len(kw))
print("Positive frames:", y.sum().item())