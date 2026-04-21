import numpy as np


for data_name in ["ETTm1", "istanbul_traffic", "synthetic_m", "synthetic_u"]:
    print(data_name)
    data = np.load(f"/playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/{data_name}/generated_text_caps.npy", allow_pickle=True)
    print(f"{len(data)} generated captions loaded from {data_name}")

    data = np.load(f"/playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/{data_name}/train_text_caps.npy", allow_pickle=True)
    print(f"{len(data)} real captions loaded from {data_name}")
    print("-"*40)

