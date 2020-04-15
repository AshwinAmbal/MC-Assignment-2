import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def PCAReduction(data):
    minmax = MinMaxScaler()
    features = minmax.fit_transform(data)

    pc = PCA(0.95)
    pca = pc.fit(features)

    principal_components = pca.transform(features)

    columns = ['principal_components_' + str(i + 1) for i in range(principal_components.shape[1])]

    final_pca = pd.DataFrame(principal_components, columns=columns)

    # final_df = pd.concat([final_pca, label], axis=1)
    final_df = final_pca.copy()

    return final_df, pca, minmax
