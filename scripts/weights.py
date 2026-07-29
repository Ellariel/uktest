import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from libpysal.weights import full2W, Queen, KNN


def compute_weights(
    y, x, data, shape_data, method="inverse_distance", k=8, verbose=True
):
    coords = shape_data[["LAT", "LONG"]].to_numpy()
    if method == "inverse_distance":
        dist_mat = distance_matrix(coords, coords)
        np.fill_diagonal(dist_mat, np.inf)  # avoid division by zero
        w = 1 / dist_mat  # Inverse distance weights
    elif method == "queen":
        w = Queen.from_dataframe(shape_data, use_index=False)
        if len(w.islands) > 0:
            if verbose:
                print(f"There are some islands in the data {method}({len(w.islands)}).")
            w_knn = KNN.from_array(coords, k=1)
            for i in w.islands:  # use knn neighbors only for islands
                w.neighbors[i] = w_knn.neighbors[i]
                w.weights[i] = w_knn.weights[i]
        w.transform = "R"
        w = w.full()[0]
    elif method == "k_nearest":
        w = KNN.from_array(coords, k=k)
        if len(w.islands) > 0:
            if verbose:
                print(f"There are some islands in the data {method}({len(w.islands)}).")
        w.transform = "R"
        w = w.full()[0]
    w = pd.DataFrame(w, index=shape_data["LAD24CD"], columns=shape_data["LAD24CD"])
    d = data[["code", "year"] + x + [y]].dropna()  # NaNs filter
    if len(d) > 0:
        g_year = d.groupby("code").size().max()
        nonna_codes = []
        for code, group in d.groupby("code"):
            if (group["year"].nunique() < g_year) or (group.dropna().shape[0] < g_year):
                print(f"Code {code} has only {group['year'].nunique()} unique years.")
                pass
            else:
                nonna_codes.append(code)
        n_periods = f"{d['year'].nunique()} ({d['year'].min()}-{d['year'].max()})"
        codes = set(nonna_codes)
        w = w[[i for i in w.columns if i in codes]]
        n_units = w.columns.size
        w = w[[i in codes for i in w.index]]
        d = d[[i in codes for i in d.code]]
        w = full2W(np.asarray(w))
        w.transform = "R"
        if verbose:
            print(
                f"Number of units: {n_units}, Number of periods: {n_periods}, Method: {method}"
            )
        Y = d[y].values.reshape(-1, 1)
        X = d[x].values
        return w, X, Y
    else:
        if verbose:
            print("No valid data after dropping NaNs.")
        return None, None, None


def compute_weights_new(
    y, x, data, shape_data, method="inverse_distance", k=8, verbose=True
):
    coords = shape_data[["LAT", "LONG"]].to_numpy()
    if method == "inverse_distance":
        dist_mat = distance_matrix(coords, coords)
        np.fill_diagonal(dist_mat, np.inf)  # avoid division by zero
        w = 1 / dist_mat  # Inverse distance weights
    elif method == "queen":
        w = Queen.from_dataframe(shape_data, use_index=False)
        if len(w.islands) > 0:
            if verbose:
                print(f"There are some islands in the data {method}({len(w.islands)}).")
            w_knn = KNN.from_array(coords, k=1)
            for i in w.islands:  # use knn neighbors only for islands
                w.neighbors[i] = w_knn.neighbors[i]
                w.weights[i] = w_knn.weights[i]
        w.transform = "R"
        w = w.full()[0]
    elif method == "k_nearest":
        w = KNN.from_array(coords, k=k)
        if len(w.islands) > 0:
            if verbose:
                print(f"There are some islands in the data {method}({len(w.islands)}).")
        w.transform = "R"
        w = w.full()[0]
    w = pd.DataFrame(w, index=shape_data["LAD24CD"], columns=shape_data["LAD24CD"])
    d = data[["code", "year"] + x + [y]].dropna()  # NaNs filter
    if len(d) > 0:
        g_year = d.groupby("code").size().max()
        nonna_codes = []
        for code, group in d.groupby("code"):
            if (group["year"].nunique() < g_year) or (group.dropna().shape[0] < g_year):
                print(f"Code {code} has only {group['year'].nunique()} unique years.")
                pass
            else:
                nonna_codes.append(code)
        n_periods = f"{d['year'].nunique()} ({d['year'].min()}-{d['year'].max()})"
        codes = set(nonna_codes)
        w = w[[i for i in w.columns if i in codes]]
        n_units = w.columns.size
        w = w[[i in codes for i in w.index]]
        d = d[[i in codes for i in d.code]]
        w = full2W(np.asarray(w))
        w.transform = "R"
        if verbose:
            print(
                f"Number of units: {n_units}, Number of periods: {n_periods}, Method: {method}"
            )
        Y = d  # [y]  # .values.reshape(-1, 1)
        X = d[x]  # .values
        return w, X, Y
    else:
        if verbose:
            print("No valid data after dropping NaNs.")
        return None, None, None
