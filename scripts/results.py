import os
import copy
import pickle
import numpy as np
import pandas as pd
from numbers import Number
from spreg.sputils import spmultiplier, _sp_effects
import spreg.diagnostics as diagnostics


base_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
results_dir = os.path.join(base_dir, "results")

print("data_dir:", data_dir)
print("results_dir:", results_dir)


def get_stars(p, p001="***", p01="**", p05="*", p10="+", p_="", p_none=""):
    if not isinstance(p, Number):
        return p_none
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return p_


def get_sp_effects(r, w):
    def rem_wx(r):
        no_var = [
            "sigma2_u",
            "lambda",
            "rho",
            "theta",
            "sigma2_epsilon",
            "sigma2_mu",
            "phi",
            "CONSTANT",
        ]
        r = copy.deepcopy(r)
        r.output["regime"] = 0
        r.output["var_type"] = r.output["var_names"].apply(
            lambda x: "wx" if x.startswith("W") else "x" if x not in no_var else "o"
        )
        xv = r.output[r.output["var_type"] == "x"]["var_names"].to_list()
        v = r.output.apply(lambda x: x["var_names"] in xv, axis=1)
        if r.slx_lags == 1:
            v = v | r.output.apply(lambda x: x["var_names"][2:] == r.name_y, axis=1)
        # print(r.output, v)
        v = r.output[v].reset_index(drop=True)
        return r, v

    r_, v_ = rem_wx(r)
    sp_ef = _sp_effects(
        r_, v_, spmultiplier(w, r_.rho, method="simple"), r_.slx_lags, "All"
    )
    sp_ef = pd.DataFrame(np.asarray(sp_ef).tolist()).T
    sp_ef.columns = ["total", "direct", "indirect"]
    v = v_[v_["var_type"] == "x"]["var_names"].to_list()
    sp_ef.index = r_.name_x[: len(sp_ef)]
    sp_ef = sp_ef.loc[v]
    return sp_ef.map(lambda x: x[0])


def get_results(r, w=None):
    def get_vif(r):
        try:
            vif = diagnostics.vif(r)
            vif = pd.DataFrame(np.asarray(vif)[:, 0])
            vif.index = r.name_x[: len(vif)]
            vif.columns = ["vif"]
        except:
            return None
        return vif

    def combine_tests(r):
        if hasattr(r, "bsk"):
            if hasattr(r, "hausman_stat"):
                return pd.concat(
                    [
                        r.bsk,
                        pd.DataFrame.from_dict(
                            {
                                "Test": "Hausman Specification Test",
                                "Statistic": r.hausman_stat,
                                "df": r.hausman_df,
                                "p-value": r.hausman_p,
                            },
                            orient="index",
                        ).T,
                    ],
                    axis=0,
                ).reset_index(drop=True)
            return r.bsk
        return None

    df_const = 1 if "CONSTANT" in set(r.output["var_names"]) else 0
    df_resid = r.df_resid if hasattr(r, "df_resid") else r.n - r.k
    df_model = r.n - df_resid - df_const
    return {
        "r2": r.pr2 if hasattr(r, "pr2") else r.r2,
        "aic": r.aic,
        "schwarz": r.schwarz,
        "llr": r.logll,
        #'f_stat': r.f_stat[0] if hasattr(r, "f_stat") else None,
        #'f_stat_p': r.f_stat[1] if hasattr(r, "f_stat") else None,
        #'f_df_resid': df_resid,
        #'f_df_model': df_model,
        "n_obs": r.n,
        #'n_params': r.k,
        #'n_const': df_const,
        #'rho': r.rho if hasattr(r, "rho") else None,
        #'sigma2': np.asarray(r.sig2).reshape(1)[0] if hasattr(r, "sig2") else None,
        #'sigma2_epsilon': r.sigma2_epsilon if hasattr(r, "sigma2_epsilon") else None,
        #'sigma2_mu': r.sigma2_mu if hasattr(r, "sigma2_mu") else None,
        #'theta': r.theta if hasattr(r, "theta") else None,
        #'vif': get_vif(r),
        "tests": combine_tests(r),  # tests
        "outputs": r.output,  # all params
        "effects": get_sp_effects(r, w)
        if w is not None and hasattr(r, "rho")
        else None,
    }


model_results = {}
cache_weights = {}
results_file = os.path.join(results_dir, "model_results.pickle")
if os.path.exists(results_file):
    with open(results_file, "rb") as f:
        model_results = pickle.load(f)

weights_file = os.path.join(results_dir, "cache_weights.pickle")
if os.path.exists(weights_file):
    with open(weights_file, "rb") as f:
        cache_weights = pickle.load(f)


def get_full_table(
    y="log_pv_cap",
    x_vars="baseline",
    w_methods=["inverse_distance", "k_nearest", "queen"],
    n_models=[
        "PooledOLS",
        # "PanelFE",
        "PanelRE",
        "ML_LagFE",
        # "ML_LagRE",
        "ML_ErrorFE",
        # "ML_ErrorRE",
        "ML_LagFE(SLX)",
        "ML_LagRE(SLX)",
    ],
    x_list=[],  # "pdensity", "hdensity", "fdensity", "adensity"
):
    _results = []
    _effects = []
    _tests = []
    i = model_results[y]
    for xvars in i.keys():
        if xvars == x_vars:
            for nmodels in sorted(
                list(i[xvars].keys()),
                key=lambda x: n_models.index(x) if x in n_models else 0,
            ):  # i[xvars].keys():
                if nmodels in n_models:
                    for wmethods in i[xvars][nmodels].keys():
                        if wmethods in w_methods:
                            for k in i[xvars][nmodels][wmethods].keys():
                                key = f"{xvars}-{nmodels}-{wmethods}-{k}"
                                x_match = [x for x in x_list if x in key]
                                if (len(x_list) > 0 and len(x_match) > 0) or (
                                    len(x_list) == 0
                                ):
                                    _r = i[xvars][nmodels][wmethods][k]
                                    if _r:
                                        # print(_r.summary)
                                        __r = get_results(
                                            copy.deepcopy(_r),
                                            cache_weights[k][0]
                                            if len(cache_weights) > 0
                                            else None,
                                        )
                                        if __r["tests"] is not None:
                                            _t = copy.deepcopy(__r["tests"])
                                            if _t is not None and len(_t) > 0:
                                                _t.columns = [
                                                    i + f"_{nmodels}_{wmethods}"
                                                    for i in _t.columns
                                                ]
                                                _t.index = _t[_t.columns[0]]
                                                _t[wmethods] = _t.apply(
                                                    lambda x: (
                                                        f"{x[_t.columns[1]]:.2f}{get_stars(x[_t.columns[3]])}"
                                                    ),
                                                    axis=1,
                                                )
                                                _t = _t[[wmethods]]
                                                _tests.append(_t.T)
                                        _o = __r["outputs"].copy()
                                        _o.index = _o["var_names"]
                                        _o[0] = _o.apply(
                                            lambda x: (
                                                f"{x['coefficients']:.3f}{get_stars(x['prob'])}"
                                            ),
                                            axis=1,
                                        )
                                        __r = pd.DataFrame.from_dict([__r]).T
                                        __r = pd.concat(
                                            [
                                                _o[[0]],
                                                __r,
                                            ],
                                            ignore_index=False,
                                        )
                                        __r.columns = [key]
                                        for c in __r.index:
                                            if c in [
                                                "r2",
                                                "aic",
                                                "schwarz",
                                                "llr",
                                            ]:
                                                __r.loc[c] = __r.loc[c].apply(
                                                    lambda x: f"{x:.3f}"
                                                )
                                        _results.append(__r)
                                        eff = copy.deepcopy(__r.loc["effects"])
                                        if len(eff) > 0:
                                            eff = eff.iloc[0]
                                            if eff is not None and len(eff) > 0:
                                                eff.columns = [
                                                    i + f"_{nmodels}_{wmethods}"
                                                    for i in eff.columns
                                                ]
                                                eff = eff.map(
                                                    lambda x: (
                                                        f"{x:.3f}"
                                                        if isinstance(x, Number)
                                                        else x
                                                    )
                                                )
                                                _effects.append(eff)

    _results = pd.concat(_results, axis=1)
    _effects = pd.concat(_effects, axis=1) if len(_effects) > 0 else None
    _tests = (
        pd.concat(
            [
                pd.concat(
                    [t for t in _tests if "Hausman" not in str(t.T.index)], axis=0
                ),
                pd.concat([t for t in _tests if "Hausman" in str(t.T.index)], axis=0),
            ],
            axis=1,
        )
        if len(_tests) > 0
        else None
    )

    res_y = os.path.join(results_dir, y)
    prefix = f"{y}_{x_vars}" if len(x_list) == 0 else f"{y}_{x_vars}_{'_'.join(x_list)}"
    os.makedirs(res_y, exist_ok=True)
    if _results is not None and len(_results) > 0:
        _results.astype(str).to_excel(
            os.path.join(res_y, f"{prefix}_estimates.xlsx"), engine="openpyxl"
        )

    if _effects is not None and len(_effects) > 0:
        _effects.astype(str).to_excel(
            os.path.join(res_y, f"{prefix}_effects.xlsx"), engine="openpyxl"
        )

    if _tests is not None and len(_tests) > 0:
        _tests.astype(str).to_excel(
            os.path.join(res_y, f"{prefix}_tests.xlsx"), engine="openpyxl"
        )

    return _results, _effects, _tests


y_vars = ["log_pv_cap", "log_pv_inst", "log_pv_cap_per_inst", "pv_cap_per_inst"]
for y in y_vars:
    r, e, t = get_full_table(
        y=y,
        x_vars="baseline",
        w_methods=["inverse_distance", "k_nearest", "queen"],
        n_models=[
            "ML_LagFE(SLX)",
            "ML_LagFE",
            "ML_ErrorFE",
            "PooledOLS",
            "PanelRE",
            # "PanelFE",
            # "ML_LagRE",
            # "ML_ErrorFE",
            # "ML_ErrorRE",
            # "ML_LagRE(SLX)",
        ],
        x_list=[],
    )

    r, e, t = get_full_table(
        y=y,
        x_vars="capacities",
        w_methods=["inverse_distance", "k_nearest", "queen"],
        n_models=[
            "ML_LagFE(SLX)",
            "ML_LagFE",
            "ML_ErrorFE",
            "PooledOLS",
            "PanelRE",
            # "PanelFE",
            # "ML_LagRE",
            # "ML_ErrorFE",
            # "ML_ErrorRE",
            # "ML_LagRE(SLX)",
        ],
        x_list=[],
    )

    r, e, t = get_full_table(
        y=y,
        x_vars="densities",
        w_methods=["inverse_distance", "k_nearest", "queen"],
        n_models=[
            "ML_LagFE(SLX)",
            "ML_LagFE",
            "ML_ErrorFE",
            "PooledOLS",
            "PanelRE",
            # "PanelFE",
            # "ML_LagRE",
            # "ML_ErrorFE",
            # "ML_ErrorRE",
            # "ML_LagRE(SLX)",
        ],
        x_list=[
            "pdensity",
            # "hdensity",
            # "fdensity",
            # "adensity",
        ],
    )

    r, e, t = get_full_table(
        y=y,
        x_vars="densities",
        w_methods=["inverse_distance", "k_nearest", "queen"],
        n_models=[
            "ML_LagFE(SLX)",
            "ML_LagFE",
            "ML_ErrorFE",
            "PooledOLS",
            "PanelRE",
            # "PanelFE",
            # "ML_LagRE",
            # "ML_ErrorFE",
            # "ML_ErrorRE",
            # "ML_LagRE(SLX)",
        ],
        x_list=[  # "pdensity",
            "hdensity",
            # "fdensity",
            # "adensity",
        ],
    )

    r, e, t = get_full_table(
        y=y,
        x_vars="densities",
        w_methods=["inverse_distance", "k_nearest", "queen"],
        n_models=[
            "ML_LagFE(SLX)",
            "ML_LagFE",
            "ML_ErrorFE",
            "PooledOLS",
            "PanelRE",
            # "PanelFE",
            # "ML_LagRE",
            # "ML_ErrorFE",
            # "ML_ErrorRE",
            # "ML_LagRE(SLX)",
        ],
        x_list=[  # "pdensity",
            # "hdensity",
            "fdensity",
            # "adensity",
        ],
    )

    r, e, t = get_full_table(
        y=y,
        x_vars="densities",
        w_methods=["inverse_distance", "k_nearest", "queen"],
        n_models=[
            "ML_LagFE(SLX)",
            "ML_LagFE",
            "ML_ErrorFE",
            "PooledOLS",
            "PanelRE",
            # "PanelFE",
            # "ML_LagRE",
            # "ML_ErrorFE",
            # "ML_ErrorRE",
            # "ML_LagRE(SLX)",
        ],
        x_list=[  # "pdensity",
            # "hdensity",
            # "fdensity",
            "adensity",
        ],
    )
