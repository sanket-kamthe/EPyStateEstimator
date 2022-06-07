# %%
import numpy as np
import sqlite3

# Experiment 1: Uniform Nonlinear Growth Model
con = sqlite3.connect("ungm_final_2.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
power_list = [1.0, 0.8, 0.6, 0.4, 0.2] # Results saved for 0.1, 0.6 or 0.8
damping = 0.2 # Results saved for 0.8 or 1.0
trans = 'MCT'

print("-"*65)
print(f"Trans type: {trans:6}" + "RMSE" + " "*20 + "NLL")
print("-"*65)
rmse_baseline = db.execute(f"SELECT RMSE FROM UNGM_EXP WHERE Power=1.0 AND Damping=1.0 AND Transform='{trans}' AND Iter=1").fetchall()
nll_baseline = db.execute(f"SELECT NLL FROM UNGM_EXP WHERE Power=1.0 AND Damping=1.0 AND Transform='{trans}' AND Iter=1").fetchall()
rmse_baseline, nll_baseline = np.array(rmse_baseline), np.array(nll_baseline)
print(f"Kalman Smoother {rmse_baseline.mean():10.2e} +/- {rmse_baseline.std():2.2e} {nll_baseline.mean():10.2e} +/- {nll_baseline.std():2.2e}")
for power in power_list:
    rmse_ = db.execute(f"SELECT RMSE FROM UNGM_EXP WHERE Power={power} AND Damping={damping} AND Transform='{trans}' AND Iter=50").fetchall()
    nll_ = db.execute(f"SELECT NLL FROM UNGM_EXP WHERE Power={power} AND Damping={damping} AND Transform='{trans}' AND Iter=50").fetchall()
    rmse_, nll_ = np.array(rmse_), np.array(nll_)
    print(f"Power EP ({power:1.1f})  {np.array(rmse_).mean():10.2e} +/- {rmse_.std():2.2e} {np.array(nll_).mean():10.2e} +/- {nll_.std():2.2e}")
print("-"*65)

# %%

# Experiment 2: Bearings Only Tracking Turn
con = sqlite3.connect("../log/bott_final_2.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
power_list = [1.0, 0.8, 0.6, 0.4, 0.2] # Results saved for 0.1, 0.6 or 0.8
damping = 1.0 # Results saved for 0.8 or 1.0
trans = 'MCT'
quant = 'angle'
quant_dict = {'position': 'Position (m)', 'velocity': 'Velocity (m/s)', 'angle': 'Angular velocity (rad/s)'}

print(f"Physical quantity: {quant_dict[quant]}")
print("-"*65)
print(f"Trans type: {trans:6}" + "RMSE" + " "*20 + "NLL")
print("-"*65)
rmse_baseline = db.execute(f"SELECT RMSE FROM BOTT_EXP WHERE Power=1.0 AND Damping=1.0 AND Transform='{trans}' AND Quantity='{quant}' AND Iter=1").fetchall()
nll_baseline = db.execute(f"SELECT NLL FROM BOTT_EXP WHERE Power=1.0 AND Damping=1.0 AND Transform='{trans}' AND Quantity='{quant}' AND Iter=1").fetchall()
rmse_baseline, nll_baseline = np.array(rmse_baseline), np.array(nll_baseline)
print(f"Kalman Smoother {rmse_baseline.mean():10.2e} +/- {rmse_baseline.std():2.2e} {nll_baseline.mean():10.2e} +/- {nll_baseline.std():2.2e}")
for power in power_list:
    rmse_ = db.execute(f"SELECT RMSE FROM BOTT_EXP WHERE Power={power} AND Damping={damping} AND Transform='{trans}' AND Quantity='{quant}' AND Iter=50").fetchall()
    nll_ = db.execute(f"SELECT NLL FROM BOTT_EXP WHERE Power={power} AND Damping={damping} AND Transform='{trans}' AND Quantity='{quant}' AND Iter=50").fetchall()
    rmse_, nll_ = np.array(rmse_), np.array(nll_)
    print(f"Power EP ({power:1.1f})  {np.array(rmse_).mean():10.2e} +/- {rmse_.std():2.2e} {np.array(nll_).mean():10.2e} +/- {nll_.std():2.2e}")
print("-"*65)

# %%

def remove_outliers(x, outlier_idxs=None, dont_remove=False):
    if dont_remove:
        return None, x
    else:
        if outlier_idxs is None:
            first_quartile = np.percentile(x, q=25)
            third_quartile = np.percentile(x, q=75)
            IQR = third_quartile - first_quartile
            upper_lim = third_quartile + 1.5*IQR
            lower_lim = first_quartile - 1.5*IQR
            outlier_idxs = np.concatenate([np.where(x > upper_lim)[0], np.where(x < lower_lim)[0]])
        
        idxs = []
        for idx in range(x.shape[0]):
            if idx not in outlier_idxs:
                idxs.append(idx)
        idxs = np.array(idxs)

        return outlier_idxs, x[idxs]


# Sensitivity analysis with dimension (nonlinear observation)
con = sqlite3.connect("../log/L96_dim_experiment_3.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
power = 1.0 # Results saved for 0.1, 0.8 or 1.0
trans = 'UT'

for dim in [5, 10, 20, 40, 100, 200]:
    print(f"Dimension: {dim:3}" + " "*8 + "RMSE" + " "*15 + "NLL")
    print("-"*60)
    rmse_baseline = db.execute(f"SELECT RMSE FROM L96_EXP WHERE Power=1.0 AND Damping=1.0 AND Transform='{trans}' AND dim={dim} AND Iter=0").fetchall()
    nll_baseline = db.execute(f"SELECT NLL FROM L96_EXP WHERE Power=1.0 AND Damping=1.0 AND Transform='{trans}' AND dim={dim} AND Iter=0").fetchall()
    rmse_ = db.execute(f"SELECT RMSE FROM L96_EXP WHERE Power={power} AND Damping=1.0 AND Transform='{trans}' AND dim={dim} AND Iter=9").fetchall()
    nll_ = db.execute(f"SELECT NLL FROM L96_EXP WHERE Power={power} AND Damping=1.0 AND Transform='{trans}' AND dim={dim} AND Iter=9").fetchall()
    rmse_baseline, nll_baseline = np.array(rmse_baseline).squeeze(), np.array(nll_baseline).squeeze()
    rmse_, nll_ = np.array(rmse_).squeeze(), np.array(nll_).squeeze()
    # Remove outliers
    out_idxs, nll_baseline_outliers_removed = remove_outliers(nll_baseline, dont_remove=True)
    _, nll_outliers_removed = remove_outliers(nll_, out_idxs, dont_remove=True)
    _, rmse_baseline_outliers_removed = remove_outliers(rmse_baseline, out_idxs, dont_remove=True)
    _, rmse_outliers_removed = remove_outliers(rmse_, out_idxs, dont_remove=True)
    # Print results
    print(f"Kalman Smoother {rmse_baseline_outliers_removed.mean():10.2f} +/- {rmse_baseline_outliers_removed.std():3.2f} {nll_baseline_outliers_removed.mean():10.2f} +/- {nll_baseline_outliers_removed.std():3.2f}")
    print(f"Power EP ({power:1.1f})  {rmse_outliers_removed.mean():10.2f} +/- {rmse_outliers_removed.std():3.2f} {nll_outliers_removed.mean():10.2f} +/- {nll_outliers_removed.std():3.2f}")
    print("-"*60)

# %%
# Results for the Iterated Kalman smoother (Bell 1994)

# Experiment 1: Uniform Nonlinear Growth Model
con = sqlite3.connect("../log/IEKS_ungm.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
SEEDS = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]
rmse_list, nll_list = [], []
for seed in SEEDS:
    rmse_IEKS = db.execute(f"SELECT RMSE FROM UNGM_EXP WHERE Seed={seed} AND Iter=50").fetchall()
    nll_IEKS = db.execute(f"SELECT NLL FROM UNGM_EXP WHERE Seed={seed} AND Iter=50").fetchall()
    rmse_list.append(rmse_IEKS)
    nll_list.append(nll_IEKS)

print(f"RMSE: {np.array(rmse_list).mean()} +/- {np.array(rmse_list).std()}")
print(f"NLL: {np.array(nll_list).mean()} +/- {np.array(nll_list).std()}")

# %%
# Experiment 2: Bearings Only Tracking Turn
con = sqlite3.connect("../log/IEKS_bott.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
quantity = 'angle'
SEEDS = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]
rmse_list, nll_list = [], []
for seed in SEEDS:
    rmse_IEKS = db.execute(f"SELECT RMSE FROM BOTT_EXP WHERE Quantity='{quantity}' AND Seed={seed} AND Iter=50").fetchall()
    nll_IEKS = db.execute(f"SELECT NLL FROM BOTT_EXP WHERE Quantity='{quantity}' AND Seed={seed} AND Iter=50").fetchall()
    rmse_list.append(rmse_IEKS)
    nll_list.append(nll_IEKS)

print(f"RMSE: {np.array(rmse_list).mean()} +/- {np.array(rmse_list).std()}")
print(f"NLL: {np.array(nll_list).mean()} +/- {np.array(nll_list).std()}")

# %%
con = sqlite3.connect("../log/IEKS_L96.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
dim = 20
SEEDS = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]
rmse_list, nll_list = [], []
for seed in SEEDS:
    rmse_IEKS = db.execute(f"SELECT RMSE FROM L96_EXP WHERE dim={dim} AND Seed={seed} AND Iter=10").fetchall()
    nll_IEKS = db.execute(f"SELECT NLL FROM L96_EXP WHERE dim={dim} AND Seed={seed} AND Iter=10").fetchall()
    rmse_list.append(rmse_IEKS)
    nll_list.append(nll_IEKS)

print(f"RMSE: {np.array(rmse_list).mean()} +/- {np.array(rmse_list).std()}")
print(f"NLL: {np.array(nll_list).mean()} +/- {np.array(nll_list).std()}")
# %%
