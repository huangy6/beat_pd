import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
from os.path import join
import os
import json
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[3]:


DATA_DIR = "data"
TSF_WINDOW_DIR = join(DATA_DIR, "cis-pd", "training_data_tsf")
TSF_WINDOW_OUT_DIR = join(DATA_DIR, "cis-pd", "training_data_tsf_stats")
TSF_WINDOW_FILES = [ join(TSF_WINDOW_DIR, f) for f in os.listdir(TSF_WINDOW_DIR) if f.endswith(".tsf.csv") ]
LABELS_FILE = join(DATA_DIR, "cis-pd", "data_labels", "CIS-PD_Training_Data_IDs_Labels.csv")


# In[4]:


labels_df = pd.read_csv(LABELS_FILE, index_col=0)
labels_df.head()

m_ids = [ os.path.basename(f[:-8]) for f in TSF_WINDOW_FILES ]
labels_df = labels_df.loc[m_ids,:]
labels_df = labels_df.sort_values(by="on_off", ascending=False)
labels_df.shape


# In[7]:


with open(join(DATA_DIR, "tsf_window_variables.json")) as f:
    top_vars = json.load(f)


# In[143]:


X_list = []
for m_id in labels_df.index.values.tolist():
    f = join(TSF_WINDOW_DIR, f"{m_id}.tsf.csv")
    m_df = pd.read_csv(f, index_col=0)
    
    m_X_df = pd.DataFrame(data=[], index=[], columns=["stat", "variable", "value", "dim"])
    for dim, dim_df in m_df.groupby("id"):
        dim_df = dim_df.set_index("window_start", drop=True)
        # TODO: maybe don't restrict to the top_vars columns?
        dim_df = dim_df[top_vars]
        dim_summary_df = dim_df.describe().reset_index()
        dim_summary_df = dim_summary_df.melt(id_vars=["index"]).rename(columns={"index": "stat"})
        dim_summary_df = dim_summary_df.loc[dim_summary_df["stat"].isin(["mean", "std"])]
        dim_summary_df["dim"] = dim
        m_X_df = m_X_df.append(dim_summary_df, ignore_index=True)
    X_list.append(m_X_df["value"].values)

    m_X_df.to_csv(join(TSF_WINDOW_OUT_DIR, f"{m_id}.stats.csv"))

len(X_list)


# In[68]:


X = np.stack(X_list, axis=-1).T
X.shape


# In[69]:


y = []
for m_id in labels_df.index.values.tolist():
    y.append(labels_df.at[m_id, "on_off"])
y = np.array(y)
X = X[~np.isnan(y)]
y = y[~np.isnan(y)]


# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[135]:


clf = RandomForestClassifier(n_estimators=1000, bootstrap=False, random_state=0)
clf.fit(X_train, y_train)


# In[139]:


y_pred = clf.predict(X_test)


# In[142]:


mse = mean_squared_error(y_test, y_pred)

out_obj = {
    "y_pred": list(y_pred),
    "mse": float(mse)
}


with open(join(DATA_DIR, "cis_pd_tsf_window_top_features_rf_pred.json"), "w") as f:
    json.dump(out_obj, f)
