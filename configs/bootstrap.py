import os
import sys
import random
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from sklearn import set_config

from pathlib import Path

logger = logging.getLogger("ML_PROJECT")

# 2. Warnings & Debugging
warnings.filterwarnings("default")
# Uncomment during development to fail fast
# warnings.filterwarnings("error")

sys.tracebacklimit = 10
np.seterr(all="raise")

# 3. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# 4. Pandas configuration
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)
pd.options.mode.chained_assignment = "raise"

# 5. Plot styling 
sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="deep"
)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "grid.alpha": 0.3
})

# 6. sklearn global behavior
set_config(
    transform_output="pandas",
    display="diagram"
)

# 7. Bootstrap confirmation
logger.info("✅ ML bootstrap configuration loaded successfully")
