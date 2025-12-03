{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
   },
   "source": [
    "# Predictive Maintenance of Turbofan Engines\n",
    "## RUL (Remaining Useful Life) Regression Model\n",
    "\n",
    "**Goal:** build a regression model that predicts the Remaining Useful Life (RUL) of aircraft engines using multivariate time series data.\n",
    "\n",
    "We use three files:\n",
    "\n",
    "- `PM_train_data.csv` : full run-to-failure histories for several engines.\n",
    "- `PM_test_data.csv`  : partial histories (engine still operational).\n",
    "- `PM_truth.csv`      : true RUL values for each engine in the test set (for the *last* cycle of each engine).\n",
    "\n",
    "The project follows the ESILV ML guidelines:\n",
    "- Data exploration and preprocessing\n",
    "- Creation of RUL target variable\n",
    "- Baseline and advanced models\n",
    "- Hyperparameter tuning (GridSearchCV)\n",
    "- Evaluation and model comparison\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:12:48.314171Z",
     "iopub.status.busy": "2025-12-02T16:12:48.313977Z",
     "iopub.status.idle": "2025-12-02T16:12:52.648010Z",
     "shell.execute_reply": "2025-12-02T16:12:52.647054Z",
     "shell.execute_reply.started": "2025-12-02T16:12:48.314152Z"
    }
   },
   "outputs": [],
   "source": [
    "# ==============================\n",
    "# 1. Imports and data loading\n",
    "# ==============================\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load datasets and basic inspection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:14:38.821612Z",
     "iopub.status.busy": "2025-12-02T16:14:38.821325Z",
     "iopub.status.idle": "2025-12-02T16:14:38.964802Z",
     "shell.execute_reply": "2025-12-02T16:14:38.963646Z",
     "shell.execute_reply.started": "2025-12-02T16:14:38.821591Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train shape : (20631, 26)\n",
      "Test shape  : (13096, 26)\n",
      "Truth shape : (100, 2)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>cycle</th>\n",
       "      <th>setting1</th>\n",
       "      <th>setting2</th>\n",
       "      <th>setting3</th>\n",
       "      <th>s1</th>\n",
       "      <th>s2</th>\n",
       "      <th>s3</th>\n",
       "      <th>s4</th>\n",
       "      <th>s5</th>\n",
       "      <th>...</th>\n",
       "      <th>s12</th>\n",
       "      <th>s13</th>\n",
       "      <th>s14</th>\n",
       "      <th>s15</th>\n",
       "      <th>s16</th>\n",
       "      <th>s17</th>\n",
       "      <th>s18</th>\n",
       "      <th>s19</th>\n",
       "      <th>s20</th>\n",
       "      <th>s21</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>-0.0007</td>\n",
       "      <td>-0.0004</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>641.82</td>\n",
       "      <td>1589.70</td>\n",
       "      <td>1400.60</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>521.66</td>\n",
       "      <td>2388.02</td>\n",
       "      <td>8138.62</td>\n",
       "      <td>8.4195</td>\n",
       "      <td>0.03</td>\n",
       "      <td>392</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>39.06</td>\n",
       "      <td>23.4190</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0019</td>\n",
       "      <td>-0.0003</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.15</td>\n",
       "      <td>1591.82</td>\n",
       "      <td>1403.14</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>522.28</td>\n",
       "      <td>2388.07</td>\n",
       "      <td>8131.49</td>\n",
       "      <td>8.4318</td>\n",
       "      <td>0.03</td>\n",
       "      <td>392</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>39.00</td>\n",
       "      <td>23.4236</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>-0.0043</td>\n",
       "      <td>0.0003</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.35</td>\n",
       "      <td>1587.99</td>\n",
       "      <td>1404.20</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>522.42</td>\n",
       "      <td>2388.03</td>\n",
       "      <td>8133.23</td>\n",
       "      <td>8.4178</td>\n",
       "      <td>0.03</td>\n",
       "      <td>390</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>38.95</td>\n",
       "      <td>23.3442</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>0.0007</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.35</td>\n",
       "      <td>1582.79</td>\n",
       "      <td>1401.87</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>522.86</td>\n",
       "      <td>2388.08</td>\n",
       "      <td>8133.83</td>\n",
       "      <td>8.3682</td>\n",
       "      <td>0.03</td>\n",
       "      <td>392</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>38.88</td>\n",
       "      <td>23.3739</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>-0.0019</td>\n",
       "      <td>-0.0002</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.37</td>\n",
       "      <td>1582.85</td>\n",
       "      <td>1406.22</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>522.19</td>\n",
       "      <td>2388.04</td>\n",
       "      <td>8133.80</td>\n",
       "      <td>8.4294</td>\n",
       "      <td>0.03</td>\n",
       "      <td>393</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>38.90</td>\n",
       "      <td>23.4044</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  cycle  setting1  setting2  setting3      s1      s2       s3       s4  \\\n",
       "0   1      1   -0.0007   -0.0004     100.0  518.67  641.82  1589.70  1400.60   \n",
       "1   1      2    0.0019   -0.0003     100.0  518.67  642.15  1591.82  1403.14   \n",
       "2   1      3   -0.0043    0.0003     100.0  518.67  642.35  1587.99  1404.20   \n",
       "3   1      4    0.0007    0.0000     100.0  518.67  642.35  1582.79  1401.87   \n",
       "4   1      5   -0.0019   -0.0002     100.0  518.67  642.37  1582.85  1406.22   \n",
       "\n",
       "      s5  ...     s12      s13      s14     s15   s16  s17   s18    s19  \\\n",
       "0  14.62  ...  521.66  2388.02  8138.62  8.4195  0.03  392  2388  100.0   \n",
       "1  14.62  ...  522.28  2388.07  8131.49  8.4318  0.03  392  2388  100.0   \n",
       "2  14.62  ...  522.42  2388.03  8133.23  8.4178  0.03  390  2388  100.0   \n",
       "3  14.62  ...  522.86  2388.08  8133.83  8.3682  0.03  392  2388  100.0   \n",
       "4  14.62  ...  522.19  2388.04  8133.80  8.4294  0.03  393  2388  100.0   \n",
       "\n",
       "     s20      s21  \n",
       "0  39.06  23.4190  \n",
       "1  39.00  23.4236  \n",
       "2  38.95  23.3442  \n",
       "3  38.88  23.3739  \n",
       "4  38.90  23.4044  \n",
       "\n",
       "[5 rows x 26 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Adjust paths if needed (e.g. Kaggle input directory)\n",
    "train_path = \"/kaggle/input/predictive-maintenance-aircraft-engine/PM_train.csv\"\n",
    "test_path = \"/kaggle/input/predictive-maintenance-aircraft-engine/PM_test.csv\"\n",
    "truth_path = \"/kaggle/input/predictive-maintenance-aircraft-engine/PM_truth.csv\"\n",
    "\n",
    "train_df = pd.read_csv(train_path)\n",
    "test_df = pd.read_csv(test_path)\n",
    "truth_df = pd.read_csv(truth_path)\n",
    "\n",
    "print(\"Train shape :\", train_df.shape)\n",
    "print(\"Test shape  :\", test_df.shape)\n",
    "print(\"Truth shape :\", truth_df.shape)\n",
    "\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1 Columns and basic statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:15:00.381311Z",
     "iopub.status.busy": "2025-12-02T16:15:00.380962Z",
     "iopub.status.idle": "2025-12-02T16:15:00.453409Z",
     "shell.execute_reply": "2025-12-02T16:15:00.452514Z",
     "shell.execute_reply.started": "2025-12-02T16:15:00.381285Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Columns: ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>51.506568</td>\n",
       "      <td>2.922763e+01</td>\n",
       "      <td>1.0000</td>\n",
       "      <td>26.0000</td>\n",
       "      <td>52.00</td>\n",
       "      <td>77.0000</td>\n",
       "      <td>100.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>cycle</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>108.807862</td>\n",
       "      <td>6.888099e+01</td>\n",
       "      <td>1.0000</td>\n",
       "      <td>52.0000</td>\n",
       "      <td>104.00</td>\n",
       "      <td>156.0000</td>\n",
       "      <td>362.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>setting1</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>-0.000009</td>\n",
       "      <td>2.187313e-03</td>\n",
       "      <td>-0.0087</td>\n",
       "      <td>-0.0015</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0015</td>\n",
       "      <td>0.0087</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>setting2</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>0.000002</td>\n",
       "      <td>2.930621e-04</td>\n",
       "      <td>-0.0006</td>\n",
       "      <td>-0.0002</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0003</td>\n",
       "      <td>0.0006</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>setting3</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>100.0000</td>\n",
       "      <td>100.0000</td>\n",
       "      <td>100.00</td>\n",
       "      <td>100.0000</td>\n",
       "      <td>100.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s1</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>518.670000</td>\n",
       "      <td>6.537152e-11</td>\n",
       "      <td>518.6700</td>\n",
       "      <td>518.6700</td>\n",
       "      <td>518.67</td>\n",
       "      <td>518.6700</td>\n",
       "      <td>518.6700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s2</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>642.680934</td>\n",
       "      <td>5.000533e-01</td>\n",
       "      <td>641.2100</td>\n",
       "      <td>642.3250</td>\n",
       "      <td>642.64</td>\n",
       "      <td>643.0000</td>\n",
       "      <td>644.5300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s3</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>1590.523119</td>\n",
       "      <td>6.131150e+00</td>\n",
       "      <td>1571.0400</td>\n",
       "      <td>1586.2600</td>\n",
       "      <td>1590.10</td>\n",
       "      <td>1594.3800</td>\n",
       "      <td>1616.9100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s4</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>1408.933782</td>\n",
       "      <td>9.000605e+00</td>\n",
       "      <td>1382.2500</td>\n",
       "      <td>1402.3600</td>\n",
       "      <td>1408.04</td>\n",
       "      <td>1414.5550</td>\n",
       "      <td>1441.4900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s5</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>14.620000</td>\n",
       "      <td>3.394700e-12</td>\n",
       "      <td>14.6200</td>\n",
       "      <td>14.6200</td>\n",
       "      <td>14.62</td>\n",
       "      <td>14.6200</td>\n",
       "      <td>14.6200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s6</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>21.609803</td>\n",
       "      <td>1.388985e-03</td>\n",
       "      <td>21.6000</td>\n",
       "      <td>21.6100</td>\n",
       "      <td>21.61</td>\n",
       "      <td>21.6100</td>\n",
       "      <td>21.6100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s7</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>553.367711</td>\n",
       "      <td>8.850923e-01</td>\n",
       "      <td>549.8500</td>\n",
       "      <td>552.8100</td>\n",
       "      <td>553.44</td>\n",
       "      <td>554.0100</td>\n",
       "      <td>556.0600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s8</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>2388.096652</td>\n",
       "      <td>7.098548e-02</td>\n",
       "      <td>2387.9000</td>\n",
       "      <td>2388.0500</td>\n",
       "      <td>2388.09</td>\n",
       "      <td>2388.1400</td>\n",
       "      <td>2388.5600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s9</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>9065.242941</td>\n",
       "      <td>2.208288e+01</td>\n",
       "      <td>9021.7300</td>\n",
       "      <td>9053.1000</td>\n",
       "      <td>9060.66</td>\n",
       "      <td>9069.4200</td>\n",
       "      <td>9244.5900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>s10</th>\n",
       "      <td>20631.0</td>\n",
       "      <td>1.300000</td>\n",
       "      <td>4.660829e-13</td>\n",
       "      <td>1.3000</td>\n",
       "      <td>1.3000</td>\n",
       "      <td>1.30</td>\n",
       "      <td>1.3000</td>\n",
       "      <td>1.3000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            count         mean           std        min        25%      50%  \\\n",
       "id        20631.0    51.506568  2.922763e+01     1.0000    26.0000    52.00   \n",
       "cycle     20631.0   108.807862  6.888099e+01     1.0000    52.0000   104.00   \n",
       "setting1  20631.0    -0.000009  2.187313e-03    -0.0087    -0.0015     0.00   \n",
       "setting2  20631.0     0.000002  2.930621e-04    -0.0006    -0.0002     0.00   \n",
       "setting3  20631.0   100.000000  0.000000e+00   100.0000   100.0000   100.00   \n",
       "s1        20631.0   518.670000  6.537152e-11   518.6700   518.6700   518.67   \n",
       "s2        20631.0   642.680934  5.000533e-01   641.2100   642.3250   642.64   \n",
       "s3        20631.0  1590.523119  6.131150e+00  1571.0400  1586.2600  1590.10   \n",
       "s4        20631.0  1408.933782  9.000605e+00  1382.2500  1402.3600  1408.04   \n",
       "s5        20631.0    14.620000  3.394700e-12    14.6200    14.6200    14.62   \n",
       "s6        20631.0    21.609803  1.388985e-03    21.6000    21.6100    21.61   \n",
       "s7        20631.0   553.367711  8.850923e-01   549.8500   552.8100   553.44   \n",
       "s8        20631.0  2388.096652  7.098548e-02  2387.9000  2388.0500  2388.09   \n",
       "s9        20631.0  9065.242941  2.208288e+01  9021.7300  9053.1000  9060.66   \n",
       "s10       20631.0     1.300000  4.660829e-13     1.3000     1.3000     1.30   \n",
       "\n",
       "                75%        max  \n",
       "id          77.0000   100.0000  \n",
       "cycle      156.0000   362.0000  \n",
       "setting1     0.0015     0.0087  \n",
       "setting2     0.0003     0.0006  \n",
       "setting3   100.0000   100.0000  \n",
       "s1         518.6700   518.6700  \n",
       "s2         643.0000   644.5300  \n",
       "s3        1594.3800  1616.9100  \n",
       "s4        1414.5550  1441.4900  \n",
       "s5          14.6200    14.6200  \n",
       "s6          21.6100    21.6100  \n",
       "s7         554.0100   556.0600  \n",
       "s8        2388.1400  2388.5600  \n",
       "s9        9069.4200  9244.5900  \n",
       "s10          1.3000     1.3000  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(\"Columns:\", train_df.columns.tolist())\n",
    "\n",
    "# Basic statistics on numeric features\n",
    "train_df.describe().T.head(15)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2 Missing values and duplicates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:15:20.567445Z",
     "iopub.status.busy": "2025-12-02T16:15:20.567096Z",
     "iopub.status.idle": "2025-12-02T16:15:20.598323Z",
     "shell.execute_reply": "2025-12-02T16:15:20.597557Z",
     "shell.execute_reply.started": "2025-12-02T16:15:20.567424Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Missing values per column (train):\n",
      "id          0\n",
      "cycle       0\n",
      "setting1    0\n",
      "setting2    0\n",
      "setting3    0\n",
      "s1          0\n",
      "s2          0\n",
      "s3          0\n",
      "s4          0\n",
      "s5          0\n",
      "s6          0\n",
      "s7          0\n",
      "s8          0\n",
      "s9          0\n",
      "s10         0\n",
      "s11         0\n",
      "s12         0\n",
      "s13         0\n",
      "s14         0\n",
      "s15         0\n",
      "s16         0\n",
      "s17         0\n",
      "s18         0\n",
      "s19         0\n",
      "s20         0\n",
      "s21         0\n",
      "dtype: int64\n",
      "\n",
      "Missing values per column (test):\n",
      "id          0\n",
      "cycle       0\n",
      "setting1    0\n",
      "setting2    0\n",
      "setting3    0\n",
      "s1          0\n",
      "s2          0\n",
      "s3          0\n",
      "s4          0\n",
      "s5          0\n",
      "s6          0\n",
      "s7          0\n",
      "s8          0\n",
      "s9          0\n",
      "s10         0\n",
      "s11         0\n",
      "s12         0\n",
      "s13         0\n",
      "s14         0\n",
      "s15         0\n",
      "s16         0\n",
      "s17         0\n",
      "s18         0\n",
      "s19         0\n",
      "s20         0\n",
      "s21         0\n",
      "dtype: int64\n",
      "\n",
      "Number of duplicated rows in train: 0\n"
     ]
    }
   ],
   "source": [
    "print(\"Missing values per column (train):\")\n",
    "print(train_df.isna().sum())\n",
    "\n",
    "print(\"\\nMissing values per column (test):\")\n",
    "print(test_df.isna().sum())\n",
    "\n",
    "print(\"\\nNumber of duplicated rows in train:\", train_df.duplicated().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.3 Example of engine trajectories\n",
    "\n",
    "We visualise a few sensors over time for one engine to understand the temporal dynamics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:17:22.949125Z",
     "iopub.status.busy": "2025-12-02T16:17:22.948768Z",
     "iopub.status.idle": "2025-12-02T16:17:23.169009Z",
     "shell.execute_reply": "2025-12-02T16:17:23.168284Z",
     "shell.execute_reply.started": "2025-12-02T16:17:22.949098Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA90AAAGGCAYAAABmGOKbAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAB0AUlEQVR4nO3dd3wU1f7/8ffupickIUASghBCQOmINOkokYBYECwgVxEVLKBesSA/v6DYELwqol7Qa8GCXitYroJIEYWItIAiICAdQqQkIYS03fn9cbKbLAmQYJZQXs/HYwk7c3bmTD+fc87M2CzLsgQAAAAAACqdvaozAAAAAADA2YqgGwAAAAAAHyHoBgAAAADARwi6AQAAAADwEYJuAAAAAAB8hKAbAAAAAAAfIegGAAAAAMBHCLoBAAAAAPARgm4AAAAAAHyEoBsAgBO45ZZbVL9+/arOxknZunWrbDabpk+ffkrnu3HjRvXq1UsRERGy2WyaNWvWKZ3/qVS/fn3dcsstp3y+CxculM1m08KFC0+YtkePHurRo4fP8wQAKI2gGwDOMr/++quuvfZaxcfHKygoSHXq1NFll12ml19+uaqz5jO7d+/W448/rtTU1KrOykn54IMPNHny5KrORqUaMmSIfv31Vz399NN677331LZt26rOEk6xX375RXfffbfatGkjf39/2Wy2qs4SAFQJm2VZVlVnAgBQOZYsWaJLLrlE9erV05AhQxQbG6sdO3bo559/1ubNm7Vp06aqzqJPLF++XO3atdPbb7/tkxbHgoICuVwuBQYGVvq0JemKK67Qb7/9pq1bt1b6tC3LUl5envz9/eVwOCp9+mU5cuSIQkJC9Oijj+qpp546JfOsSnl5ebLb7fL39z+l83W5XMrPz1dAQIDs9uO3o7hbucvTKl5ZHn/8cT3zzDNq2bKlDh06pD/++EMUOwGci/yqOgMAgMrz9NNPKyIiQsuWLVNkZKTXuPT09KrJVCU5fPiwQkNDK2VaOTk5CgkJKXf6Ux1MVYbCwkK5XC4FBAQoKCjolM77r7/+kqRS++DfUZnbv7L5qjLmROx2+ynfthVx1113afTo0QoODtbIkSP1xx9/VHWWAKBK0L0cAM4imzdvVrNmzcoMdqKjo0sNe//999WmTRsFBwcrKipKAwcO1I4dO7zS9OjRQ82bN9fvv/+uSy65RCEhIapTp44mTZpUanovv/yymjVrppCQEFWvXl1t27bVBx984JVm1apV6tOnj8LDwxUWFqaePXvq559/9kozffp02Ww2/fDDD7r77rsVHR2t8847r8xlXrhwodq1aydJGjp0qGw2m9c9zO78r1ixQt26dVNISIj+3//7f5KkL774Qn379lVcXJwCAwOVmJioJ598Uk6n02seZd3T7XK5NHnyZDVr1kxBQUGKiYnRHXfcoYMHD5bK47fffqvu3burWrVqCg8PV7t27TzrpUePHvrf//6nbdu2efJecl7p6em67bbbFBMTo6CgILVq1UrvvPOO1/Td923/61//0uTJk5WYmKjAwED9/vvvx7yne/369br22msVFRWloKAgtW3bVl9++aVXmoKCAo0fP16NGjVSUFCQatSooS5dumju3LllbgvJtG7Gx8dLkh566KFSy1PZ298tLy9Pjz32mBo2bKjAwEDVrVtXDz/8sPLy8rzS2Ww2jRw5UrNmzVLz5s0VGBioZs2aafbs2aWmuXDhQrVt21ZBQUFKTEzUa6+9pscff7xUN+mj7+l253/x4sUaNWqUatWqpdDQUF1zzTWeComSvv32W3Xt2lWhoaGqVq2a+vbtq7Vr1x53ed35K+ue7tdff12JiYkKDg5W+/bt9eOPP55wWr4QExOj4ODgKpk3AJxOaOkGgLNIfHy8UlJS9Ntvv6l58+bHTfv0009r7Nixuv7663X77bfrr7/+0ssvv6xu3bpp1apVXoH7wYMH1bt3b/Xv31/XX3+9Pv30U40ePVotWrRQnz59JEn/+c9/dO+99+raa6/Vfffdp9zcXK1Zs0ZLly7VjTfeKElau3atunbtqvDwcD388MPy9/fXa6+9ph49euiHH35Qhw4dvPJ49913q1atWho3bpwOHz5c5nI0adJETzzxhMaNG6fhw4era9eukqROnTp50uzfv199+vTRwIED9Y9//EMxMTGSTHAUFhamUaNGKSwsTPPnz9e4ceOUlZWl55577rjr74477tD06dM1dOhQ3XvvvdqyZYteeeUVrVq1SosXL/a0jk+fPl233nqrmjVrpjFjxigyMlKrVq3S7NmzdeONN+rRRx9VZmamdu7cqRdffFGSFBYWJsl00+7Ro4c2bdqkkSNHKiEhQZ988oluueUWZWRk6L777vPK09tvv63c3FwNHz5cgYGB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777TG2+8oRdffFHTpk3T7bffXnkZP4bybn+Xy6UWLVrohRdeKHP80fcyl2d/rgwnmo97W7z33nuKjY0tlc7P79QX0w4cOKD8/PwTpgsODi536zkAnOsIugHgHNC2bVtJ0p49eyRJiYmJsixLCQkJOv/88yttPqGhobrhhht0ww03KD8/X/3799fTTz+tMWPGqFatWgoJCdGGDRtK/W79+vWy2+3letBTWcrTVftoCxcu1P79+/X555+rW7dunuFbtmw54W8TExP1/fffq3PnzscNDN1dmH/77bfjtpAeK//x8fFas2aNXC6XV2u3u+u7+ynhFeFu8fT391dSUtIJ00dFRWno0KEaOnSosrOz1a1bNz3++OMVDrp9uf0TExO1evVq9ezZ86T2haNFR0crKCiozPfaV+a77t37R3R0dLm2xYm494eNGzd6tZ4XFBRoy5YtatWq1Qmn0b9//1K3LZRlyJAhpZ6IDwAoG/d0A8BZZMGCBWW21n3zzTeS5Ona279/fzkcDo0fP75UesuytH///grP++jfBAQEqGnTprIsSwUFBXI4HOrVq5e++OILbd261ZNu7969+uCDD9SlS5cKd6d1c7+/OSMjo9y/cbdCllz+/Px8/fvf/z7hb6+//no5nU49+eSTpcYVFhZ68tGrVy9Vq1ZNEyZMUG5urle6kvMNDQ0tszv75ZdfrrS0NK+nXRcWFurll19WWFiYunfvfsK8Hi06Olo9evTQa6+95qmEKankK62O3qZhYWFq2LBhqddwlYcvt//111+vXbt26T//+U+pcUeOHDnuk8+PldekpCTNmjVLu3fv9gzftGmTvv3225PKY1mSk5MVHh6uZ555pswnhpf1erHjadu2rWrVqqVp06Z5tVZPnz693McG93QDQOWjpRsAziL33HOPcnJydM0116hx48bKz8/XkiVL9NFHH6l+/fqeh4klJibqqaee0pgxY7R161b169dP1apV05YtWzRz5kwNHz5cDz74YIXm3atXL8XGxqpz586KiYnRunXr9Morr6hv376qVq2aJOmpp57S3Llz1aVLF919993y8/PTa6+9pry8vDLf+11eiYmJioyM1LRp01StWjWFhoaqQ4cOx70nuFOnTqpevbqGDBmie++9VzabTe+99165uhh3795dd9xxhyZMmKDU1FT16tVL/v7+2rhxoz755BO99NJLuvbaaxUeHq4XX3xRt99+u9q1a6cbb7xR1atX1+rVq5WTk+N533abNm300UcfadSoUWrXrp3CwsJ05ZVXavjw4Xrttdd0yy23aMWKFapfv74+/fRTLV68WJMnT/as14p69dVX1aVLF7Vo0ULDhg1TgwYNtHfvXqWkpGjnzp1avXq1JKlp06bq0aOH2rRpo6ioKC1fvlyffvqpRo4ceVLz9dX2v+mmm/Txxx/rzjvv1IIFC9S5c2c5nU6tX79eH3/8sebMmePp7VFejz/+uL777jt17txZd911l5xOp1555RU1b95cqampJ53XksLDwzV16lTddNNNuuiiizRw4EDVqlVL27dv1//+9z917txZr7zySrmn5+/vr6eeekp33HGHLr30Ut1www3asmWL3n777Sq5p3vbtm167733JJmHvElmH5BMq/xNN91UafMCgNNalTwzHQDgE99++6116623Wo0bN7bCwsKsgIAAq2HDhtY999xj7d27t1T6zz77zOrSpYsVGhpqhYaGWo0bN7ZGjBhhbdiwwZOme/fuZb4K7OjXaL322mtWt27drBo1aliBgYFWYmKi9dBDD1mZmZlev1u5cqWVnJxshYWFWSEhIdYll1xiLVmyxCuN+5VLy5YtK/eyf/HFF1bTpk09r3RyvyLrWPm3LMtavHixdfHFF1vBwcFWXFyc5xVrKvEaprKW1e3111+32rRpYwUHB1vVqlWzWrRoYT388MPW7t27vdJ9+eWXVqdOnazg4GArPDzcat++vfXhhx96xmdnZ1s33nijFRkZaUnymtfevXutoUOHWjVr1rQCAgKsFi1alHo12rFe0VVy3NG/2bx5s3XzzTdbsbGxlr+/v1WnTh3riiuusD799FNPmqeeespq3769FRkZaQUHB1uNGze2nn766VKvoDvWPMvKj6+2f35+vjVx4kSrWbNmVmBgoFW9enWrTZs21vjx4732QUnWiBEjSv3+6Nd+WZZlzZs3z2rdurUVEBBgJSYmWm+88Yb1wAMPWEFBQcf97bHyf/QrvkoOT05OtiIiIqygoCArMTHRuuWWW6zly5cfd5mPNb1///vfVkJCghUYGGi1bdvWWrRokdW9e/dyvTKsMrnzV9bnVOcFAKqSzbIq+akhAACcZW666SalpKRU6v28ODP169ev0l6jBgA4N3BPNwAAJ7Bn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6iRJWrx4sSZOnKhRo0ZVegYBAAAAAHBr1aqVevbsqRYtWig5OVm9evXStddeq+rVq0uSqlWrptTUVGVnZ2vevHkaNWqUGjRooB49elRJfit8T7dlWZo8ebKef/557d69W5IUFxenhx56SPfee69s5ewScKbhnm4AAAAAZ5NS9zKfId3LJROXLlmyRN99951mzpyptLQ0LV26VAkJCaXS3n777dqxY8cxH6Z22t3TbbPZdP/99+v+++/XoUOHJJmaBAAAAADAGcxmK3cX76pms9nUuXNnde7cWePGjVN8fLxmzpxZZu9rl8ulvLy8KsilUeGg+8iRI7IsSyEhIapWrZq2bdumN998U02bNlWvXr18kUcAAAAAACRJS5cu1bx589SrVy9FR0dr6dKl+uuvv9SkSRNNmDBBbdu2VWJiovLy8vTNN9/ovffe09SpU6ssvxUOuq+++mr1799fd955pzIyMtS+fXsFBARo3759euGFF3TXXXf5Ip8AAAAAACg8PFyLFi3S5MmTlZWVpfj4eD3//PPq06ePFi9erLvvvls7d+5UcHCwGjdurPfff1833HBDleW3wvd016xZUz/88IOaNWumN954Qy+//LJWrVqlzz77TOPGjdO6det8ldcqxT3dAAAAAM4mZ/J7uiuTr+/ptlf0Bzk5OZ57uL/77jv1799fdrtdF198sbZt23ZSmQAAAAAA4GxU4aC7YcOGmjVrlufpb+77uNPT02kBBgAAAACghAoH3ePGjdODDz6o+vXrq0OHDurYsaMk0+rdunXrSs8gAAAAAABnqgo/SO3aa69Vly5dtGfPHrVq1cozvGfPnrrmmmsqNXMAAAAAAJzJKhx0S1JsbKxiY2O9hrVv375SMgQAAAAAwNmiwkH34cOH9eyzz2revHlKT0+Xy+XyGv/nn39WWuYAAAAAAL5VwRdanXV8vfwVDrpvv/12/fDDD7rppptUu3Zt2Ww2X+QLAAAAAOBD/v7+kswbqoKDg6s4N1UnJydHUvH6qGwVDrq//fZb/e9//1Pnzp19kR8AAAAAwCngcDgUGRmp9PR0SVJISMg51ahqWZZycnKUnp6uyMhIORwOn8ynwkF39erVFRUV5Yu8AAAAAABOIfezutyB97koMjKy1DPLKpPNqmAH9vfff19ffPGF3nnnHYWEhPgqX6edrKwsRUREKDMzk/eR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iT777DNJUmxsrCRp7969ql27tifN3r17deGFF3rSpKene02jsLBQBw4c8Py+LGPGjNGoUaM83zMzM1WvXj1avAEAAADgHOGO//7Oo9BO66C7c+fO2rBhg9ewP/74Q/Hx8ZKkhIQExcbGat68eZ4gOysrS0uXLtVdd90lSerYsaMyMjK0YsUKtWnTRpI0f/58uVwudejQ4ZjzDgwMVGBgoOe7e2XXrVu30pYPAAAAAHD6O3TokCIiIk7qt6f108uXLVumTp06afz48br++uv1yy+/aNiwYXr99dc1ePBgSdLEiRP17LPP6p133lFCQoLGjh2rNWvW6Pfff1dQUJAkqU+fPtq7d6+mTZumgoICDR06VG3bttUHH3xQ7ry4XC7t3r1b1apVk81m88nyVkRWVpbq1q2rHTt28DT10xzb6szBtjpzsK3OHGyrMwPb6czBtjpzsK3OHMfbVpZl6dChQ4qLi5PdfnKPRDutW7rbtWunmTNnasyYMXriiSeUkJCgyZMnewJuSXr44Yd1+PBhDR8+XBkZGerSpYtmz57tCbglacaMGRo5cqR69uwpu92uAQMGaMqUKRXKi91u13nnnVdpy1ZZwsPDOYjPEGyrMwfb6szBtjpzsK3ODGynMwfb6szBtjpzHGtbnWwLt9tpHXRL0hVXXKErrrjimONtNpueeOIJPfHEE8dMExUVVaFWbQAAAAAAKsNp/cowAAAAAADOZATdZ6jAwEA99thjXg97w+mJbXXmYFudOdhWZw621ZmB7XTmYFudOdhWZw5fb6vT+kFqAAAAAACcyWjpBgAAAADARwi6AQAAAADwEYJuAAAAAAB8hKD7DPXqq6+qfv36CgoKUocOHfTLL79UdZbOaRMmTFC7du1UrVo1RUdHq1+/ftqwYYNXmh49eshms3l97rzzzirK8bnr8ccfL7UdGjdu7Bmfm5urESNGqEaNGgoLC9OAAQO0d+/eKszxuat+/fqltpXNZtOIESMkcUxVpUWLFunKK69UXFycbDabZs2a5TXesiyNGzdOtWvXVnBwsJKSkrRx40avNAcOHNDgwYMVHh6uyMhI3XbbbcrOzj6FS3FuON62Kigo0OjRo9WiRQuFhoYqLi5ON998s3bv3u01jbKOxWefffYUL8nZ70TH1S233FJqO/Tu3dsrDceV751oO5V13bLZbHruuec8aTim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\n",
      "text/plain": [
       "<Figure size 1000x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Choose an example engine id\n",
    "example_id = train_df['id'].iloc[0]\n",
    "\n",
    "example_engine = train_df[train_df['id'] == example_id].copy()\n",
    "\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(example_engine['cycle'], example_engine['s2'], label='s2')\n",
    "plt.plot(example_engine['cycle'], example_engine['s3'], label='s3')\n",
    "plt.xlabel(\"Cycle\")\n",
    "plt.ylabel(\"Sensor value\")\n",
    "plt.title(f\"Sensor trajectories for engine id = {example_id}\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Target engineering: Remaining Useful Life (RUL)\n",
    "\n",
    "For each engine in the **training set**:\n",
    "\n",
    "\\\\[\n",
    "\\text{RUL} = \\max(\\text{cycle}) - \\text{cycle}\n",
    "\\\\]\n",
    "\n",
    "- At the last cycle of the engine: RUL = 0.\n",
    "- At earlier cycles: RUL > 0.\n",
    "\n",
    "We will create a new column `RUL` in `train_df`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:17:45.625372Z",
     "iopub.status.busy": "2025-12-02T16:17:45.625013Z",
     "iopub.status.idle": "2025-12-02T16:17:45.652842Z",
     "shell.execute_reply": "2025-12-02T16:17:45.651920Z",
     "shell.execute_reply.started": "2025-12-02T16:17:45.625347Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>cycle</th>\n",
       "      <th>max_cycle</th>\n",
       "      <th>RUL</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>192</td>\n",
       "      <td>191</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>192</td>\n",
       "      <td>190</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>192</td>\n",
       "      <td>189</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>192</td>\n",
       "      <td>188</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>192</td>\n",
       "      <td>187</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>192</td>\n",
       "      <td>186</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>192</td>\n",
       "      <td>185</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>192</td>\n",
       "      <td>184</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>192</td>\n",
       "      <td>183</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>192</td>\n",
       "      <td>182</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  cycle  max_cycle  RUL\n",
       "0   1      1        192  191\n",
       "1   1      2        192  190\n",
       "2   1      3        192  189\n",
       "3   1      4        192  188\n",
       "4   1      5        192  187\n",
       "5   1      6        192  186\n",
       "6   1      7        192  185\n",
       "7   1      8        192  184\n",
       "8   1      9        192  183\n",
       "9   1     10        192  182"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Compute max cycle per engine in the training set\n",
    "max_cycle_per_engine = train_df.groupby('id')['cycle'].max().rename('max_cycle')\n",
    "\n",
    "# Merge back to training dataframe\n",
    "train_df = train_df.merge(max_cycle_per_engine, on='id', how='left')\n",
    "\n",
    "# Compute RUL = max_cycle - current_cycle\n",
    "train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']\n",
    "\n",
    "# Sanity check\n",
    "train_df[['id', 'cycle', 'max_cycle', 'RUL']].head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.1 Distribution of RUL in the training set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:18:02.377921Z",
     "iopub.status.busy": "2025-12-02T16:18:02.377529Z",
     "iopub.status.idle": "2025-12-02T16:18:02.690782Z",
     "shell.execute_reply": "2025-12-02T16:18:02.690063Z",
     "shell.execute_reply.started": "2025-12-02T16:18:02.377897Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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7rnnlJ6e7m0XHBwsSVU+GrW4uNjbpr7UV36VHW+FhYW68cYb5XQ6lZGRoa5duyo4OFifffaZHn744WoVymbGa8Wx8swzz2js2LF6//33lZWVpfvuu09z5szR5s2b1bFjR5WVlclms+nDDz+sdDzcRwE0LpyxANCgLFmyRNKPH0DPJyAgQEOGDNGzzz6rPXv26PHHH9f69eu9l07U9V+29+3b5/PaMAzt37/f58k0bdq0UWFhYYVtz/1rf03GFhsbqyNHjlS4NKz8MpnY2Nhq7+tC/ezbt6/Ch9m67keSkpKSdOONN+qJJ57Q6dOnfcYgqcrva/jnP/9Zp+OoS7U53j7++GMdO3ZMixcv1v33369bbrlF8fHxDeam9sjISAUHB2v//v0V1lW2rCq9e/fW1KlTtXHjRv2///f/dPjwYS1cuFCS1LVrVxmGobi4OMXHx1f4GThwoHc/fHs40PBRWABoMNavX6/Zs2crLi5Oo0ePrrLd8ePHKyzr27evJMntdkuSWrVqJUmVftCvjddee83nw/3bb7+to0eP+lxr3rVrV23evFklJSXeZStWrKjwWNqajO3mm29WaWmpXnzxRZ/l8+bNk81mq9a17tVx8803Ky8vT2+++aZ32dmzZ/XCCy8oNDRUN954Y530U+7hhx/WsWPHfL4kr1+/foqMjNRf//pX779juffee0+HDx+us/nWtdocb+V/of/pGYKSkhK99NJLdTq22goMDFR8fLzee+89HTlyxLt8//79Fe7tqUxRUZHOnj3rs6x3794KCAjw/vuOHDlSgYGBmjVrVoUzJYZh6NixY97XrVq1qtblYQD8h0uhAPjFhx9+qK+++kpnz55Vfn6+1q9fr+zsbMXGxurvf//7eS95ycjI0MaNG5WUlKTY2FgVFBTopZdeUseOHXXddddJ+vFDfnh4uBYuXKjWrVurVatWGjBggOLi4mo13oiICF133XUaN26c8vPzNX/+fF1yySU+j8S9++679fbbb2vYsGH61a9+pa+//lpLly71uUG2pmO79dZbddNNN+kPf/iDDh48qCuuuEJZWVl6//33NWnSpAr7rq0JEyboT3/6k8aOHavc3Fx16dJFb7/9tj799FPNnz//vPe81Mbw4cPVq1cvPfvss0pNTZXdbldQUJCefvpppaSk6KqrrtLtt9+utm3baseOHXrllVfUp08fTZgwocK+tm/frscee6zC8kGDBnmPh/pWm+PtmmuuUZs2bZSSkqL77rtPNptNS5Ys8etla+eaOXOmsrKydO2112rixIneIrdXr14X/Jbx9evXKy0tTb/85S916aWX6uzZs1qyZIkCAwOVnJws6cfcHnvsMU2ZMkUHDx7UiBEj1Lp1ax04cEDvvvuuJkyYoN///veSfiw833zzTaWnp+uqq65SaGiobr311vqOAEBN+OdhVACaq/JHipb/BAUFGdHR0cbQoUON5557zuexpuXOfVzmunXrjNtuu82IiYkxgoKCjJiYGOOOO+4w/vnPf/ps9/777xs9e/b0Phqz/FGgN954o3H55ZdXOr6qHjf7xhtvGFOmTDEiIyONli1bGklJScY333xTYftnnnnGuOiiiwyHw2Fce+21xvbt2yvs83xjO/exooZhGCdPnjQmT55sxMTEGHa73ejWrZvxxz/+0SgrK/NppyoevVrVY3DPlZ+fb4wbN85o166dERQUZPTu3bvSx6fW9HGzVbVdvHhxpY9o/fDDD42bbrrJcDqdht1uN+Li4oz09HTj+++/r7CPnx5L5/7Mnj27ynFV9QhWM/nV5nj79NNPjYEDBxotW7Y0YmJijIceeshYs2aNIcn46KOPvO2qetzsH//4xwr7lGTMmDGjzua6bt0648orrzSCgoKMrl27Gn/961+NBx54wAgODj5vHv/617+Mu+66y+jatasRHBxsREREGDfddJOxdu3aCm3/7//+z7juuuuMVq1aGa1atTK6d+9upKamGnv37vW2OXXqlPHrX//aCA8PNyTx6FmgAbIZRgP60wgAAGjwRowYod27d1e49whA88Y9FgAAoErnPqVr3759WrVqlQYNGuSfAQFosDhjAQAAqtShQweNHTtWF198sb755hstWLBAbrdbO3bsqPD9LgCaN27eBgAAVRo2bJjeeOMN5eXlyeFwyOVy6YknnqCoAFABZywAAAAAmMY9FgAAAABMo7AAAAAAYBr3WEgqKyvTkSNH1Lp1a9lsNn8PBwAAAGgQDMPQyZMnFRMTo4CA85+ToLCQdOTIEXXq1MnfwwAAAAAapH//+9/q2LHjedtQWEhq3bq1pB8Dczqdlvfv8XiUlZWlhIQE2e12y/tvLsjZGuRsDXK2Bjlbg5ytQc7WaGo5FxUVqVOnTt7Py+dDYSF5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\n",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 4))\n",
    "train_df['RUL'].hist(bins=50)\n",
    "plt.xlabel(\"RUL\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.title(\"Distribution of RUL in training set\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature selection and correlation analysis\n",
    "\n",
    "We use the following predictors:\n",
    "- `setting1`, `setting2`, `setting3`\n",
    "- sensor measurements `s1` to `s21`\n",
    "\n",
    "The target is `RUL`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:18:22.819368Z",
     "iopub.status.busy": "2025-12-02T16:18:22.819004Z",
     "iopub.status.idle": "2025-12-02T16:18:22.824595Z",
     "shell.execute_reply": "2025-12-02T16:18:22.823790Z",
     "shell.execute_reply.started": "2025-12-02T16:18:22.819342Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of features: 24\n",
      "Feature columns: ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']\n"
     ]
    }
   ],
   "source": [
    "# Identify feature columns (all numeric except id, cycle, max_cycle, RUL)\n",
    "feature_cols = [col for col in train_df.columns \n",
    "                if col not in ['id', 'cycle', 'max_cycle', 'RUL']]\n",
    "\n",
    "target_col = 'RUL'\n",
    "\n",
    "print(\"Number of features:\", len(feature_cols))\n",
    "print(\"Feature columns:\", feature_cols)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.1 Correlation heatmap on a sample (for readability)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:18:50.387068Z",
     "iopub.status.busy": "2025-12-02T16:18:50.386344Z",
     "iopub.status.idle": "2025-12-02T16:18:50.915325Z",
     "shell.execute_reply": "2025-12-02T16:18:50.914462Z",
     "shell.execute_reply.started": "2025-12-02T16:18:50.387039Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.11/dist-packages/matplotlib/colors.py:721: RuntimeWarning: invalid value encountered in less\n",
      "  xa[xa < 0] = -1\n"
     ]
    },
    {
     "data": {
      "image/png": "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a665Rh06dFBmZqa2bNmip59+WpGRkXrwwQdP+r6WlpZq8ODBSklJ0TXXXKNVq1bprrvuktPp1JAhQyQd/4f4H//4R7333nu6/vrrFRcXp08//VQPP/ywvvjiC73yyivu6z3xxBO68MIL9cc//lGNGjXS66+/rptuuknV1dW6+eabJUlZWVm65ZZbFBoaqrvvvluSaoxOuOWWWxQVFaWMjAx98MEHeuqpp9S8eXNt3LhR5513nh544AG99dZbmjdvnrp166YxY8a4z33kkUf0xz/+UaNHj1ZlZaVWrlyp1NRUvfHGGxo6dKjHff71r3/phRde0K233iqHw6FFixZp8ODB2rx586+u+bF371598803uuiiizzaly9frqeeekqbN292T4c58T0+e/Zs3Xvvvbrmmms0adIklZSUaOHChfrd736nrVu3qnnz5pKkF198UUeOHNGNN96oiIgIbd68WQsXLtS3336rF198UZJ0ww036LvvvlNubq6WL19+0s/511RVVSk5OVmXXnqp5s+fryZNmrjvsXTpUo0fP1633nqrdu/erccee0xbt27V+++/r8aNG+vAgQNKSkpS69atNW3aNDVv3lx79uzRyy+//JsynSkn1nFp0aLFWblfeHi4OnXqpPfff1+33377WbknAABewQQAeKWysjJTkjls2LBT6l9YWGhKMidNmuTRfscdd5iSzLffftvdFh0dbUoyc3JyPPq+8847piSzY8eO5pEjR9zt1dXV5vnnn28mJyeb1dXV7vYjR46YHTp0MC+//HJ325IlS0xJ5u7duz36/dINN9xgNmnSxDx69Ki7bejQoWZ0dHSNvrt37zYlmUuWLHG39ejRw4yMjDS///57d9u///1vMyAgwBwzZoy7bcaMGaYkc8KECR7XHDFihBkREVHjXr80YMAAU5L597//3d3mcrnMqKgo86qrrnK3LV++3AwICDDfffddj/MXL15sSjLff/99d1tt70dycrLZsWNHj7YLL7zQHDBgQI2+J97jX34e/fr1Mw3DMKdMmeJuq6qqMs8999wa1/llhsrKSrNbt27mH/7wB492SaYk86OPPnK3ff3112ZwcLA5YsSIGtl+bv369aYk8/XXX6/x2tixY82mTZt6tO3Zs8cMDAw0Z8+e7dH+6aefmo0aNfJor+09zMzMNA3DML/++mt3280332zW9uPOie/1d955x6O9tu+1sWPHmpLMadOmefR99913TUnm888/79Gek5Pj0b569WpTkvnhhx/WyFFfJ/L9MvepOPGel5SUmCUlJebOnTvN+fPnm4ZhmN26dfP4Xjpxn3nz5tV6rXnz5tX4fT5gwADzwgsvPKUsSUlJZlxcXL2fAQAAX8YUJADwUiem/TRr1uyU+r/11luSpLS0NI/2v/zlL5JUY62YDh06KDk5udZrjR071mM9mMLCQn355Ze67rrr9P333+vgwYM6ePCgKioqNGjQIG3YsEHV1dV1Zvv5tQ4dOqSDBw/qsssu05EjR/T555+f0vP9XHFxsQoLCzVu3Di1bNnS3d69e3ddfvnl7vfi56ZMmeLx9WWXXabvv//e/T7/mtDQUI91M4KCgtSnTx+PKRsvvvii4uLi1KVLF/f7c/DgQf3hD3+QJL3zzjvuvj9/P8rKynTw4EENGDBAX331lcrKyk7hHThu4sSJHlN7+vbtK9M0NXHiRHdbYGCgevfuXWN6yc8zlJaWqqysTJdddpm2bNlS4z79+vVTr1693F+fd955GjZsmNauXesxjeyXvv/+e0mnPrLi5ZdfVnV1ta655hqP9zAqKkrnn39+ne9hRUWFDh48qP79+8s0TW3duvWU7ldfN954o8fXL774osLDw3X55Zd75O3Vq5dCQ0PdeU+M2nnjjTf0008/1euehw8f9rj2ialNJ75vThyn+n1TUVGh1q1bq3Xr1urcubPuuOMOJSQk6NVXX/X4XrJaixYtdPDgwbN2PwAAvAFTkADAS4WFhUk6XrA4FV9//bUCAgLUuXNnj/aoqCg1b95cX3/9tUd7hw4d6rzWL1/78ssvJR0vzNSlrKyszn9ob9u2Tffcc4/efvvtGgWP+hQcTjjxLD+fNnVCXFyc1q5dW2OR1PPOO8+j34mspaWl7ve6Lueee26Nf5y2aNFCn3zyifvrL7/8UkVFRWrdunWt1zhw4ID71++//75mzJihTZs21ViHpqysTOHh4b+a54RfPtOJ89q3b1+j/ZdrkrzxxhuaNWuWCgsLPdaoqe0f4eeff36NtgsuuEBHjhxRSUmJoqKifjWn+Yv1Z+ry5ZdfyjTNWu8nyWMXrG+++UbTp0/Xa6+9VuPZTud76mQaNWqkc889t0besrIyRUZG1nrOic98wIABuuqqq5SRkaGHH35YAwcO1PDhw3XdddfJ4XD86n2nTp2qZcuW1WgfPny4x9cDBgyosZ5NbYKDg/X6669Lkr799lvNnTvXvSDy6Tjdoo1pmme14AMAgDegAAMAXiosLEzt2rXTZ599Vq/zTvUfNb/2D65fvnZidMu8efPUo0ePWs+pa72WH3/8UQMGDFBYWJjuu+8+derUScHBwdqyZYvuuuuuXx05cyYFBgbW2n4qxYFTObe6ulpOp1MPPfRQrX1PFEV27dqlQYMGqUuXLnrooYfUvn17BQUF6a233tLDDz9cr/ejrly1tf8867vvvqs//vGP+t3vfqdFixapbdu2aty4sZYsWaIVK1ac8v1PJiIiQpJqFEjqUl1dLcMwtGbNmlqf4cT32LFjx3T55Zfrhx9+0F133aUuXbqoadOm2rt3r8aNG3dK72Fdv0/qGtHjcDhq7ARWXV2tyMjIOrdvPlGMMwxDq1at0gcffKDXX39da9eu1YQJE7RgwQJ98MEHv7rW0Z133ukx+mr//v3605/+pPnz5ys+Pt7dfqqjjAIDA92L4EpScnKyunTpohtuuEGvvfaau/3ErmM/X9T4504UDk93d7LS0lK1atXqtM4FAMBXUYABAC/2P//zP3rqqae0adMm9evX71f7RkdHq7q6Wl9++aXi4uLc7fv379ePP/6o6Ojo087RqVMnSceLQj//x9upyM/P1/fff6+XX35Zv/vd79ztu3fvrtH3VItHJ55lx44dNV77/PPP1apVq7O+RXCnTp3073//W4MGDfrV53j99dflcrn02muveYxg+fn0mhOsGiHw0ksvKTg4WGvXrvUYgbFkyZJa+58YAfVzX3zxhZo0aVLniB9J6tKli6TaP+vadOrUSaZpqkOHDrrgggvq7Pfpp5/qiy++0LJlyzwWFs7Nza3Rt6738ETB4scff/Ro/+VIsZPlXb9+vRISEk5pBMkll1yiSy65RLNnz9aKFSs0evRorVy5UpMmTarznK5du6pr167ur08smNurV6/T2ob6l9q2bavbb7/dvZDzie3CW7durSZNmtT6e0w6/nuvSZMmp11E2b17t0cBCQCAhoA1YADAi915551q2rSpJk2apP3799d4fdeuXXrkkUckSVdccYWk47vn/NyJERm/3NmmPnr16qVOnTpp/vz5Onz4cI3XS0pK6jz3xEiGn4/AqKys1KJFi2r0bdq06SlNH2nbtq169OihZcuWefwD+rPPPtO6devc78XZdM0112jv3r3Kzs6u8dp///tfVVRUSKr9/SgrK6u1+NG0adMaBYIzITAwUIZheIz22LNnj8dOTT+3adMmj7Vh/vOf/+jVV19VUlJSnaNwJOmcc85R+/bt9dFHH51SrpSUFAUGBiojI6PGyCTTNN1rytT2Hpqm6f698HMnCnG/fB+jo6MVGBhYYzvt2r4v63LNNdfo2LFjuv/++2u8VlVV5b5naWlpjec5MZLsl1uU2+GWW25RkyZNNGfOHHdbYGCgkpKS9Prrr+ubb77x6P/NN9/o9ddfP+nnX5eysjLt2rXrlHd3AwDAXzACBgC8WKdOnbRixQpde+21iouL05gxY9StWzdVVlZq48aNevHFFzVu3DhJUnx8vMaOHaunnnrKPe1n8+bNWrZsmYYPH67f//73p50jICBATz/9tIYMGaILL7xQ48eP1znnnKO9e/fqnXfeUVhYmHtdiV/q37+/WrRoobFjx+rWW2+VYRhavnx5rVN/evXqpRdeeEFpaWm6+OKLFRoaqiuvvLLW686bN09DhgxRv379NHHiRPc21OHh4Zo5c+ZpP+vp+t///V/985//1JQpU/TOO+8oISFBx44d0+eff65//vOfWrt2rXr37q2kpCQFBQXpyiuv1A033KDDhw8rOztbkZGRKi4u9rhmr1699MQTT2jWrFnq3LmzIiMj3Yv6/hZDhw7VQw89pMGDB+u6667TgQMH9Pjjj6tz584e69qc0K1bNyUnJ3tsQy1JGRkZJ73XsGHDtHr16lNa86NTp06aNWuW0tPTtWfPHg0fPlzNmjXT7t27tXr1al1//fW644471KVLF3Xq1El33HGH9u7dq7CwML300ku1TnU6sXjwrbfequTkZAUGBmrkyJEKDw9XamqqFi5cKMMw1KlTJ73xxhsea/WczIABA3TDDTcoMzNThYWFSkpKUuPGjfXll1/qxRdf1COPPKKrr75ay5Yt06JFizRixAh16tRJhw4dUnZ2tsLCwmwpFv5SRESExo8fr0WLFqmoqMg9gu6BBx7QJZdcoosuukjXX3+9YmJitGfPHj311FMyDEMPPPBAjWuVlJRo1qxZNdo7dOig0aNHSzq+hbxpmho2bJi1DwYAgLc5y7suAQBOwxdffGFOnjzZjImJMYOCgsxmzZqZCQkJ5sKFCz22cf7pp5/MjIwMs0OHDmbjxo3N9u3bm+np6R59TPP4NtRDhw6tcZ8TW/O++OKLtebYunWrmZKSYkZERJgOh8OMjo42r7nmGjMvL8/dp7ZtqN9//33zkksuMUNCQsx27dqZd955p7l27doa2+kePnzYvO6668zmzZubktxbUte2NbBpHt/mOCEhwQwJCTHDwsLMK6+80ty+fbtHnxPbUJeUlHi015azNnVtrTt27NgaW2ZXVlaaDz74oHnhhReaDofDbNGihdmrVy8zIyPDLCsrc/d77bXXzO7du5vBwcFmTEyM+eCDD5rPPvtsjTz79u0zhw4dajZr1syU5N5K+kT2X25rXNez1rbl8zPPPGOef/75psPhMLt06WIuWbLEff7PSTJvvvlm87nnnnP379mz5ylvg7xlyxZTUo3tuWvLdMJLL71kXnrppWbTpk3Npk2bml26dDFvvvlmc8eOHe4+27dvNxMTE83Q0FCzVatW5uTJk81///vfNb5PqqqqzFtuucVs3bq1aRiGx/OVlJSYV111ldmkSROzRYsW5g033GB+9tlntW5DXVdW0zTNp556yuzVq5cZEhJiNmvWzHQ6neadd95pfvfdd+73YNSoUeZ5551nOhwOMzIy0vyf//kfj629T9WZ2Ia6Nrt27TIDAwPNsWPHerQXFRWZ1157rRkZGWk2atTIjIyMNEeOHGkWFRXVuMaJLdtrOwYNGuTud+2115qXXnppvfMDAODrDNM8xa0JAABAg2MYhm6++WY99thjp32NQYMGqV27dlq+fPkZTAZftG/fPnXo0EErV65kBAwAoMFhDRgAAGCpBx54QC+88EK9FriFf8rKypLT6aT4AgBokFgDBgAAWKpv376qrKy0Owa8wM8X+gUAoKFhBAwAAAAAAIDFKMAAAIA6mab5m9Z/AQAA+K02bNigK6+8Uu3atZNhGHrllVdOek5+fr4uuugiORwOde7cWUuXLq3R5/HHH1dMTIyCg4PVt29fbd68+cyH/xkKMAAAAAAAwGtVVFQoPj5ejz/++Cn13717t4YOHarf//73Kiws1G233aZJkyZp7dq17j4vvPCC0tLSNGPGDG3ZskXx8fFKTk7WgQMHrHoMsQsSAAAAAAA4q1wul1wul0ebw+GQw+H41fMMw9Dq1as1fPjwOvvcddddevPNN/XZ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795q0qSJ+vfvrx07dnhkmzVrliIjI9WsWTNNmjRJ06ZNU48ePdyvP//887rpppvUo0cPdenSRU8//bSqq6uVl5dX37cBAAAAAADglNVriElxcbFGjRqluXPnasSIETp06JDeffddmaap559/XtOnT9djjz2mnj17auvWrZo8ebKaNm2qsWPHavPmzerTp4/Wr1+vCy+8UEFBQQoKClJRUZHKy8u1ZMkSSVLLli313Xff1Xr/u+++WwsWLFDr1q01ZcoUTZgwQe+//76k48WV2bNna9GiRUpISNDKlSu1YMECd+GnNkeOHNFPP/2kli1b1udtAAAAAAAAqJd6F2CqqqqUkpKi6OhoSZLT6ZQkzZgxQwsWLFBKSoqk4yNetm/frieffFJjx45V69atJUkRERGKiopyXzMkJEQul8ujrS6zZ8/WgAEDJEnTpk3T0KFDdfToUQUHB2vhwoWaOHGixo8fL0maPn261q1bp8OHD9d5vbvuukvt2rVTYmJinX1cLpdcLtcv2irlcASdNC8AAAAAAGdDQKBhdwScRL2mIMXHx2vQoEFyOp1KTU1Vdna2SktLVVFRoV27dmnixIkKDQ11H7NmzdKuXbvOWNju3bu7f922bVtJ0oEDByRJO3bsUJ8+fTz6//Lrn5szZ45Wrlyp1atXKzg4uM5+mZmZCg8P9zieePLJ3/IYAAAAAACgganXCJjAwEDl5uZq48aNWrdunRYuXKi7775br7/+uqTjuwz17du3xjlnSuPGjd2/Nozj1b3q6up6X2f+/PmaM2eO1q9f71HUqU16errS0tI82or/80297wkAAAAAABquem8zZBiGEhISlJCQoOnTpys6Olrvv/++2rVrp6+++kqjR4+u9bygoONTdo4dO1aj/ZdtpyM2NlYffvihxowZ42778MMPa/SbO3euZs+erbVr16p3794nva7D4ZDD4fBo+4HpRwAAAAAAL2IEMAXJ29WrAFNQUKC8vDwlJSUpMjJSBQUFKikpUVxcnDIyMnTrrbcqPDxcgwcPlsvl0kcffaTS0lKlpaUpMjJSISEhysnJ0bnnnqvg4GCFh4crJiZGa9eu1Y4dOxQREaHw8PDTepBbbrlFkydPVu/evdW/f3+98MIL+uSTT9SxY0d3nwcffFDTp0/XihUrFBMTo3379kmSe8oUAAAAAACAFeq1BkxYWJg2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n4qKCsXHx+vxxx8/y+kAAAAAAEBD5nOL8K5atUoZGRnauXOnmjRpop49e+rVV1/VxRdfrIsvvliSNG3atFrPHTJkiIYMGXI24wIAAAAAAPjWCJji4mKNGjVKEyZMUFFRkfLz85WSkiLTNO2OBgAAAAAALPL4448rJiZGwcHB6tu3rzZv3vyr/V988UV16dJFwcHBcjqdeuuttzxeN01T06dPV9u2bRUSEqLExER9+eWXVj6C7xVgqqqqlJKSopiYGDmdTt10000KDQ217J4ul0vl5eUeh8tVadn9AAAAAACoL1OGTx318cILLygtLU0zZszQli1bFB8fr+TkZB04cKDW/hs3btSoUaM0ceJEbd26VcOHD9fw4cP12WefufvMnTtXjz76qBYvXqyCggI1bdpUycnJOnr06G/6HH6NTxVg4uPjNWjQIDmdTqWmpio7O1ulpaWW3jMzM1Ph4eEexxNPPmnpPQEAAAAAwHEPPfSQJk+erPHjx6tr165avHixmjRpomeffbbW/o888ogGDx6sv/71r4qLi9P999+viy66SI899pik46NfsrKydM8992jYsGHq3r27/v73v+u7776zdLMenyrABAYGKjc3V2vWrFHXrl21cOFCxcbGavfu3ZbdMz09XWVlZR7HjTfcYNn9AAAAAADwd7XPNnHV6FdZWamPP/5YiYmJ7raAgAAlJiZq06ZNtV5706ZNHv0lKTk52d1/9+7d2rdvn0ef8PBw9e3bt85rngk+VYCRJMMwlJCQoIyMDG3dulVBQUFavXq1ZfdzOBwKCwvzOByOIMvuBwAAAACAv6tttklmZmaNfgcPHtSxY8fUpk0bj/Y2bdpo3759tV573759v9r/xH/rc80zwad2QSooKFBeXp6SkpIUGRmpgoIClZSUKC4uTpWVldq+fbuk4xWyvXv3qrCwUKGhoercubMk6fDhw9q5c6f7ert371ZhYaFatmyp8847z5ZnAgAAAADgtzKN+q2rYrf09HSlpaV5tDkcDpvSnB0+VYAJCwvThg0blJWVpfLyckVHR2vBggUaMmSI9uzZo549e7r7zp8/X/Pnz9eAAQOUn58vSfroo4/0+9//3t3nxIc9duxYLV269Gw+CgAAAAAADZbD4TilgkurVq0UGBio/fv3e7Tv379fUVFRtZ4TFRX1q/1P/Hf//v1q27atR58e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iYiIMG3atDG33HKLCQoKMg899JD57//+b1O7dm1z8803m6KiIkuyGGPMl19+aZo3b26qV69uOnbsaO677z5z3333mY4dO5rq1aubuLg48+WXX1qW52ysfAafD7vlAS6EwxjajXrTM888c9btBw4c0OzZsy2put9+++1q0qSJFi1apOPHj2v06NF69dVXlZ2draSkJH3//feKjo627BuApKQkPfzww3rssce0bt06de/eXVOnTtXw4cMlSXPmzFFWVpY+/PBDr2fhc6qcnT4np9Op2267Tc2aNfNYv3TpUt11112qXbu2JFnSKsfpdKqwsFCRkZG65557VF5ertdff13VqlVTeXm5evXqpWPHjumtt97yehZJql69ur744gs1bdpUCQkJmjBhgtLS0tzb/+///k/Dhg2zpKWQ0+nUZ599prVr12rx4sXasWOHEhIS9Oijj6pXr16qU6eO1zP8qmnTplq8eLGSk5P11Vdf6aqrrtLrr7+uu+++W9Ivrbv69++vffv2eT1LUlKS/vSnP2ngwIHuaz/22GPatWuXTp06pW7duqlJkyaWtSq76aablJiYqAULFsjhcHhsM8Zo4MCB2r59uzZt2uT1LP/617/Ouv2rr77S448/bslzr0ePHjp16pReeOEFHTlyRMOGDdPOnTu1fv16NW3a1PJn8M0336xOnTppypQpeuWVV/TnP/9ZgwYN0tSpUyVJY8eO1ZYtW7R27VqvZ3E6nUpMTHQ/a3/1/vvvq23btqpZs6YcDoclrXJ++wx+8MEHlZ+fr3feeUcRERE6duyYevToofr161vWTaFGjRratWuXYmJidP3112vQoEHq37+/e/uKFSs0depU7dixw+tZnE6nsrOz9dZbb2n58uU6evSounXrpv79++uOO+6Q02ldA/i4uDjNnDlTqamp2rZtm9q0aaMXX3xRvXr1kiRlZWVp1KhRlvy/6ZZbbtHtt9+uiRMnSpJeeuklPfvss/roo490+PBhderUSbfddpv+53/+x+tZpF9+36tZs6aWLl2q8PBwj21FRUXq3bu3Tpw4oTVr1ng9y/bt28+6/fPPP1d6erpPWglVZNu2bbr++uttkwe4ID4uAF32HA6HiY6ONrGxsRUu0dHRllVw69SpY3bv3u2xLjMz09SpU8fk5uZa/q1ezZo1Pb6dCg4O9mgxsGvXLlOvXj1LsvA5Vc5On9PLL79sGjdubBYvXuyxvlq1ambHjh2WZPiVw+Ew33//vTHGmCZNmrhbn/zqk08+MQ0bNrQsT8OGDc2mTZuMMcY0aNDAfPLJJx7bv/jiCxMaGmpJlt++N8YYk5OTYwYMGGAiIiJMaGioSU9Pt+wb+9DQULN//3736+DgYI8WZfn5+aZGjRqWZKlevbrJz893vy4vLzfBwcHu1hUbNmww9evXtyTLr3l27dpV6fZdu3aZ6tWrW5Ll12/sHQ5HpYtVz73IyEizfft29+vy8nIzcOBA07RpU7N3717Ln8Hh4eHub8HLyspMtWrVPP59f/rpp6ZBgwaWZMnMzDTNmjU749+vr5/BzZs3N2vXrvXY/p///Mc0adLEsjz16tVzt/aLjIw0eXl5Htv37Nnjk2dwaWmp+d///V/TpUsXExQUZKKjo824ceMsa1lxrmfwvn37LHsGh4aGmr1797pfl5WVmeDgYFNYWGiMMWbt2rUmOjrakiy/5vn0008r3b59+3ZLf2Yqewb/ut5OLU5oAQN/Vs3XBaDLXUxMjGbMmKH77ruvwu15eXlq06aNZXlOnjzp8XrMmDGqVq2aUlJStHjxYstySFJwcLBH/1aXy6WwsDCP11aNIcHnVDk7fU7333+/brrpJj344IN6++23tWjRIktbU/yWw+FwtxpwOp2KiIjw2F67dm0dPnzYsjw9evTQ1KlT9cYbb+juu+/W3//+dz333HPujPPmzdN1111nWZ7fuuGGG3TDDTfo6aef1quvvqrnn39et99+uyXfXDVt2lSbNm1S06ZNtXnzZjkcDuXm5uqaa66RJOXk5KhRo0ZezyFJjRo10u7duxUbGyvpl7GlysvLVa9ePUlS48aNdezYMUuySFJUVJRyc3PVsmXLCrfn5uaqQYMGlmRp2LCh/v73v7tbJv2elc/gEydOqFq1///XI4fDofnz52vIkCHq2LGjTwZ+/O2zpnr16h7Pm1q1auno0aOW5BgzZow6d+6sBx98UN27d1dmZqaCg4MtuXZFfn1fTp48qYYNG3psa9SokaXjGHXr1k3z58/XokWL1LFjR61cuVKJiYnu7a+++qri4uIsy/Or4OBg3Xfffbrvvvv09ddfa/HixXrhhRc0ffp0S57BUVFR2rlzp5o2baovv/xSZWVl2rlzp/sZvGPHDkVGRno9hyRFRkaqoKBAzZs3l/TLuCanT592tz656qqr9NNPP1mSRfrl94R9+/bp2muvrXD7vn37zmht5i1169bVzJkz1blz5wq379ixQ927d7cki3R+LdMBf0UBxsvatGmjLVu2VPqHvcPhsGxgzGuvvVYbN25U69atPdaPHDlS5eXlSk9PtyTHr+Li4vT555+rRYsWkn55mNaqVcu9fe/evWrcuLElWficKmenz0mSYmNjtWHDBmVkZCgxMVELFy48o/uEFYwxuvrqq+VwOHTs2DFt377d4zPbs2ePoqKiLMszbdo0JScnq2XLlmrfvr1ee+01ZWdn6+qrr9aePXv0008/WdKM+Wxq1Kihvn37qm/fvvriiy8suebAgQPVt29fLVq0SFu2bNHs2bM1btw4ff7553I6nZo/f74ef/xxS7L07t1bjz76qP7617/K5XLpqaee0l133aWQkBBJvxQZft+9zptGjhypAQMGaMuWLercubO72PL9999r3bp1WrhwoWbPnm1Jll+fwZUVYKx8Brds2VIff/yx4uPjPdY/++yzkqS77rrLkhy/io2N1Zdffqkrr7xSktwFxV99/fXXZxQfvKldu3basmWLBg8erLZt22r58uU+eQZLUufOnVWtWjUVFRVp9+7dHn/I7t+/313ctMKMGTPUoUMHdezYUW3bttWcOXO0fv16xcfHa/fu3froo4+UlZVlWZ6KNG3aVE8++aQmTpyod99915Jr9urVS71799bdd9+tdevWadSoURo5cqR+/PFHORwOTZ06Vffee68lWe655x4NHDhQs2bNksvl0uTJk9WxY0eFhoZKknbv3m1ZQV6SHn30UfXu3Vvjx4+v8Bk8ZcoUDR061JIsbdq00XfffaeYmJgKtx85csTS2c6efvrpc+7z2+cg4E8owHjZpEmTVFxcXOn2Vq1aeYya7029e/fW+++/7x5/4LdGjRolY4wWLFhgSRZJGjdunEfrhd/3f/34448rLYhUNT6nytnpc/qV0+lURkaGbr/9dvXu3dsnUy3/fpyO33+z+dFHH6lHjx6W5YmIiNDGjRv1/PPP66233lJsbKzKy8tVWlqq9PR0DRo0yLJCWceOHd1FhcpcffXVlmQZNmyYIiMjtWnTJvXr10/p6enuMXKKi4s1fPhwjRs3zpIs48aN0/HjxzV58mSVlJSoS5cuHmMNNGrUSPPnz7ck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\n",
      "text/plain": [
       "<Figure size 1200x1000 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Sample for faster plotting if dataset is large\n",
    "sample_df = train_df.sample(n=5000, random_state=42) if len(train_df) > 5000 else train_df.copy()\n",
    "\n",
    "corr = sample_df[feature_cols + [target_col]].corr()\n",
    "\n",
    "plt.figure(figsize=(12, 10))\n",
    "sns.heatmap(corr, cmap='coolwarm', center=0)\n",
    "plt.title(\"Correlation heatmap (features + RUL)\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Train/Validation split and scaling\n",
    "\n",
    "We perform:\n",
    "- Train/validation split (e.g. 80% / 20%)\n",
    "- Standardization of features using `StandardScaler` inside a `Pipeline`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:19:06.321755Z",
     "iopub.status.busy": "2025-12-02T16:19:06.321105Z",
     "iopub.status.idle": "2025-12-02T16:19:06.334878Z",
     "shell.execute_reply": "2025-12-02T16:19:06.334120Z",
     "shell.execute_reply.started": "2025-12-02T16:19:06.321726Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training size   : (16504, 24)\n",
      "Validation size : (4127, 24)\n"
     ]
    }
   ],
   "source": [
    "X = train_df[feature_cols]\n",
    "y = train_df[target_col]\n",
    "\n",
    "X_train, X_valid, y_train, y_valid = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "print(\"Training size   :\", X_train.shape)\n",
    "print(\"Validation size :\", X_valid.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Helper function for model evaluation\n",
    "We will use RMSE, MAE and R² on the validation set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:19:35.615400Z",
     "iopub.status.busy": "2025-12-02T16:19:35.614755Z",
     "iopub.status.idle": "2025-12-02T16:19:35.621121Z",
     "shell.execute_reply": "2025-12-02T16:19:35.620438Z",
     "shell.execute_reply.started": "2025-12-02T16:19:35.615375Z"
    }
   },
   "outputs": [],
   "source": [
    "def evaluate_regressor(model, X_train, y_train, X_valid, y_valid, model_name=\"Model\"):\n",
    "    \"\"\"\n",
    "    Fit the model, predict on validation set and print metrics.\n",
    "    \"\"\"\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_valid)\n",
    "    \n",
    "    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))\n",
    "    mae = mean_absolute_error(y_valid, y_pred)\n",
    "    r2 = r2_score(y_valid, y_pred)\n",
    "    \n",
    "    print(f\"=== {model_name} ===\")\n",
    "    print(f\"RMSE: {rmse:.2f}\")\n",
    "    print(f\"MAE : {mae:.2f}\")\n",
    "    print(f\"R²  : {r2:.4f}\")\n",
    "    print()\n",
    "    \n",
    "    return rmse, mae, r2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Baseline and tree-based models\n",
    "\n",
    "We will compare several models:\n",
    "\n",
    "1. **LinearRegression** (baseline)\n",
    "2. **RandomForestRegressor**\n",
    "3. **GradientBoostingRegressor**\n",
    "4. **ExtraTreesRegressor**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:19:55.912947Z",
     "iopub.status.busy": "2025-12-02T16:19:55.912641Z",
     "iopub.status.idle": "2025-12-02T16:20:35.669408Z",
     "shell.execute_reply": "2025-12-02T16:20:35.668538Z",
     "shell.execute_reply.started": "2025-12-02T16:19:55.912927Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Linear Regression ===\n",
      "RMSE: 44.34\n",
      "MAE : 34.05\n",
      "R²  : 0.5696\n",
      "\n",
      "=== Random Forest ===\n",
      "RMSE: 41.40\n",
      "MAE : 29.62\n",
      "R²  : 0.6249\n",
      "\n",
      "=== Gradient Boosting ===\n",
      "RMSE: 41.30\n",
      "MAE : 29.73\n",
      "R²  : 0.6266\n",
      "\n",
      "=== Extra Trees ===\n",
      "RMSE: 41.13\n",
      "MAE : 29.45\n",
      "R²  : 0.6297\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{'LinearRegression': (44.34186807015132,\n",
       "  34.050297279230946,\n",
       "  0.569645560744755),\n",
       " 'RandomForest': (41.39910974865987, 29.622609643809064, 0.6248712754209377),\n",
       " 'GradientBoosting': (41.302550185656386,\n",
       "  29.727488815765632,\n",
       "  0.6266191402135162),\n",
       " 'ExtraTrees': (41.132683512064254, 29.453428640659073, 0.6296840618291522)}"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results = {}\n",
    "\n",
    "# 1) Linear Regression (baseline)\n",
    "linreg_pipe = Pipeline(steps=[\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('model', LinearRegression())\n",
    "])\n",
    "\n",
    "results['LinearRegression'] = evaluate_regressor(\n",
    "    linreg_pipe, X_train, y_train, X_valid, y_valid, model_name=\"Linear Regression\"\n",
    ")\n",
    "\n",
    "# 2) Random Forest\n",
    "rf_pipe = Pipeline(steps=[\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('model', RandomForestRegressor(\n",
    "        n_estimators=200,\n",
    "        random_state=42,\n",
    "        n_jobs=-1\n",
    "    ))\n",
    "])\n",
    "\n",
    "results['RandomForest'] = evaluate_regressor(\n",
    "    rf_pipe, X_train, y_train, X_valid, y_valid, model_name=\"Random Forest\"\n",
    ")\n",
    "\n",
    "# 3) Gradient Boosting\n",
    "gb_pipe = Pipeline(steps=[\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('model', GradientBoostingRegressor(\n",
    "        n_estimators=300,\n",
    "        learning_rate=0.05,\n",
    "        max_depth=3,\n",
    "        random_state=42\n",
    "    ))\n",
    "])\n",
    "\n",
    "results['GradientBoosting'] = evaluate_regressor(\n",
    "    gb_pipe, X_train, y_train, X_valid, y_valid, model_name=\"Gradient Boosting\"\n",
    ")\n",
    "\n",
    "# 4) Extra Trees\n",
    "et_pipe = Pipeline(steps=[\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('model', ExtraTreesRegressor(\n",
    "        n_estimators=300,\n",
    "        random_state=42,\n",
    "        n_jobs=-1\n",
    "    ))\n",
    "])\n",
    "\n",
    "results['ExtraTrees'] = evaluate_regressor(\n",
    "    et_pipe, X_train, y_train, X_valid, y_valid, model_name=\"Extra Trees\"\n",
    ")\n",
    "\n",
    "results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Hyperparameter tuning with GridSearchCV (Random Forest example)\n",
    "\n",
    "We perform a grid search on a subset of hyperparameters for `RandomForestRegressor` using cross-validation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:21:58.067410Z",
     "iopub.status.busy": "2025-12-02T16:21:58.067018Z",
     "iopub.status.idle": "2025-12-02T16:25:44.092545Z",
     "shell.execute_reply": "2025-12-02T16:25:44.091658Z",
     "shell.execute_reply.started": "2025-12-02T16:21:58.067386Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 12 candidates, totalling 36 fits\n",
      "Best parameters: {'model__max_depth': 10, 'model__min_samples_split': 5, 'model__n_estimators': 200}\n",
      "Best CV RMSE  : 41.461387203103875\n"
     ]
    }
   ],
   "source": [
    "param_grid = {\n",
    "    'model__n_estimators': [100, 200],\n",
    "    'model__max_depth': [None, 10, 20],\n",
    "    'model__min_samples_split': [2, 5],\n",
    "}\n",
    "\n",
    "rf_base = Pipeline(steps=[\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))\n",
    "])\n",
    "\n",
    "grid_search = GridSearchCV(\n",
    "    estimator=rf_base,\n",
    "    param_grid=param_grid,\n",
    "    scoring='neg_root_mean_squared_error',\n",
    "    cv=3,\n",
    "    n_jobs=-1,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "print(\"Best parameters:\", grid_search.best_params_)\n",
    "print(\"Best CV RMSE  :\", -grid_search.best_score_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8.1 Evaluate the best Random Forest model on the validation set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:26:03.900880Z",
     "iopub.status.busy": "2025-12-02T16:26:03.900075Z",
     "iopub.status.idle": "2025-12-02T16:26:03.989848Z",
     "shell.execute_reply": "2025-12-02T16:26:03.988979Z",
     "shell.execute_reply.started": "2025-12-02T16:26:03.900853Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Best Random Forest on validation set ===\n",
      "RMSE: 41.10\n",
      "MAE : 29.34\n",
      "R²  : 0.6303\n"
     ]
    }
   ],
   "source": [
    "best_rf = grid_search.best_estimator_\n",
    "\n",
    "y_valid_pred = best_rf.predict(X_valid)\n",
    "rmse_valid = np.sqrt(mean_squared_error(y_valid, y_valid_pred))\n",
    "mae_valid = mean_absolute_error(y_valid, y_valid_pred)\n",
    "r2_valid = r2_score(y_valid, y_valid_pred)\n",
    "\n",
    "print(\"=== Best Random Forest on validation set ===\")\n",
    "print(f\"RMSE: {rmse_valid:.2f}\")\n",
    "print(f\"MAE : {mae_valid:.2f}\")\n",
    "print(f\"R²  : {r2_valid:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:26:14.611794Z",
     "iopub.status.busy": "2025-12-02T16:26:14.611477Z",
     "iopub.status.idle": "2025-12-02T16:26:14.618346Z",
     "shell.execute_reply": "2025-12-02T16:26:14.617150Z",
     "shell.execute_reply.started": "2025-12-02T16:26:14.611770Z"
    }
   },
   "source": [
    "## 9. Final training on full training data\n",
    "\n",
    "We retrain the **best model** on the whole training dataset (`X`, `y`) to use all available data.\n",
    "Here, we use the tuned Random Forest."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:26:23.192779Z",
     "iopub.status.busy": "2025-12-02T16:26:23.192042Z",
     "iopub.status.idle": "2025-12-02T16:26:33.640144Z",
     "shell.execute_reply": "2025-12-02T16:26:33.639281Z",
     "shell.execute_reply.started": "2025-12-02T16:26:23.192748Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Pipeline(steps=[(&#x27;scaler&#x27;, StandardScaler()),\n",
       "                (&#x27;model&#x27;,\n",
       "                 RandomForestRegressor(max_depth=10, min_samples_split=5,\n",
       "                                       n_estimators=200, n_jobs=-1,\n",
       "                                       random_state=42))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;scaler&#x27;, StandardScaler()),\n",
       "                (&#x27;model&#x27;,\n",
       "                 RandomForestRegressor(max_depth=10, min_samples_split=5,\n",
       "                                       n_estimators=200, n_jobs=-1,\n",
       "                                       random_state=42))])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor(max_depth=10, min_samples_split=5, n_estimators=200,\n",
       "                      n_jobs=-1, random_state=42)</pre></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "Pipeline(steps=[('scaler', StandardScaler()),\n",
       "                ('model',\n",
       "                 RandomForestRegressor(max_depth=10, min_samples_split=5,\n",
       "                                       n_estimators=200, n_jobs=-1,\n",
       "                                       random_state=42))])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Refit best model on all training data\n",
    "best_rf.fit(X, y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Evaluation on the test set using PM_truth\n",
    "\n",
    "The file `PM_truth.csv` gives the *true RUL* for each engine in the test set, but only at the **last cycle**.\n",
    "\n",
    "Steps:\n",
    "1. For each engine in `test_df`, keep only the last cycle.\n",
    "2. Align these rows with `truth_df` by `id`.\n",
    "3. Predict RUL for these last cycles using the trained model.\n",
    "4. Compute RMSE and MAE between predictions and true RUL."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:39:29.870755Z",
     "iopub.status.busy": "2025-12-02T16:39:29.870432Z",
     "iopub.status.idle": "2025-12-02T16:39:29.900099Z",
     "shell.execute_reply": "2025-12-02T16:39:29.899417Z",
     "shell.execute_reply.started": "2025-12-02T16:39:29.870736Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of test_last_df: (100, 26)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>cycle</th>\n",
       "      <th>setting1</th>\n",
       "      <th>setting2</th>\n",
       "      <th>setting3</th>\n",
       "      <th>s1</th>\n",
       "      <th>s2</th>\n",
       "      <th>s3</th>\n",
       "      <th>s4</th>\n",
       "      <th>s5</th>\n",
       "      <th>...</th>\n",
       "      <th>s12</th>\n",
       "      <th>s13</th>\n",
       "      <th>s14</th>\n",
       "      <th>s15</th>\n",
       "      <th>s16</th>\n",
       "      <th>s17</th>\n",
       "      <th>s18</th>\n",
       "      <th>s19</th>\n",
       "      <th>s20</th>\n",
       "      <th>s21</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>31</td>\n",
       "      <td>-0.0006</td>\n",
       "      <td>0.0004</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.58</td>\n",
       "      <td>1581.22</td>\n",
       "      <td>1398.91</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>521.79</td>\n",
       "      <td>2388.06</td>\n",
       "      <td>8130.11</td>\n",
       "      <td>8.4024</td>\n",
       "      <td>0.03</td>\n",
       "      <td>393</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>38.81</td>\n",
       "      <td>23.3552</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>49</td>\n",
       "      <td>0.0018</td>\n",
       "      <td>-0.0001</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.55</td>\n",
       "      <td>1586.59</td>\n",
       "      <td>1410.83</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>521.74</td>\n",
       "      <td>2388.09</td>\n",
       "      <td>8126.90</td>\n",
       "      <td>8.4505</td>\n",
       "      <td>0.03</td>\n",
       "      <td>391</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>38.81</td>\n",
       "      <td>23.2618</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>126</td>\n",
       "      <td>-0.0016</td>\n",
       "      <td>0.0004</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.88</td>\n",
       "      <td>1589.75</td>\n",
       "      <td>1418.89</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>520.83</td>\n",
       "      <td>2388.14</td>\n",
       "      <td>8131.46</td>\n",
       "      <td>8.4119</td>\n",
       "      <td>0.03</td>\n",
       "      <td>395</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>38.93</td>\n",
       "      <td>23.2740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>106</td>\n",
       "      <td>0.0012</td>\n",
       "      <td>0.0004</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.78</td>\n",
       "      <td>1594.53</td>\n",
       "      <td>1406.88</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>521.88</td>\n",
       "      <td>2388.11</td>\n",
       "      <td>8133.64</td>\n",
       "      <td>8.4634</td>\n",
       "      <td>0.03</td>\n",
       "      <td>395</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>38.58</td>\n",
       "      <td>23.2581</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>98</td>\n",
       "      <td>-0.0013</td>\n",
       "      <td>-0.0004</td>\n",
       "      <td>100.0</td>\n",
       "      <td>518.67</td>\n",
       "      <td>642.27</td>\n",
       "      <td>1589.94</td>\n",
       "      <td>1419.36</td>\n",
       "      <td>14.62</td>\n",
       "      <td>...</td>\n",
       "      <td>521.00</td>\n",
       "      <td>2388.15</td>\n",
       "      <td>8125.74</td>\n",
       "      <td>8.4362</td>\n",
       "      <td>0.03</td>\n",
       "      <td>394</td>\n",
       "      <td>2388</td>\n",
       "      <td>100.0</td>\n",
       "      <td>38.75</td>\n",
       "      <td>23.4117</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  cycle  setting1  setting2  setting3      s1      s2       s3       s4  \\\n",
       "0   1     31   -0.0006    0.0004     100.0  518.67  642.58  1581.22  1398.91   \n",
       "1   2     49    0.0018   -0.0001     100.0  518.67  642.55  1586.59  1410.83   \n",
       "2   3    126   -0.0016    0.0004     100.0  518.67  642.88  1589.75  1418.89   \n",
       "3   4    106    0.0012    0.0004     100.0  518.67  642.78  1594.53  1406.88   \n",
       "4   5     98   -0.0013   -0.0004     100.0  518.67  642.27  1589.94  1419.36   \n",
       "\n",
       "      s5  ...     s12      s13      s14     s15   s16  s17   s18    s19  \\\n",
       "0  14.62  ...  521.79  2388.06  8130.11  8.4024  0.03  393  2388  100.0   \n",
       "1  14.62  ...  521.74  2388.09  8126.90  8.4505  0.03  391  2388  100.0   \n",
       "2  14.62  ...  520.83  2388.14  8131.46  8.4119  0.03  395  2388  100.0   \n",
       "3  14.62  ...  521.88  2388.11  8133.64  8.4634  0.03  395  2388  100.0   \n",
       "4  14.62  ...  521.00  2388.15  8125.74  8.4362  0.03  394  2388  100.0   \n",
       "\n",
       "     s20      s21  \n",
       "0  38.81  23.3552  \n",
       "1  38.81  23.2618  \n",
       "2  38.93  23.2740  \n",
       "3  38.58  23.2581  \n",
       "4  38.75  23.4117  \n",
       "\n",
       "[5 rows x 26 columns]"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 10.1 Extract the last cycle for each engine in the test set\n",
    "last_cycle_test = (\n",
    "    test_df\n",
    "        .groupby('id', as_index=False)['cycle']\n",
    "        .max()  # creates columns [\"id\", \"cycle\"]\n",
    ")\n",
    "\n",
    "# Keep only rows that correspond to each engine's last cycle\n",
    "test_last_df = test_df.merge(last_cycle_test,\n",
    "                             on=['id', 'cycle'],\n",
    "                             how='inner')\n",
    "\n",
    "print(\"Shape of test_last_df:\", test_last_df.shape)\n",
    "test_last_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10.2 Prepare the truth dataframe\n",
    "\n",
    "Depending on the CMAPSS format, `PM_truth.csv` usually contains only one RUL value per engine, in order.\n",
    "\n",
    "Example:\n",
    "\n",
    "Engine 1 → truth_df row 1  \n",
    "Engine 2 → truth_df row 2  \n",
    "..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:46:09.978007Z",
     "iopub.status.busy": "2025-12-02T16:46:09.977595Z",
     "iopub.status.idle": "2025-12-02T16:46:10.004320Z",
     "shell.execute_reply": "2025-12-02T16:46:10.003408Z",
     "shell.execute_reply.started": "2025-12-02T16:46:09.977981Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Truth_df shape : (100, 2)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RUL_true</th>\n",
       "      <th>id</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   RUL_true  id\n",
       "0         1   1\n",
       "1         2   2\n",
       "2         3   3\n",
       "3         4   4\n",
       "4         5   5"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 10.2 Build the truth dataframe (robust version)\n",
    "\n",
    "# Read with header (let pandas detect the column name)\n",
    "truth_raw = pd.read_csv(truth_path)\n",
    "\n",
    "# Take the first column, whatever its name is (often \"cycle\")\n",
    "first_col = truth_raw.columns[0]\n",
    "\n",
    "truth_df = truth_raw[[first_col]].rename(columns={first_col: 'RUL_true'})\n",
    "\n",
    "# Add engine IDs: 1, 2, ..., number_of_test_engines\n",
    "truth_df['id'] = np.arange(1, len(truth_df) + 1)\n",
    "\n",
    "print(\"Truth_df shape :\", truth_df.shape)\n",
    "truth_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10.3 Merge test_last_df with truth_df\n",
    "This aligns each engine’s last test cycle with its true RUL."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:46:19.975911Z",
     "iopub.status.busy": "2025-12-02T16:46:19.975621Z",
     "iopub.status.idle": "2025-12-02T16:46:19.988766Z",
     "shell.execute_reply": "2025-12-02T16:46:19.987943Z",
     "shell.execute_reply.started": "2025-12-02T16:46:19.975891Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape after merge: (100, 27)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>cycle</th>\n",
       "      <th>RUL_true</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>31</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>49</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>126</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>106</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>98</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  cycle  RUL_true\n",
       "0   1     31         1\n",
       "1   2     49         2\n",
       "2   3    126         3\n",
       "3   4    106         4\n",
       "4   5     98         5"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 10.3 Merge last cycle test rows with truth\n",
    "test_merged = test_last_df.merge(truth_df, on='id', how='inner')\n",
    "\n",
    "print(\"Shape after merge:\", test_merged.shape)\n",
    "test_merged[['id', 'cycle', 'RUL_true']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10.4 Predict RUL for the last cycle of each engine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-02T16:46:28.447296Z",
     "iopub.status.busy": "2025-12-02T16:46:28.446918Z",
     "iopub.status.idle": "2025-12-02T16:46:28.514709Z",
     "shell.execute_reply": "2025-12-02T16:46:28.513857Z",
     "shell.execute_reply.started": "2025-12-02T16:46:28.447271Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Final Evaluation on Test Set ===\n",
      "RMSE: 77.29\n",
      "MAE : 64.89\n"
     ]
    }
   ],
   "source": [
    "# 10.4 Predict RUL and compute metrics\n",
    "\n",
    "X_test_last = test_merged[feature_cols]\n",
    "y_test_true = test_merged['RUL_true'].astype(float)  # <- ensure numeric\n",
    "\n",
    "y_test_pred = best_rf.predict(X_test_last)\n",
    "\n",
    "rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))\n",
    "mae_test = mean_absolute_error(y_test_true, y_test_pred)\n",
    "\n",
    "print(\"=== Final Evaluation on Test Set ===\")\n",
    "print(f\"RMSE: {rmse_test:.2f}\")\n",
    "print(f\"MAE : {mae_test:.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 11. Conclusion\n",
    "\n",
    "- The predictive maintenance problem was modeled as a **regression** task with RUL (Remaining Useful Life) as the target.  \n",
    "- The RUL variable was engineered from run-to-failure training data.  \n",
    "- Multiple regression models were trained and compared:  \n",
    "  - Linear Regression  \n",
    "  - Random Forest Regressor  \n",
    "  - Gradient Boosting Regressor  \n",
    "  - Extra Trees Regressor  \n",
    "- A `GridSearchCV` optimization was performed on the Random Forest model.  \n",
    "- The best model was retrained on the full training dataset and evaluated on `PM_test` using `PM_truth`.  \n",
    "- The final RMSE/MAE represent the model’s predictive performance on real engine degradation scenarios.\n",
    "\n",
    "**Potential improvements:**\n",
    "- Use sequential models (GRU, LSTM) to fully capture time dependencies.  \n",
    "- Add engineered temporal features (rolling means, deltas, exponential smoothing).  \n",
    "- Test other ensemble models such as XGBoost or LightGBM.  \n",
    "- Apply more advanced hyperparameter optimization (Random Search, Bayesian Optimization).\n"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 5101272,
     "sourceId": 8539436,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31192,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
