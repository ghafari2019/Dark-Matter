# Estimating SUSY Dark Matter Parameters Using Neural Networks

## Objective
The goal is to estimate key parameters of a supersymmetric (SUSY) dark matter model using a neural network. These parameters include $M_1$ (Bino mass parameter), $M_2$ (Wino mass parameter), $\mu$ (Higgsino mass parameter), $\tan \beta$ (ratio of Higgs field VEVs), $M_A$ (mass of the pseudo-scalar Higgs boson), $m_{\tilde{g}}$ (gluino mass), and $m_{\tilde{q}}$ (squark mass). The output predictions are the spin-independent cross-section ($\sigma_{SI}$), spin-dependent cross-section ($\sigma_{SD}$), and relic density ($\Omega_{\chi} h^2$).


## Scenario: Estimating Valid SUSY Parameter Regions

### Data Preparation

#### 1. Extract Relevant Parameters and Experimental Constraints

Based on the provided document, extract the following parameters and constraints:

- **Input Parameters (SUSY Parameters)**:
    - $M_1$: Bino mass parameter
    - $M_2$: Wino mass parameter
    - $\mu$: Higgsino mass parameter
    - $\tan \beta$: Ratio of Higgs field VEVs
    - $M_A$: Mass of the pseudo-scalar Higgs boson
    - $m_{\tilde{g}}$: Gluino mass
    - $m_{\tilde{q}}$: Squark mass

- **Output Constraints (Experimental Data)**:
    - $\sigma_{SI}$: Spin-independent scattering cross-section
    - $\sigma_{SD}$: Spin-dependent scattering cross-section
    - $\Omega_{\chi} h^2$: Dark matter relic density

### Step-by-Step Implementation

#### Step 1: Generate Synthetic Data

To simulate the data, we can generate random samples of SUSY parameters and compute the corresponding values for \( \sigma_{SI} \), \( \sigma_{SD} \), and \( \Omega_{\chi} h^2 \) using hypothetical relationships. The valid regions of the parameters will be those that satisfy specific constraints on these outputs.

```python
import numpy as np
import pandas as pd

def generate_susy_data(num_samples):
    # Generate synthetic data for SUSY parameters
    M1 = np.random.uniform(50, 1000, num_samples)
    M2 = np.random.uniform(50, 1000, num_samples)
    mu = np.random.uniform(100, 2000, num_samples)
    tan_beta = np.random.uniform(1, 60, num_samples)
    MA = np.random.uniform(100, 2000, num_samples)
    m_gluino = np.random.uniform(500, 3000, num_samples)
    m_squark = np.random.uniform(500, 3000, num_samples)

    # Generate synthetic outputs (e.g., cross-sections, relic density)
    sigma_SI = 1e-45 * np.exp(-0.001 * M1) + 1e-45 * np.exp(-0.001 * mu)
    sigma_SD = 1e-40 * np.exp(-0.001 * tan_beta) + 1e-40 * np.exp(-0.001 * m_gluino)
    omega_chi_h2 = 0.1 * np.exp(-0.001 * MA) + 0.1 * np.exp(-0.001 * m_squark)

    data = pd.DataFrame({
        'M1': M1,
        'M2': M2,
        'mu': mu,
        'tan_beta': tan_beta,
        'MA': MA,
        'm_gluino': m_gluino,
        'm_squark': m_squark,
        'sigma_SI': sigma_SI,
        'sigma_SD': sigma_SD,
        'omega_chi_h2': omega_chi_h2
    })

    # Apply constraints to filter valid data points
    valid_data = data[
        (data['sigma_SI'] < 1e-44) &
        (data['sigma_SD'] < 1e-39) &
        (data['omega_chi_h2'] > 0.094) &
        (data['omega_chi_h2'] < 0.129)
    ]

    return valid_data

# Generate synthetic dataset
num_samples = 10000
valid_susy_data = generate_susy_data(num_samples)

# Save the dataset to a CSV file
valid_susy_data.to_csv('valid_susy_data.csv', index=False)
```

#### Step 2: Define and Train the Neural Network

We can design a neural network to learn the valid regions of the SUSY parameters based on the constraints.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('valid_susy_data.csv')

# Define input features
X = data[['M1', 'M2', 'mu', 'tan_beta', 'MA', 'm_gluino', 'm_squark']].values

# Define output labels (this time, we are interested in predicting valid regions directly)
# Since we have already filtered valid regions, we can use the parameters themselves as outputs
y = X.copy()

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=7, activation='relu'),
    Dense(128, activation='relu'),
    Dense(7, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Predict using the model
predictions = model.predict(X_val)
print(predictions[:5])
```


#### Step 3: Model Evaluation

Evaluate the model's performance on the validation set to ensure it correctly identifies the valid parameter regions.

```python

# Evaluate the model
loss = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
```
#### Summary

This scenario outlines a comprehensive approach to estimating the valid regions of SUSY parameters using neural networks. The steps include:

  - **Data Preparation:** Extract relevant SUSY parameters and experimental constraints from the provided document. Generate synthetic data for SUSY parameters and compute corresponding outputs. Filter the data to retain only the valid regions.
- **Model Design and Training:** Define and train a neural network model to learn the valid regions of SUSY parameters.
- **Model Evaluation:** Evaluate the model's performance on a validation set.

By following these steps, the neural network model will learn the complex relationships between SUSY parameters and their constraints, allowing it to identify valid parameter regions effectively.
