Let
$$N_{neutron}\sim Poisson(\lambda_{neutron}).$$
$$D_{neutron}|N_{nuetron}\sim Binom(N_{neutron},\eta_{neutron})$$

where
* $N_{neutron}$ is the number of incident neutrons
* $D_{neuton}$ is the number of neutron interactions
* $\eta_{neutron}$ is the quantum efficiency of the scintillator

Futhermore, for a single nuetron interatction suppose the number of emitted photons have the following distribution
$$K_{photon}\sim Poisson({\lambda_0})$$

thus, the total number of emitted photons has the following distribution
$$N_{photon}|D_{neutron} = \sum_{i=1}^{D_{neutron}}K_i\sim Poisson(D_n\lambda_0)$$

It follows that 
$$D_{photons}|N_{photons}\sim Binom(N_{photons}, \eta_{photons})$$

where
* $D_{photons}$ is the total number of photon interactions
* $\eta_{photons}$ si the quantum efficiency of the array (?)

1. Critique this model
2. Determine the marginal distributions of $D_{neutron}, N_{photon}$ and $D_{photon}$. Futhermore, find their expected value and variance
3. Determine the SNR of $D_{neutron}$ and $D_{photon}$