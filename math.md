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

Then
$$D_n\sim Poisson(\lambda_n\eta_n)$$
$$E(D_n) = Var(D_n) = \lambda_n\eta_n$$
$$N_p\sim NA $$
$$E(N_p) = \lambda_n\lambda_0\eta_n$$
$$Var(N_p) = \lambda_n\lambda_0\eta_n(1+\lambda_0)$$
$$E(D_p) =   \lambda_n\lambda_0\eta_n\eta_p$$
$$Var(D_p) =  \lambda_n\lambda_0\eta_n(1+\lambda_0\eta_p)$$

$$SNR(D_n) = (\lambda_n\eta_n)^{1/2}$$
$$SNR(D_p) = \left(\frac{\lambda_n\lambda_0\eta_n\eta_p}{1+\lambda_0\eta_p}\right)^{1/2}$$
