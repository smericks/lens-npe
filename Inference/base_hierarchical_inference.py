import numpy as np
import h5py

class BaseHierarchicalInference():
    """
    Store functions common across network HI and forward model HI here
    (i.e. same prior, same generation of initial state, same defaults)
    """

    def __init__(self,hypermodel_type='regular',sigmas_log_uniform=False,
        n_emcee_samps=int(1e4)):

        self.sigmas_log_uniform = sigmas_log_uniform

        # which hypermodel to use. 'regular' or 'fixed_params'
        if hypermodel_type == 'regular':
            self.hypermodel_type = 'regular'
            self.ndim = 12
        elif hypermodel_type == 'fixed_param':
            self.hypermodel_type = 'fixed_param'
            self.ndim = 8
        elif hypermodel_type == 'fixed_param_no_coords':
            self.hypermodel_type = 'fixed_param_no_coords'
            self.ndim = 6
        else:
            raise ValueError('hypermodel_type not supported')

        # some defaults for emcee
        self.n_walkers = 40
        self.n_emcee_samps = int(n_emcee_samps)

        self.last_h5_saved = None

    def uniform_log_prior(self,hyperparameters):
        # hyperparameters come in order: 
        # (mu_theta_E, mu_gamma1, mu_gamma2, mu_gamma_lens, mu_e1, mu_e2
        # sigma_theta_E, sigma_gamma1, sigma_gamma2, sigma_gamma_lens,
        #  sigma_e1, sigma_e2)

        # mu_theta_E
        if np.abs(1-hyperparameters[0]) > 0.4998:
            return -np.inf
        # mu_gamma1, mu_gamma2
        if (np.abs(hyperparameters[1]) > 0.4 or 
                np.abs(hyperparameters[2]) > 0.4):
            return -np.inf
        # mu_gamma_lens
        if hyperparameters[3] < 1 or hyperparameters[3] > 3:
            return -np.inf
        # mu_gamma1, mu_gamma2
        if (np.abs(hyperparameters[4]) > 0.6 or 
                np.abs(hyperparameters[5]) > 0.6):
            return -np.inf
        
        # sigmas strictly narrower than training prior
        # need to set bounds to avoid random singular matrix proposals
        for i in range(0,len(hyperparameters)-6):
            if hyperparameters[6+i] > self.std_omega_i[i]:
                return -np.inf
            # also penalize too narrow
            elif hyperparameters[6+i] < 0.001:
                return -np.inf
            
        # sigmas are uniform in log space
        if self.sigmas_log_uniform:
            return -np.sum(np.log(hyperparameters[6:]))
        
        return 0


    def uniform_log_prior_fixed_param(self,hyperparameters):
        # hyperparameters come in order: 
        # (mu_theta_E, mu_gamma_lens, sigma_theta_E, 
        # sigma_gamma12, sigma_gamma_lens, sigma_e12)

        # mu_theta_E
        if np.abs(1-hyperparameters[0]) > 0.4998:
            return -np.inf
        # mu_gamma_lens
        if hyperparameters[1] < 1 or hyperparameters[1] > 3:
            return -np.inf
        
        # sigmas strictly narrower than training prior
        # need to set bounds to avoid random singular matrix proposals
        # sigma_theta_E
        if hyperparameters[2] > self.std_omega_i[0]:
            return -np.inf
        # sigma_gamma12
        if (hyperparameters[3] > self.std_omega_i[1] or 
                hyperparameters[3] > self.std_omega_i[2]):
            return -np.inf
        # sigma_gamma
        if hyperparameters[4] > self.std_omega_i[3]:
            return -np.inf
        # sigma_e12
        if (hyperparameters[5] > self.std_omega_i[4] or 
                hyperparameters[5] > self.std_omega_i[5]):
            return -np.inf
        # include sigma_xy_lens, sigma_xy_src if requested
        if self.hypermodel_type == 'fixed_param':
            if (hyperparameters[6] > self.std_omega_i[6] or 
                    hyperparameters[6] > self.std_omega_i[7]):
                return -np.inf
            # sigma_xy_src
            if (hyperparameters[7] > self.std_omega_i[8] or 
                    hyperparameters[7] > self.std_omega_i[9]):
                return -np.inf
        # penalize too narrow
        for i in range(2,len(hyperparameters)):
            if hyperparameters[i] < 0.001:
                return -np.inf
        
        if self.sigmas_log_uniform:
            return -np.sum(np.log(hyperparameters[2:]))
        # sigmas are uniform in regular space
        return 0
        
    def generate_initial_state(self):

        if self.hypermodel_type == 'regular':
            # Generate an initial state informed by prior range
            cur_state_mu = np.concatenate((
                np.random.uniform(low=0.5,high=1.2,size=(self.n_walkers,1)),
                np.random.uniform(low=-0.1,high=0.1,size=(self.n_walkers,1)),
                np.random.uniform(low=-0.1,high=0.1,size=(self.n_walkers,1)),
                np.random.uniform(low=1.85,high=2.15,size=(self.n_walkers,1)),
                np.random.uniform(low=-0.15,high=0.15,size=(self.n_walkers,1)),
                np.random.uniform(low=-0.15,high=0.15,size=(self.n_walkers,1))),
                axis=1)
            if self.sigmas_log_uniform:
                low = np.log(0.01)
                cur_state_sigmas = np.exp(np.concatenate((
                    np.random.uniform(low=low,high=np.log(0.15),size=(self.n_walkers,1)),
                    np.random.uniform(low=low,high=np.log(0.12),size=(self.n_walkers,1)),
                    np.random.uniform(low=low,high=np.log(0.12),size=(self.n_walkers,1)),
                    np.random.uniform(low=low,high=np.log(0.12),size=(self.n_walkers,1)),
                    np.random.uniform(low=low,high=np.log(0.18),size=(self.n_walkers,1)),
                    np.random.uniform(low=low,high=np.log(0.18),size=(self.n_walkers,1))),
                    axis=1))
            else:
                cur_state_sigmas = np.concatenate((
                    np.random.uniform(low=0.01,high=0.15,size=(self.n_walkers,1)),
                    np.random.uniform(low=0.01,high=0.08,size=(self.n_walkers,1)),
                    np.random.uniform(low=0.01,high=0.08,size=(self.n_walkers,1)),
                    np.random.uniform(low=0.01,high=0.12,size=(self.n_walkers,1)),
                    np.random.uniform(low=0.01,high=0.18,size=(self.n_walkers,1)),
                    np.random.uniform(low=0.01,high=0.18,size=(self.n_walkers,1))),
                    axis=1)
            cur_state = np.concatenate((cur_state_mu,cur_state_sigmas),axis=1)

        elif self.hypermodel_type in ['fixed_param','fixed_param_no_coords']:
            cur_state_mu = np.concatenate((
                np.random.uniform(low=0.5,high=1.2,size=(self.n_walkers,1)),
                np.random.uniform(low=1.85,high=2.15,size=(self.n_walkers,1))),
                axis=1)
            if self.sigmas_log_uniform:
                low = np.log(0.01)
                cur_state_sigmas = np.exp(np.concatenate((
                    np.random.uniform(low=low,high=np.log(0.15),
                        size=(self.n_walkers,1)),
                    np.random.uniform(low=low,high=np.log(0.12),
                        size=(self.n_walkers,1)),
                    np.random.uniform(low=low,high=np.log(0.2),
                        size=(self.n_walkers,1)),
                    np.random.uniform(low=low,high=np.log(0.2),
                        size=(self.n_walkers,1))),axis=1))
                
                if self.hypermodel_type == 'fixed_param':
                    cur_state_sigmas = np.concatenate((cur_state_sigmas,
                        np.exp(np.random.uniform(low=low,high=np.log(0.07),
                        size=(self.n_walkers,1))),
                        np.exp(np.random.uniform(low=low,high=np.log(0.1),
                        size=(self.n_walkers,1)))),axis=1)

            else:
                cur_state_sigmas = np.concatenate((
                    np.random.uniform(low=0.01,high=0.15,size=(self.n_walkers,1)), #theta_E
                    np.random.uniform(low=0.01,high=0.12,size=(self.n_walkers,1)), #gamma1,2
                    np.random.uniform(low=0.01,high=0.2,size=(self.n_walkers,1)), #gamma
                    np.random.uniform(low=0.01,high=0.2,size=(self.n_walkers,1))),axis=1) # e1,2
                
                if self.hypermodel_type == 'fixed_param':
                    cur_state_sigmas = np.concatenate((cur_state_sigmas,
                        np.random.uniform(low=0.01,high=0.07,size=(self.n_walkers,1)),
                        np.random.uniform(low=0.01,high=0.1,size=(self.n_walkers,1))),axis=1)
                
            
            cur_state = np.concatenate((cur_state_mu,cur_state_sigmas),axis=1)

        return cur_state
    

    def retrieve_last_chains(self):
        """
        Looks for recently saved chains & returns a list of chains from a 
            .h5 file 

        Returns:
            None if no file found
            list[array[float]]: list of emcee chains
        """
        if self.last_h5_saved is None:
            print('No chains recently saved')
            return None
        
        return self.retrieve_chains_h5(self.last_h5_saved)
    
    @staticmethod
    def retrieve_chains_h5(file_path):
        """Returns a list of chains from a .h5 file
        Args:
            file_path (string)
        Returns:
            chains (list[array[float]]): list of emcee chains
        """
        h5f = h5py.File(file_path, 'r')
        chain_names = list(h5f.keys())
        chains = []
        for name in chain_names:
            chains.append(h5f.get(name).value)
        h5f.close()

        return chains