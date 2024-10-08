from paltas.Analysis import hierarchical_inference
from paltas.Analysis.hierarchical_inference import gaussian_product_analytical
from Inference.base_hierarchical_inference import BaseHierarchicalInference
import numpy as np
import emcee
import h5py

# try object orientated hierarchical inference code
class NetworkHierarchicalInference(BaseHierarchicalInference):
    """
    Class to drive hierarchical inference using NPEs as input
    Assumes parameter ordering: theta_E, gamma1, gamma2, gamma_lens, e1, e2

    Args:
        y_train ([float]): means of training prior
        std_train ([float]): std devs of training prior
        y_pred_list (array[n_test_sets,n_lenses,n_params]): list of network 
            predicted Gaussian means
        prec_pred_list (array[n_test_sets,n_lenses,n_params,n_params]): list of
            network Gaussian predicted precision matrices (inverse of 
            covariance matrices)
        hypermodel_type: 'regular','fixed_param','fixed_param_no_coords' supported
        method: 'analytical' supported. ('sampling' version coming soon...)
        sigmas_log_uniform (bool): If True, standard deviations in hypermodel
            are sampled uniformly in log-space rather than uniformly in real 
            space
        n_emcee_samps (int): number samples each emcee walker goes for 
            (includes burn in, to be taken off later by user)

    """
    def __init__(self,y_train,std_train,y_pred_list,prec_pred_list,
                 hypermodel_type='regular',method='analytical',
                 sigmas_log_uniform=False,n_emcee_samps=int(1e4)):
        
        super().__init__(hypermodel_type,sigmas_log_uniform,n_emcee_samps)

        # The interim training distribution.
        self.mu_omega_i = np.array(y_train).astype(np.float32)
        self.std_omega_i = np.array(std_train)
        self.cov_omega_i =np.diag(np.array(std_train)**2)
        self.prec_omega_i = np.linalg.inv(self.cov_omega_i).astype(np.float32)

        # neural posterior estimates (NPEs)
        # store list of y_pred, prec_pred (can do HI over multiple samples)
        self.y_pred_list = y_pred_list
        self.prec_pred_list = prec_pred_list
        
        if method != 'analytical':
            raise ValueError('only analytical method is currently supported')
        
        self.paltas_prob_class = None
    
    def log_likelihood_fixed_params(self,hyperparameters):

        # p(Omega) Gaussian from hyperparameter proposal
        mus_omega = np.asarray([hyperparameters[0],0.,0.,
                                hyperparameters[1],0.,0.]).astype(np.float32)
        sigmas = np.asarray([hyperparameters[2],hyperparameters[3],
                            hyperparameters[3],hyperparameters[4],
                            hyperparameters[5],hyperparameters[5]]).astype(np.float32)
        if self.hypermodel_type == 'fixed_param':
            mus_omega = np.append(mus_omega,[0.,0.,0.,0.])
            sigmas = np.append(sigmas,[hyperparameters[6],hyperparameters[6],
                            hyperparameters[7],hyperparameters[7]])
            
        # this inverse could be a possible issue
        prec_omega = np.linalg.inv(np.diag(sigmas**2))

        # product over lenses = sum in log space
        # TODO: use this function from paltas: log_integral_product
        result = 0
        for p in range(0,np.shape(self.y_pred_npe)[0]):
            # self.y_pred_hi, self.prec_pred_hi contains current
            # part of y_pred_list

            prec_comb = self.prec_pred_npe[p,:,:]+prec_omega-self.prec_omega_i

            # TODO: bring this back (switch to float32)
            # this check is in the paltas function, printing here for debugging
            if not np.array_equal(prec_comb, prec_comb.T):
                print("failing tranpose test, returning -inf")

            result += gaussian_product_analytical(self.y_pred_npe[p,:],
                self.prec_pred_npe[p,:,:],self.mu_omega_i,self.prec_omega_i,
                mus_omega,prec_omega)
            
        return result
    
    def log_posterior_regular(self,hyperparameters):
        
        lp = self.uniform_log_prior(hyperparameters)

        if np.isfinite(lp):

            mus_omega = np.asarray(hyperparameters[:6])
            sigmas = np.asarray(hyperparameters[6:])
            prec_omega = np.linalg.inv(np.diag(sigmas**2))

            lp += self.paltas_prob_class.log_integral_product(
                self.y_pred_npe,self.prec_pred_npe,
                self.mu_omega_i,self.prec_omega_i,
                mus_omega,prec_omega)

        return lp

    def log_posterior_fixed_param(self,hyperparameters):
        
        lp = self.uniform_log_prior_fixed_param(hyperparameters)
        
        if np.isfinite(lp):
            lp += self.log_likelihood_fixed_params(hyperparameters)

        return lp
    
    def run_HI(self,chains_filepath=None):
        """
        Runs HI over all y_pred_list,prec_pred_list and saves results in a 
        .h5 file

        Args:
            chains_filepath (string): path to .h5 file to save chains. If None,
                does not save to file. 

        Returns:
            list[emcee chains]
        """

        # set up paltas prob class if needed
        if self.hypermodel_type == 'regular':
            self.paltas_prob_class = hierarchical_inference.ProbabilityClassAnalytical(
                self.mu_omega_i,self.cov_omega_i,self.uniform_log_prior)

        if chains_filepath is not None:
            h5f = h5py.File(chains_filepath, 'w')

        chains_list = []
        for i in range(len(self.y_pred_list)):
            # set current predictions from network
            self.y_pred_npe = np.ascontiguousarray(
                np.asarray(self.y_pred_list[i])).astype(np.float32)
            self.prec_pred_npe = np.ascontiguousarray(
                np.asarray(self.prec_pred_list[i])).astype(np.float32)
            
            # generate a fresh current state
            cur_state = self.generate_initial_state()

            # generate a fresh sampler object
            if self.hypermodel_type == 'regular':
                sampler = emcee.EnsembleSampler(self.n_walkers,
                    self.ndim,self.log_posterior_regular)
            elif self.hypermodel_type in ['fixed_param','fixed_param_no_coords']:
                sampler = emcee.EnsembleSampler(self.n_walkers,
                    self.ndim,self.log_posterior_fixed_param)

            # run mcmc
            _ = sampler.run_mcmc(cur_state,self.n_emcee_samps,progress=True,
                skip_initial_state_check=True)
            
            # save chain to .h5 file
            if chains_filepath is not None:
                h5f.create_dataset('chain_%d'%(i), data=sampler.chain)

            # append to list
            chains_list.append(sampler.chain)

        if chains_filepath is not None:
            self.last_h5_saved = chains_filepath
            h5f.close()

        return chains_list