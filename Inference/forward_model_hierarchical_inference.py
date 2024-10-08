import numpy as np
from Inference.base_hierarchical_inference import BaseHierarchicalInference
from scipy.stats import multivariate_normal, norm
import pickle
from lenstronomy.Sampling.parameters import Param
import numba
import emcee
import h5py
import copy

class ForwardModelHierarchicalInference(BaseHierarchicalInference):
    """
    Class to drive hierarchical inference using mcmc_samples & kwargs files 
        from Thomas Schmidt STRIDES30 modeling (cite)

    Note: Assumes parameter ordering: theta_E, gamma1, gamma2, gamma_lens,
        e1, e2
    
    Args:
        kwargs_folder (string): path to kwargs.txt files
        samples_folder (string): path to mcmc_samples.txt files
        doppel_names ([string]): list of lens names, used as prefix to files
        training_widths ([float]): list of standard devs. for training distribution
            (needed to make sure same prior is enforced)
        hypermodel_type: 'regular','fixed_param' supported
        sigmas_log_uniform (bool): If True, standard deviations in hypermodel
            are sampled uniformly in log-space rather than uniformly in real 
            space
        n_emcee_samps (int): number samples each emcee walker goes for 
            (includes burn in, to be taken off later by user)
        """

    def __init__(self,kwargs_folder,samples_folder,doppel_names,training_widths,
        hypermodel_type='regular',sigmas_log_uniform=False,
        n_emcee_samps=int(1e4)):

        super().__init__(hypermodel_type,sigmas_log_uniform,n_emcee_samps)

        self.kwargs_file_list = [(kwargs_folder + d + '_kwargs.txt') 
                                 for d in doppel_names]
        self.samples_file_list = [(samples_folder + d + '_5000_mcmc_samples.txt') 
                                 for d in doppel_names]
        self.chains_list = None

        # setup place to save gaussianized posteriors
        self.means_list = None
        self.cov_list = None

        # set upper limit of widths of distribution to match network HI
        self.std_omega_i = training_widths
        
    def deflector_params_chains_list(self):
        """Based on code from Thomas Schmidt @ UCLA, received 8/1/2023. 
            Provided mcmc chains need some extra calculations to get 'fixed' 
            parameters (theta_E for example) from free parameters

        Note: 
            Assumes parameter order: theta_E, gamma1, gamma2, gamma_lens, e1, e2
        Returns:
            list of chains w/ samples in order specified above.
                Shape: (n_lenses,n_samps,n_params)
        """

        chains_list = []

        for i in range(0,len(self.kwargs_file_list)):
            current_chain = []

            # retrieve provided samples & kwargs from Thomas
            f_samps = open(self.samples_file_list[i],'rb')
            [param_mcmc, samples_mcmc] = pickle.load(f_samps)
            f_samps.close()
            f_kwargs = open(self.kwargs_file_list[i],'rb')
            [kwargs_result, kwargs_model,kwargs_params, kwargs_constraints] = pickle.load(f_kwargs)
            f_kwargs.close()
    
            # construct lenstronomy Param object
            param = Param(kwargs_model, kwargs_params['lens_model'][2],
                kwargs_params['source_model'][2],
                kwargs_params['lens_light_model'][2],
                kwargs_params['point_source_model'][2], 
                kwargs_lens_init=kwargs_result['kwargs_lens'],
                **kwargs_constraints)
                
            # construct current_chain by looping through indvidiual samples
            for mcmc in samples_mcmc:
                # convert mcmc samps of free params to full list of 
                # lens parameters (fixed & free)
                kwargs_result_new = param.args2kwargs(mcmc)
                # parameter order: theta_E, gamma1, gamma2, gamma_lens, e1, e2
                current_samp = [
                    kwargs_result_new['kwargs_lens'][0]['theta_E'],
                    kwargs_result_new['kwargs_lens'][1]['gamma1'],
                    kwargs_result_new['kwargs_lens'][1]['gamma2'],
                    kwargs_result_new['kwargs_lens'][0]['gamma'],
                    kwargs_result_new['kwargs_lens'][0]['e1'],
                    kwargs_result_new['kwargs_lens'][0]['e2']
                ]

                current_chain.append(np.asarray(current_samp))

            chains_list.append(np.asarray(current_chain))

        # should have shape (n_lenses,n_samps,n_params)
        self.chains_list = chains_list
        return chains_list
    
    def gaussianize_chains_list(self,chains_list=None):
        """
        Convert list of mcmc chains to a list of means & covariance matrices
            that approximate the shape of the chain posterior

        Args:
            chains_list: Shape (n_lenses, n_samples, n_params)

        Returns:
            means_list: Shape (n_lenses, n_params)
            cov_list: Shape (n_lenses, n_params, n_params)
        """
        if chains_list is None:
            if self.chains_list is not None:
                chains_list = self.chains_list
            else:
                raise ValueError('No chain stored or passed as argument')

        means_list = []
        cov_list = []
        for c in chains_list:
            means = np.mean(c,axis=0)
            # columns are variables
            cov = np.cov(c,rowvar=False)
            means_list.append(means)
            cov_list.append(cov)

        self.means_list = means_list
        self.cov_list = cov_list

        return means_list, cov_list
    
    # overwrite base version so only 6 params are included!
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
        
        # penalize too narrow
        for i in range(2,len(hyperparameters)):
            if hyperparameters[i] < 0.001:
                return -np.inf
        
        if self.sigmas_log_uniform:
            return -np.sum(np.log(hyperparameters[2:]))
        # sigmas are uniform in regular space
        return 0

    @staticmethod
    #@numba.njit
    def two_gaussian_product_analytical(mu_pred,prec_pred,mu_omega,prec_omega):
        # copied & modified from paltas code, written by Sebastian Wagner-Carena
        # https://github.com/swagnercarena/paltas/blob/2edd7f418a63273d5b2fcc75819e811bceb1f149/paltas/Analysis/hierarchical_inference.py#L85C10-L85C10
        """ Calculate the log of the integral of p(xi_k|omega)*p(xi_k|d_k) 
            when both pdfs are Gaussian.

        Args:
            mu_pred (np.array): The mean output by the network
            prec_pred (np.array): The precision matrix output by the network
            mu_omega (np.array): The mean of the proposed hyperparameter
                posterior.
            prec_omega (np.array): The precision matrix of the proposed
                hyperparameter posterior.

        Returns:
            (float): The lof of the product of the two Gaussians integrated over
            all space.
        Notes:
            The equation used here breaks down when the combination of precision
            matrices does not yield a valid precision matrix. When this happen, the
            output will be -np.inf.
        """
        # This implements the final formula derived in the appendix of
        # Wagner-Carena et al. 2021.

        # Calculate the values of eta and the combined precision matrix
        prec_comb = prec_pred+prec_omega

        if not np.array_equal(prec_pred,prec_pred.T):
            print('not symmetric')
            return -np.inf

        # prec_comb is not guaranteed to be a valid precision matrix.
        # When it isn't, the analytical equation used here is wrong.
        # In those cases, return -np.inf.
        # To check the matrix is positive definite, we check that is symmetric
        # and that its Cholesky decomposition exists
        # (see https://stackoverflow.com/questions/16266720)
        if not np.array_equal(prec_comb, prec_comb.T):
            return -np.inf
        try:
            np.linalg.cholesky(prec_comb)
        except Exception:  # LinAlgError, but numba can't match exceptions
            return -np.inf

        cov_comb = np.linalg.inv(prec_comb)
        # eta is prec_matrix*mu
        eta_pred = np.dot(prec_pred,mu_pred)
        eta_omega = np.dot(prec_omega,mu_omega)
        eta_comb = eta_pred + eta_omega

        # Now calculate each of the terms in our exponent
        exponent = 0
        exponent -= np.log(abs(np.linalg.det(prec_pred)))
        exponent -= np.log(abs(np.linalg.det(prec_omega)))
        exponent += np.log(abs(np.linalg.det(prec_comb)))
        exponent += np.dot(mu_pred.T,np.dot(prec_pred,mu_pred))
        exponent += np.dot(mu_omega.T,np.dot(prec_omega,mu_omega))
        exponent -= np.dot(eta_comb.T,np.dot(cov_comb,eta_comb))

        return -0.5*exponent
    
    @staticmethod
    #@numba.njit
    def gaussian_product_analytical_gamma_lens(mu_pred,prec_pred,mu_gamma_int,
        std_gamma_int,mu_omega,prec_omega):  # pragma: no cover
        # copied & modified from paltas code, written by Sebastian Wagner-Carena
        # https://github.com/swagnercarena/paltas/blob/2edd7f418a63273d5b2fcc75819e811bceb1f149/paltas/Analysis/hierarchical_inference.py#L85C10-L85C10
        """ Calculate the log of the integral of p(xi_k|omega)*p(xi_k|d_k,omega_int)/
        p(xi_k|omega_int) when omega_int is only informative in gamma_lens

        Args:
            mu_pred (np.array): The mean output by the network
            prec_pred (np.array): The precision matrix output by the network
            mu_gamma_int (float): The mean of gamma_lens interim prior
            std_gamma_int (float): The std. dev. of gamma_lens interim prior
            mu_omega (np.array): The mean of the proposed hyperparameter
                posterior.
            prec_omega (np.array): The precision matrix of the proposed
                hyperparameter posterior.

        Returns:
            (float): The lof of the product of the three Gaussian integrated over
            all space.
        Notes:
            The equation used here breaks down when the combination of precision
            matrices does not yield a valid precision matrix. When this happen, the
            output will be -np.inf.
        """
        # This implements the final formula derived in the appendix of
        # Wagner-Carena et al. 2021.

        # Calculate the values of eta and the combined precision matrix
        num_params = len(mu_omega)
        prec_omega_i = np.zeros(shape=(num_params,num_params),dtype=np.float32)
        prec_omega_i[3,3] = 1/(std_gamma_int**2)
        prec_comb = prec_pred+prec_omega-prec_omega_i

        # prec_comb is not guaranteed to be a valid precision matrix.
        # When it isn't, the analytical equation used here is wrong.
        # In those cases, return -np.inf.
        # To check the matrix is positive definite, we check that is symmetric
        if not np.array_equal(prec_comb, prec_comb.T):
            return -np.inf
        
        #mu_omega_i only matters in gamma_lens dimension, b/c zeros in precision
        # matrix for all other dimensions
        mu_omega_i = np.zeros(shape=(num_params),dtype=np.float32)
        mu_omega_i[3] = mu_gamma_int

        cov_comb = np.linalg.inv(prec_comb)
        eta_pred = np.dot(prec_pred,mu_pred)
        eta_omega_i = np.dot(prec_omega_i,mu_omega_i)
        eta_omega = np.dot(prec_omega,mu_omega)
        eta_comb = eta_pred + eta_omega - eta_omega_i

        # Now calculate each of the terms in our exponent
        exponent = 0
        exponent -= np.log(abs(np.linalg.det(prec_pred)))
        exponent -= np.log(abs(np.linalg.det(prec_omega)))
        exponent += np.log(abs(np.linalg.det(prec_omega_i)))
        exponent += np.log(abs(np.linalg.det(prec_comb)))
        exponent += np.dot(mu_pred.T,np.dot(prec_pred,mu_pred))
        exponent += np.dot(mu_omega.T,np.dot(prec_omega,mu_omega))
        exponent -= np.dot(mu_omega_i.T,np.dot(prec_omega_i,mu_omega_i))
        exponent -= np.dot(eta_comb.T,np.dot(cov_comb,eta_comb))

        return -0.5*exponent
    

    @staticmethod
    #@numba.njit
    def gaussian_product_sampling_gamma_lens(npe_samps,mu_gamma_int,
        std_gamma_int,mu_omega,prec_omega):  # pragma: no cover
        # copied & modified from paltas code, written by Sebastian Wagner-Carena
        # https://github.com/swagnercarena/paltas/blob/2edd7f418a63273d5b2fcc75819e811bceb1f149/paltas/Analysis/hierarchical_inference.py#L85C10-L85C10
        """ sampling version of the integral of p(xi_k|omega)*p(xi_k|d_k,omega_int)/
        p(xi_k|omega_int) when omega_int is only informative in gamma_lens

        Args:
            npe_samps (np.array): samples from the interim posterior
            mu_gamma_int (float): The mean of gamma_lens interim prior
            std_gamma_int (float): The std. dev. of gamma_lens interim prior
            mu_omega (np.array): The mean of the proposed hyperparameter
                posterior.
            prec_omega (np.array): The precision matrix of the proposed
                hyperparameter posterior.

        Returns:
            (float): The lof of the product of the three Gaussian integrated over
            all space.
        Notes:
            The equation used here breaks down when the combination of precision
            matrices does not yield a valid precision matrix. When this happen, the
            output will be -np.inf.
        """
        # This implements the final formula derived in the appendix of
        # Wagner-Carena et al. 2021.

        # time to do this old school:
        gamma_samps = npe_samps[:,3]

        # we know we're dividing two Gaussians. So, we compute the combined exponent,
        # only exponentiate once
        std_omega = np.diag(np.sqrt(np.linalg.inv(prec_omega)))
        exponent = -0.5 * ( np.sum(((npe_samps - mu_omega)/std_omega)**2,axis=1)
            - ((gamma_samps - mu_gamma_int)/std_gamma_int)**2)
        to_multiply = std_gamma_int / np.prod(std_omega)

        return np.log((to_multiply/npe_samps.shape[0]) * np.sum(np.exp(exponent)))

        #evaluated_prior = multivariate_normal(mean=mu_omega,cov=np.linalg.inv(prec_omega)).pdf(npe_samps)
        # TODO: make sure I understand shape of npe_samps correctly
        #evaluated_omega_int = norm(loc=mu_gamma_int,scale=std_gamma_int).pdf(npe_samps[:,3])
        #print('shape 1: ', evaluated_prior.shape)
        #print('shape 2: ', evaluated_omega_int.shape)
        #sum = (1/1000) * np.sum(evaluated_prior/evaluated_omega_int)

        # TODO: move to log space to avoid overflow?
        #return np.log(sum)
    
    def log_likelihood_regular(self,hyperparameters):

        # p(Omega) Gaussian from hyperparameter proposal
        mus_omega = np.asarray(hyperparameters[:6])
        sigmas = np.asarray(hyperparameters[6:])
        prec_omega = np.linalg.inv(np.diag(sigmas**2))

        integral = 0
        for pi in range(self.y_pred_fm.shape[0]):
            integral += ForwardModelHierarchicalInference.two_gaussian_product_analytical(self.y_pred_fm[pi,:],
                self.prec_pred_fm[pi,:,:],mus_omega,prec_omega)
        # Treat nan as probability 0.
        if np.isnan(integral):
            integral = -np.inf

        return integral

    def log_likelihood_fixed(self,hyperparameters):

        mus_omega = np.asarray([hyperparameters[0],0.,0.,
                                hyperparameters[1],0.,0.])
        sigmas = np.asarray([hyperparameters[2],hyperparameters[3],
                            hyperparameters[3],hyperparameters[4],
                            hyperparameters[5],hyperparameters[5]])
        # this inverse could be a possible issue
        prec_omega = np.linalg.inv(np.diag(sigmas**2))

        integral = 0
        for pi in range(len(self.y_pred_fm)):
            integral += ForwardModelHierarchicalInference.two_gaussian_product_analytical(self.y_pred_fm[pi],
                self.prec_pred_fm[pi],mus_omega,prec_omega)
        # Treat nan as probability 0.
        if np.isnan(integral):
            integral = -np.inf

        return integral
    
    def log_likelihood_fixed_gamma_lens(self,hyperparameters):
        """
        Computes log likelihood with an interim prior that is only informative
        in gamma_lens

        Returns:
            (float): Log likelihood 
        """

        mus_omega = np.asarray([hyperparameters[0],0.,0.,
                                hyperparameters[1],0.,0.]).astype(np.float32)
        sigmas = np.asarray([hyperparameters[2],hyperparameters[3],
                            hyperparameters[3],hyperparameters[4],
                            hyperparameters[5],hyperparameters[5]]).astype(np.float32)
        # this inverse could be a possible issue
        prec_omega = np.linalg.inv(np.diag(sigmas**2))

        integral = 0
        for pi in range(len(self.y_pred_fm)):
            #integral += ForwardModelHierarchicalInference.gaussian_product_analytical_gamma_lens(
            #    self.y_pred_fm[pi,:],
            #    self.prec_pred_fm[pi,:,:],
            #    2.078,0.027, #mu_gamma_int, sigma_gamma_int
            #    mus_omega,prec_omega)
            # or 
            integral += ForwardModelHierarchicalInference.gaussian_product_sampling_gamma_lens(
                self.NPE_samps[pi],
                2.078,0.027, #mu_gamma_int, sigma_gamma_int
                mus_omega,prec_omega)
        # Treat nan as probability 0.
        if np.isnan(integral):
            integral = -np.inf

        return integral

    def log_posterior_regular(self,hyperparameters):
        
        lp = self.uniform_log_prior(hyperparameters)

        if np.isfinite(lp):
            lp += self.log_likelihood_regular(hyperparameters)

        return lp

    def log_posterior_fixed(self,hyperparameters):

        lp = self.uniform_log_prior_fixed_param(hyperparameters)

        if np.isfinite(lp):
            lp += self.log_likelihood_fixed(hyperparameters)

        return lp
    
    def log_posterior_fixed_gamma_lens(self,hyperparameters):

        lp = self.uniform_log_prior_fixed_param(hyperparameters)

        if np.isfinite(lp):
            lp += self.log_likelihood_fixed_gamma_lens(hyperparameters)

        return lp
        

    def run_HI(self,chains_filepath=None):
        """
        Runs HI and saves results in a .h5 file

        Args:
            chains_filepath (string): path to .h5 file to save chains. If None,
                does not save to file. 
        Returns:
            array[float]: emcee chain
        """

        # retrieve FM chains
        if self.chains_list is not None:
            _ = self.deflector_params_chains_list()
        fm_chains_list = self.chains_list
        # gaussianize chains
        if self.means_list is None:
            _,_ = self.gaussianize_chains_list(fm_chains_list)
        y_pred_fm = np.asarray(copy.deepcopy(self.means_list))
        prec_pred_fm = np.linalg.inv(np.asarray(copy.deepcopy(self.cov_list)))

        # do a HI w/out the interim training prior bit!
                # create sampler object
        if self.hypermodel_type == 'regular':
            sampler = emcee.EnsembleSampler(self.n_walkers,
                self.ndim,self.log_posterior_regular)

        elif self.hypermodel_type == 'fixed_param':
            sampler = emcee.EnsembleSampler(self.n_walkers,
                self.ndim,self.log_posterior_fixed)
            
        # this is the option that includes weighting out the informative prior
        elif self.hypermodel_type == 'fixed_param_no_coords':
            sampler = emcee.EnsembleSampler(self.n_walkers,
                self.ndim,self.log_posterior_fixed_gamma_lens)
            
        if chains_filepath is not None:
            h5f = h5py.File(chains_filepath, 'w')

        # set current predictions
        # change to float 32, float 64 had some rounding errors on precision
        # matrices not being symmetric
        self.y_pred_fm = np.ascontiguousarray(
            np.asarray(y_pred_fm)).astype(np.float32)
        self.prec_pred_fm = np.ascontiguousarray(
            np.asarray(prec_pred_fm)).astype(np.float32)
        
        if self.hypermodel_type == 'fixed_param_no_coords':
            self.N_imp_samps = 500
            NPE_samps = np.empty((self.y_pred_fm.shape[0],self.N_imp_samps,self.y_pred_fm.shape[-1]))
            for j in range(0,self.y_pred_fm.shape[0]):
                NPE_samps[j] = multivariate_normal(mean=self.y_pred_fm[j],
                    cov=np.linalg.inv(self.prec_pred_fm[j])).rvs(self.N_imp_samps)

            self.NPE_samps = np.ascontiguousarray(NPE_samps).astype(np.float32)

        # generate a fresh current state
        # HARDCODED: using 8 param initial state and truncating (b/c i'm lazy)
        # generate a fresh current state
        cur_state = self.generate_initial_state()
        #init_state = copy.deepcopy(self.generate_initial_state()[:,:6])

        # run mcmc
        _ = sampler.run_mcmc(cur_state,self.n_emcee_samps,progress=True,
            skip_initial_state_check=True)
        
        print('mean acceptance fraction: ',np.mean(sampler.acceptance_fraction))
        
        # save chain to .h5 file
        if chains_filepath is not None:
            h5f.create_dataset('chain_fm', data=sampler.chain)

        if chains_filepath is not None:
            self.last_h5_saved = chains_filepath
            h5f.close()

        return sampler.chain

         