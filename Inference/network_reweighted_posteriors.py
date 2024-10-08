import numpy as np
from Inference.network_hierarchical_inference import NetworkHierarchicalInference
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import h5py

class NetworkReweightedPosteriors():
    """
    Class to compute reweighted neural posterior estimates

    Args:
        samps_weights_path (string): Path to .h5 file to store samples & weights
            for each lens
        kwargs_networkHI (dict): kwargs for NetworkHierarchicalInference object
        small_number (bool): If true, must do re-do HI for each lens k,
            excluding lens k from the sample, recovering p(Omega|d!=d_k)
    """
    def __init__(self,kwargs_networkHI,small_number=True):
        
        self.kwargs_networkHI = kwargs_networkHI
        self.small_number = small_number

    def reweighted_lens_posteriors_small_number(self,y_pred,prec_pred,
        train_mean,train_scatter,samps_weights_path=None,debug=False,
        reweight_indices=None,check_chains=False):
        """
        Loops through all lenses (length of y_pred) and computes weights for
            samples from the NPE that will re-weight to account for bias from
            the interim training prior
        
        Args:
            y_pred (array[float]): Shape:(n_lenses,n_params)
            prec_pred (array[float]) Shape:(n_lenses,n_params,n_params)
            train_mean (array[float]): Shape:(n_params)
            train_scatter (array[float]): Shape:(n_params)
            samps_weights_path (string): path to .h5 file to store samples & 
                weights for each lens. If None, does not save the info.
            debug (bool): If True, stops after trying one lens
            reweight_indices (list[int]): Which lenses to re-weight (i.e. could
                use whole list to inform HI, but only reweight a subset of lenses)
            check_chains (bool): If true, saves plots to make sure chains are
                moving around.
        """
        n_lenses = np.shape(y_pred)[0]
        n_params = np.shape(y_pred)[1]
        # TODO: need to check assumptions ab inversing
        cov_pred = np.linalg.inv(prec_pred)

        if samps_weights_path is not None:
            h5f = h5py.File(samps_weights_path, 'w')
        samples_list = []
        weights_list = []
        
        # which lenses to compute weights for
        if reweight_indices is None:
            reweight_indices = np.arange(0,np.shape(y_pred)[0])
        # current lenses included in HI
        iofi = np.arange(0,n_lenses)

        # loop over each lens you want weights fors
        for i in reweight_indices:
            # index every lens but current lens
            iofi = np.concatenate((np.arange(0,i),np.arange(i+1,n_lenses)))

            # run HI excluding lens of interest
            network_hi = NetworkHierarchicalInference(train_mean,train_scatter,
                [y_pred[iofi,:]],[prec_pred[iofi,:,:]],**self.kwargs_networkHI)
            chains_list = network_hi.run_HI()
    
            # save plot of chain if requested
            if check_chains:
                plt.figure()
                plt.plot(chains_list[0][:,:,1].T,'.')
                plt.title('$\gamma_{lens},\mu$')
                plt.savefig('HI_RW_chain_%d.png'%(i))
            burnin = int(1e3)
            if network_hi.hypermodel_type == 'fixed_param':
                n_HI_params = 8
            elif network_hi.hypermodel_type == 'regular':
                n_HI_params = 12
            chain_HI = chains_list[0][:,burnin:,:].reshape((-1,n_HI_params))
            # generate samples from multivariate Gaussian NPE
            NPE_multivariate_sampler = multivariate_normal(mean=y_pred[i,:],cov=cov_pred[i,:,:])
            NPE_samples = NPE_multivariate_sampler.rvs(size=int(5e3))
            # calculate weights using chain from sampler
            weights = np.empty(np.shape(NPE_samples)[0])
            # loop through xi_k samples & calculate a weight for each one
            for k in range(0,np.shape(NPE_samples)[0]):
                # TODO: check that array dimensions are in order: batch_size, num_params
                xi_k = NPE_samples[k,:]

                # construct Gaussian mus, sigmas from HI chain
                # chain_HI has shape: [num_samps,num_params]
                if network_hi.hypermodel_type == 'fixed_param':
                    mus = np.vstack((chain_HI[:,0],np.zeros(np.shape(chain_HI[:,0])),
                                    np.zeros(np.shape(chain_HI[:,0])),chain_HI[:,1],
                                    np.zeros(np.shape(chain_HI[:,0])),np.zeros(np.shape(chain_HI[:,0])),
                                    np.zeros(np.shape(chain_HI[:,0])),np.zeros(np.shape(chain_HI[:,0])),
                                    np.zeros(np.shape(chain_HI[:,0])),np.zeros(np.shape(chain_HI[:,0])))).T
                    sigmas = np.vstack((chain_HI[:,2],chain_HI[:,3],chain_HI[:,3],
                                        chain_HI[:,4],chain_HI[:,5],chain_HI[:,5],
                                        chain_HI[:,6],chain_HI[:,6],
                                        chain_HI[:,7],chain_HI[:,7])).T
                    
                elif network_hi.hypermodel_type == 'regular':
                    mus = np.vstack((chain_HI[:,0],chain_HI[:,1],
                                    chain_HI[:,2],chain_HI[:,3],
                                    chain_HI[:,4],chain_HI[:,5])).T
                    sigmas = np.vstack((chain_HI[:,6],chain_HI[:,7],chain_HI[:,8],
                                        chain_HI[:,9],chain_HI[:,10],chain_HI[:,11])).T
                
                # dot product: (xi_k - mu)**2 dot (1/sigmas)**2 over num_params dimension
                # ((xi_k - mus)/sigmas)**2 should have dimension [num_samps,num_params]
                exponent = -0.5*(np.sum(((xi_k - mus)/sigmas)**2,axis=1))
                # exponent should now have dimension num_samps
                # sqrt(det(sigmas**2)) = prod(sigmas)
                to_sum = (1/(np.product(sigmas,axis=1)))*np.exp(exponent)
                sum = np.sum(to_sum)

                # divide out interim prior piece
                interim_exponent = 0.5*np.sum(((xi_k - train_mean)/train_scatter)**2)
                to_multiply = np.prod(train_scatter)*np.exp(interim_exponent)

                weights[k] = NPE_multivariate_sampler.pdf(xi_k) * sum * to_multiply / np.shape(chain_HI)[0] 

            if samps_weights_path is not None:
                h5f.create_dataset('samples_%d'%(i), data=NPE_samples)
                h5f.create_dataset('weights_%d'%(i), data=weights)
            samples_list.append(NPE_samples)
            weights_list.append(weights)

            # only evaluate once if de-bugging
            if debug:
                break
        
        if samps_weights_path is not None:
            h5f.close()

        # return samples & weights for each lens posterior
        return samples_list, weights_list
    

    @staticmethod
    def load_samps_weights(file_path):
        """
        Load samples_list, weights_list from .h5 file

        Args:
            file_path (string)
        """

        h5f = h5py.File(file_path, 'r')
        num_lenses = int(len(list(h5f.keys()))/2)

        samples_list = []
        weights_list = []

        for i in range(0,num_lenses):
            samples_list.append(h5f.get('samples_%d'%(i)).value)
            weights_list.append(h5f.get('weights_%d'%(i)).value)
        h5f.close()

        return samples_list, weights_list