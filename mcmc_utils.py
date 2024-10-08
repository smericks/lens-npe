import numpy as np
import incredible as cr
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal
import visualization_utils
import pandas as pd
import seaborn as sns

# define some global colors to standardize plotting
palette = sns.color_palette('muted').as_hex()
COLORS = {
    'prior':palette[7],
    'hyperparam':palette[3],
    'hyperparam_narrow':palette[1],
    'unweighted_NPE':palette[0],
    'reweighted_NPE':palette[4],
    'fm_posterior':palette[2]
}

def fixed_param_sampling_HI(n_lenses,y_pred,prec_pred,val_metadata_path,n_emcee_samps=1e4):
    # doesn't require a functional form for the training prior, just a histogram of values
    # one of the args should just be the path to the validation metadata.csv (is 5,000 samples enough?)
    
    # num samples used for importance sampling
    N_IS = 5000
    # get samples from p(xi_k|Omega_int,d_k)
    cov_pred = np.linalg.inv(prec_pred)
    # NPE samples in shape (5000,11,6)
    from scipy.stats import multivariate_normal
    NPE_samples = np.empty((5000,n_lenses,6))
    for j in range(0,n_lenses):
        NPE_samples[:,j,:] = multivariate_normal(mean=y_pred[j,:6],
                                          cov=cov_pred[j,:6,:6]).rvs(size=(N_IS,6))

    # need p(xi_k|Omega_int) for each NPE sample
    from scipy.stats import rv_histogram
    # retrieve distribution of lens parameters from validation set metadata
    val_df = pd.read_csv(val_metadata_path)
    params_of_interest = ['main_deflector_parameters_theta_E',
                          'main_deflector_parameters_gamma1','main_deflector_parameters_gamma2',
                          'main_deflector_parameters_gamma',
                          'main_deflector_parameters_e1','main_deflector_parameters_e2']
    NPE_interim_probs = np.empty((5000,n_lenses,6))
    for i,p in enumerate(params_of_interest):
        hist = np.histogram(val_df[p].to_numpy())
        NPE_interim_probs[:,:,i] = rv_histogram(hist).pdf(NPE_samples[:,:,i])

    def log_likelihood(hyperparameters):
        # compute p(xi_k|Omega) for NPE samples given hyperparameters from above
        loc_sampled = np.asarray([hyperparameters[0],0.,0.,hyperparameters[1],0.,0.])
        cov_sampled = np.diag(np.asarray([hyperparameters[2],hyperparameters[3],hyperparameters[3],
            hyperparameters[4],hyperparameters[5],hyperparameters[5]])**2)
        NPE_omega_probs = multivariate_normal(loc=loc_sampled,cov=cov_sampled).pdf(NPE_samples)

        # make sure we don't overflow when coming out of log space for this calculation
        #with jax.experimental.enable_x64():

        # compute importance sampling sum/product (rip log space)
        imp_sampling_sum = np.sum(NPE_omega_probs/NPE_interim_probs,axis=0)
        # log(product) = sum(logs)
        imp_sampling_log_factor = np.sum(np.log(imp_sampling_sum))

        return imp_sampling_log_factor

    return fixed_param_HI(log_likelihood,n_emcee_samps)

def individ_corner_plots(emcee_chain,true_hyperparameters,param_labels,n_lenses,burnin=int(1e3)):
    """
    Args:
        emcee_chain (array[n_walkers,n_samples,n_params])
    """

    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))
    fontsize = 18
    color='#FFAA00'
    truth_color = 'grey'
    hist_kwargs = {'density':True,'color':color,'lw':3}
    
    for j in range(0,int(num_params/2)):
        # show just gamma_lens 
        figure = corner.corner(chain[:,[j,j+int(num_params/2)]],labels=np.asarray(param_labels)[[j,j+int(num_params/2)]],
                bins=20,show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=fontsize),
                levels=[0.68,0.95],color=color,fill_contours=True,hist_kwargs=hist_kwargs,title_fmt='.2f',
                max_n_ticks=3,fig=None,truths=true_hyperparameters[[j,j+int(num_params/2)]],
                truth_color=truth_color,range=np.ones(2)*0.98)

        axes = np.array(figure.axes).reshape((2, 2))
        custom_lines = [Line2D([0], [0], color=color, lw=4),
            Line2D([0], [0], color=truth_color, lw=4)]

        """"
        # set some custom axes limits
        if j == 3:
            mu_lim = [1.9,2.06]
            sigma_lim = [0.001,0.1]
            axes[0,0].set_xlim(mu_lim)
            axes[1,0].set_xlim(mu_lim)
            axes[1,0].set_ylim(sigma_lim)
            axes[1,1].set_xlim(sigma_lim)
        """

        axes[0,1].legend(custom_lines,['Samples','Truth'],frameon=False,fontsize=12)

        plt.suptitle('%d Lenses'%(n_lenses),fontsize=15)
        plt.show()


def doppel_individ_corner_plots(emcee_chain,param_labels,n_lenses,burnin=int(1e3)):
    """
    Args:
        emcee_chain (array[n_walkers,n_samples,n_params])
    """

    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))
    fontsize = 18
    color='#FFAA00'
    hist_kwargs = {'density':True,'color':color,'lw':3}
    
    for j in range(0,int(num_params/2)):
        # show just gamma_lens 
        figure = corner.corner(chain[:,[j,j+int(num_params/2)]],labels=np.asarray(param_labels)[[j,j+int(num_params/2)]],
                bins=20,show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=fontsize),
                levels=[0.68,0.95],color=color,fill_contours=True,hist_kwargs=hist_kwargs,title_fmt='.2f',
                max_n_ticks=3,fig=None,range=np.ones(2)*0.98)

        #axes = np.array(figure.axes).reshape((2, 2))
        #custom_lines = [Line2D([0], [0], color=color, lw=4),
        #    Line2D([0], [0], color=truth_color, lw=4)]

        """"
        # set some custom axes limits
        if j == 3:
            mu_lim = [1.9,2.06]
            sigma_lim = [0.001,0.1]
            axes[0,0].set_xlim(mu_lim)
            axes[1,0].set_xlim(mu_lim)
            axes[1,0].set_ylim(sigma_lim)
            axes[1,1].set_xlim(sigma_lim)
        """

        #axes[0,1].legend(custom_lines,['Samples','Truth'],frameon=False,fontsize=12)

        plt.suptitle('%d Doppelganger Lenses'%(n_lenses),fontsize=15)
        plt.show()

def overlay_contours(emcee_chains_list,colors_list,iofi,
    param_labels,sampler_labels,true_params=None,save_path=None,bounds=None,
    y_bounds = None,burnin=int(1e3),annotate_medians=False,dpi=300):
    """
    Args:
        emcee_chains_list (list): list of chains from emcee sampler
        colors_list (list): list of colors for each contour
        iofi (list[int]): list of indices of parameters to be plotted
        true_params ([float]): ground truth for iofi ONLY
        param_labels (list[string]): list of labels for each parameter of interest
        sampler_labels (list[string]): list of labels for each sampler/contour
        bounds (list): list of [min,max] bounds for each param's contour
        y_bounds (list): list of [min,max] bounds for y-axis of marginal hists
    """
    import copy


    corner_kwargs = {
        'labels':np.asarray(param_labels),
        'bins':20,
        'show_titles':False,
        'plot_datapoints':False,
        'levels':[0.68,0.95],
        'color':colors_list[0],
        'fill_contours':True,
        'contourf_kwargs':{},
        'hist_kwargs':{'density':True,'color':colors_list[0],
                       'lw':3},
        'title_fmt':'.2f',
        'plot_density':False,
        'max_n_ticks':3,
        'range':np.ones(len(iofi))*0.98,
        'smooth':0.5,
        'label_kwargs':{
            'color':'black',
            'fontsize':40,
            'labelpad':0.02
        }
    }

    if true_params is not None:
        corner_kwargs['truths'] = true_params
        corner_kwargs['truth_color'] = 'black'

    figure = plt.figure(dpi=dpi,figsize=(12,12))

    for i,emcee_chain in enumerate(emcee_chains_list):
        if i == 0:
            corner_kwargs_copy = copy.deepcopy(corner_kwargs)
            figure = param_of_interest_corner(emcee_chain,iofi,corner_kwargs_copy,
                burnin=burnin,figure=figure)
        else:
            corner_kwargs_copy = copy.deepcopy(corner_kwargs)
            corner_kwargs_copy['color'] = colors_list[i]
            corner_kwargs_copy['hist_kwargs']['color'] = colors_list[i] 
            figure = param_of_interest_corner(emcee_chain,iofi,corner_kwargs_copy,
                figure=figure,burnin=burnin)

    num_params = len(iofi)
    axes = np.array(figure.axes).reshape((num_params, num_params))
    custom_lines = []
    for color in colors_list:
        custom_lines.append(Line2D([0], [0], color=color, lw=6))

    axes[0,num_params-1].legend(custom_lines,sampler_labels,frameon=False,
                fontsize=30,loc=7)
    #axes[0,num_params-1].legend(custom_lines,sampler_labels,
    #    frameon=False,fontsize=30)
    plt.subplots_adjust(wspace=0.17,hspace=0.17)
    for r in range(0,num_params):
        for c in range(0,r+1):

            axes[r,c].tick_params(labelsize=17)

            if bounds is not None:
                axes[r,c].set_xlim(bounds[c])
                if r != c :
                    axes[r,c].set_ylim(bounds[r])
                elif y_bounds is not None:
                    # this means r == c
                    axes[r,c].set_ylim(y_bounds[r])

            if annotate_medians and r==c:
                legend_text_list = []
                for emcee_chain in emcee_chains_list:

                    num_params = emcee_chain.shape[-1]
                    if (len(emcee_chain.shape) == 3):
                        chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))
                    else:
                        chain = emcee_chain[burnin:,:]
                        
                    
                    med = np.median(chain[:,iofi[c]])
                    low = np.quantile(chain[:,iofi[c]],q=0.1586)
                    high = np.quantile(chain[:,iofi[c]],q=0.8413)

                    legend_text_list.append('%.2f $\pm^{%.2f}_{%.2f}$'%(med,high-med,med-low))

                axes[r,c].legend(custom_lines,legend_text_list,
                    frameon=True,fontsize=16,handlelength=1,loc='lower right',
                    color='black')
                #axes[r,c].set_title(param_labels[c])

    if save_path:
        plt.savefig(save_path,bbox_inches='tight')

    return figure



def overlay_shot_noise_contours(emcee_chains_list,iofi,param_labels,true_params,
    burnin=int(1e3)):

    transparent_color = COLORS['hyperparam_narrow'] + '4D'

    corner_kwargs = {
        'labels':np.asarray(param_labels),
        'bins':20,
        'show_titles':False,
        'plot_datapoints':False,
        'label_kwargs':dict(fontsize=25),
        'levels':[0.68,0.95],
        'color':transparent_color,
        'fill_contours':True,
        'contourf_kwargs':{},
        'hist_kwargs':{'density':True,'color':COLORS['hyperparam_narrow'],
                       'lw':3,'alpha':0.3},
        'title_fmt':'.2f',
        'plot_density':False,
        'max_n_ticks':3,
        'truths':np.zeros(len(iofi)),
        'truth_color':'grey',
        'range':np.ones(len(iofi))*0.98,
    }

    for i,emcee_chain in enumerate(emcee_chains_list):
        if i == 0:
            figure = param_of_interest_corner(emcee_chain,iofi,corner_kwargs,true_params,
                                              burnin=burnin)
        else:
            figure = param_of_interest_corner(emcee_chain,iofi,corner_kwargs,true_params,
                                              figure=figure,burnin=burnin)

    return figure

def param_of_interest_corner(emcee_chain,iofi,corner_kwargs,true_params=None,
                             burnin=int(1e3),bounds=None,title=None,
                             figure=None,display_metric=False):
    """
    Args: 
        emcee_chain (array[n_walkers,n_samples,n_params])
        iofi ([int]): list of indices of which params to plot
        corner_kwargs (dict): corner.corner() arguments
        true_params (list[float]): list of ground truth ONLY for iofi
    Returns:
        matplotlib figure object
    """
    num_params = emcee_chain.shape[-1]
    if (len(emcee_chain.shape) == 3):
        chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))
    else:
        chain = emcee_chain[burnin:,:]
    # subtract off the ground truth for params of interest
    if true_params is not None:
        for j,i in enumerate(iofi):
            chain[:,i] -= true_params[j]

    print(chain.shape)
    if figure is None:
        figure = corner.corner(chain[:,iofi],fig=None,**corner_kwargs)
    else:
        figure = corner.corner(chain[:,iofi],fig=figure,**corner_kwargs)

    num_params = len(iofi)
    axes = np.array(figure.axes).reshape((num_params, num_params))
    custom_lines = [Line2D([0], [0], color=COLORS['hyperparam_narrow'], lw=2),
        Line2D([0], [0], color='grey', lw=2)]

    axes[0,num_params-1].legend(custom_lines,['Narrow','Truth'],frameon=False,fontsize=12)

    if title is not None:
        plt.suptitle(title,fontsize=20)


    if bounds is not None:

        for r in range(0,num_params):
            for c in range(0,r+1):
                axes[r,c].set_xlim(bounds[c])
                if r != c :
                    axes[r,c].set_ylim(bounds[r])
                else:
                    if display_metric:
                        c_idx = iofi[c]
                        med = np.median(chain[:,c_idx])
                        low = np.quantile(chain[:,c_idx],q=0.1586)
                        high = np.quantile(chain[:,c_idx],q=0.8413)
                        axes[r,c].legend(custom_lines,
                            ['%.2f $\pm^{%.2f}_{%.2f}$'%(med,high-med,med-low),'%.2f'%(true_params[c])],
                            frameon=True,fontsize=14,handlelength=1,loc='lower right')

    #plt.show()

    return figure

##############################
# corner plot for all params
##############################
def full_corner_plot(emcee_chain,true_hyperparameters,param_labels,title,bounds=None,burnin=int(1e3),
                     label_fontsize=18,title_fontsize=18,save_path=None):
    """
    Args:
        emcee_chain (array[n_walkers,n_samples,n_params])
    """
    fontsize = label_fontsize
    color=COLORS['hyperparam']
    truth_color = 'black'
    hist_kwargs = {'density':True,'color':color,'lw':3}

    num_params = emcee_chain.shape[2]
    chain3 = emcee_chain[:,burnin:,:].reshape((-1,num_params))

    print(chain3.shape)

    figure = corner.corner(chain3,labels=np.asarray(param_labels),bins=20,
                show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=fontsize),
                levels=[0.68,0.95],color=color,fill_contours=True,
                hist_kwargs=hist_kwargs,title_fmt='.2f',max_n_ticks=3,fig=None,
                truths=true_hyperparameters,
                truth_color=truth_color,range=np.ones(num_params)*0.98)

    axes = np.array(figure.axes).reshape((num_params, num_params))
    custom_lines = [Line2D([0], [0], color=color, lw=4),
            Line2D([0], [0], color=truth_color, lw=4)]
    #axes[0,5].legend(custom_lines,['Samples','Truth'],frameon=False,fontsize=15)

    # set some custom axes limits if bounds is not None
    custom_lines = [Line2D([0], [0], color=color, lw=2),
            Line2D([0], [0], color=truth_color, lw=2)]
    if bounds is not None:

        for r in range(0,num_params):
            for c in range(0,r+1):
                if r == c and r == 0:
                    print('0,0')
                axes[r,c].set_xlim(bounds[c])
                if r != c :
                    axes[r,c].set_ylim(bounds[r])
                else:
                    med = np.median(chain3[:,c])
                    low = np.quantile(chain3[:,c],q=0.1586)
                    high = np.quantile(chain3[:,c],q=0.8413)
                    axes[r,c].legend(custom_lines,
                        ['%.2f $\pm^{%.2f}_{%.2f}$'%(med,high-med,med-low),'%.2f'%(true_hyperparameters[c])],
                        frameon=True,fontsize=12,handlelength=1,loc='lower right')
                    axes[r,c].set_title(param_labels[c])
                    """
                    axes[r,c].text(0.8,0.9,'%.2f $\pm^{%.2f}_{%.2f}$'%(med,high,low),
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   transform=axes[r,c].transAxes,fontsize=10,color=color)
                    axes[r,c].text(0.5,0.9,'%.2f'%(true_hyperparameters[c]),
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   transform=axes[r,c].transAxes,fontsize=10,color=truth_color)
                    """
                    

    plt.suptitle(title,fontsize=title_fontsize)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def doppel_corner_plot(emcee_chain,param_labels,title,bounds=None,burnin=int(1e3)):
    fontsize = 18
    color='#FFAA00'
    hist_kwargs = {'density':True,'color':color,'lw':3}

    chain3 = emcee_chain[:,burnin:,:].reshape((-1,6))

    figure = corner.corner(chain3,labels=np.asarray(param_labels),bins=20,
                show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=fontsize),
                levels=[0.68,0.95],color=color,fill_contours=True,
                hist_kwargs=hist_kwargs,title_fmt='.2f',max_n_ticks=3,fig=None,range=np.ones(6)*0.98)

    axes = np.array(figure.axes).reshape((6, 6))
    custom_lines = [Line2D([0], [0], color=color, lw=4)]
    axes[0,5].legend(custom_lines,['Samples'],frameon=False,fontsize=15)

    # set some custom axes limits if bounds is not None
    custom_lines = [Line2D([0], [0], color=color, lw=2)]

    if bounds is not None:
        for r in range(0,6):
            for c in range(0,r+1):
                if r == c and r == 0:
                    print('0,0')
                axes[r,c].set_xlim(bounds[c])
                if r != c :
                    axes[r,c].set_ylim(bounds[r])
                else:
                    med = np.median(chain3[:,c])
                    low = np.quantile(chain3[:,c],q=0.1586)
                    high = np.quantile(chain3[:,c],q=0.8413)
                    axes[r,c].legend(custom_lines,
                        ['%.2f $\pm^{%.2f}_{%.2f}$'%(med,high-med,med-low)],
                        frameon=True,fontsize=9,handlelength=1,loc='lower right')
                    axes[r,c].set_title(param_labels[c])
                    """
                    axes[r,c].text(0.8,0.9,'%.2f $\pm^{%.2f}_{%.2f}$'%(med,high,low),
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   transform=axes[r,c].transAxes,fontsize=10,color=color)
                    axes[r,c].text(0.5,0.9,'%.2f'%(true_hyperparameters[c]),
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   transform=axes[r,c].transAxes,fontsize=10,color=truth_color)
                    """

    for r in range(0,6):
        for c in range(0,r+1):
            if r==c:
                med = np.median(chain3[:,c])
                low = np.quantile(chain3[:,c],q=0.1586)
                high = np.quantile(chain3[:,c],q=0.8413)
                axes[r,c].legend(custom_lines,
                    ['%.2f $\pm^{%.2f}_{%.2f}$'%(med,high-med,med-low)],
                    frameon=True,fontsize=12,handlelength=1,loc='lower right')
                axes[r,c].set_title(param_labels[c])

                    

    plt.suptitle(title,fontsize=20)
    plt.show()

def apply_corner_bounds(bounds,figure):

    axes = np.array(figure.axes).reshape((len(bounds), len(bounds)))
    for r in range(0,len(bounds)):
        for c in range(0,r+1):
            axes[r,c].set_xlim(bounds[c])
            if r != c :
                axes[r,c].set_ylim(bounds[r])


def HI_medians_table(emcee_chain,param_labels,burnin=1e3):

    burnin = int(burnin)
    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))

    med = np.median(chain,axis=0)
    low = np.quantile(chain,q=0.1586,axis=0)
    high = np.quantile(chain,q=0.8413,axis=0)
    sigma = ((high-med)+(med-low))/2

    for i in range(0,num_params):
        print(param_labels[i],': ',med[i],' $\pm$', sigma[i])



#################################
# analyze chains from original HI
#################################
def analyze_chains(emcee_chain,param_labels,true_hyperparameters,
                    outfile,effective_samples=False,show_chains=False,
                    burnin=int(1e3)):
    """
    Args:
        emcee_chain (array[n_walkers,n_samples,n_params])
    """

    if show_chains:
        for i in range(emcee_chain.shape[2]):
            plt.figure()
            #indices = np.arange(0,emcee_chain.shape[1],10)
            plt.plot(emcee_chain[:,:,i].T,'.')
            plt.title(param_labels[i])
            plt.show()

    if effective_samples:
        print("Effective Samples Per Parameter:")
        print(cr.effective_samples(emcee_chain[:,burnin:,:], maxlag=5000))
        print("")

    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))

    labels = ['Ground Truth', 'Inferred Value', 'Bias in $\sigma$', 'Fractional Error']

    med = np.median(chain,axis=0)
    low = np.quantile(chain,q=0.1586,axis=0)
    high = np.quantile(chain,q=0.8413,axis=0)

    error = med - true_hyperparameters
    sigma = ((high-med)+(med-low))/2
    bias = error/sigma

    metrics = [true_hyperparameters,med,bias,error]

    with open(outfile,'w') as f:

        f.write('\hline')
        f.write('\n')

        for i,lab in enumerate(labels):
            f.write(lab)
            f.write(' ')

            for k,m in enumerate(metrics[i]):
                f.write('& ')
                f.write(str(np.around(m,2)))
                f.write(' ')
                if i == 1:
                    f.write('$\pm$' + str(np.around(sigma[k],2)))

            f.write(r'\\')
            f.write('\n')
            f.write('\hline')
            f.write('\n')


    for j in range(num_params):

        med = np.median(chain[:,j])
        low = np.quantile(chain[:,j],q=0.1586)
        high = np.quantile(chain[:,j],q=0.8413)
        print(param_labels[j])
        print("\t", round(med,3), "+", round(high-med,3), "-", round(med-low,3))
        error = med - true_hyperparameters[j]
        if error > 0:
            bias = error/round(med-low,3)
        else:
            bias = error/round(high-med,3)
        print("\t", "Bias in Std. Devs: ", round(bias,3))

        frac_error = error/true_hyperparameters[j]
        print("\t","Fractional Error: ",round(frac_error,3))

def investigate_chain_param(emcee_chain,param_index):
    """In depth look at all chains for a single parameter
        Args:
            emcee_chain (array[n_walkers,n_samps,n_params])
            param_index (int): index into n_params for param of interest
    """

    indices = np.arange(0,emcee_chain.shape[1],10)
    #5 walkers at a time
    for i in range(0,emcee_chain.shape[0]//5):
        plt.figure()
        plt.plot(emcee_chain[5*i:(5*(i+1)),:,:][:,indices,:][:,:,param_index].T)
        plt.show()


############################
# Fractional error on params
############################
def fractional_error_chain(emcee_chain,true_hyperparameters,burnin=int(1e3)):
    """Compute fractional error on hyperparameters given a sampler object from 
    MCMC

    Args:
        emcee_chain (array[n_walkers,n_samples,n_params])
        true_hyperparameters (list[float]): ground truth for hyperparameters
        burnin (int,default=1e3): How many samples to remove from the front
            of the MCMC chain
    Returns:
        list(float)
    """

    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))

    med = np.median(chain,axis=0)
    error = med - true_hyperparameters
    frac_error = error/true_hyperparameters

    return frac_error


# TAKEN FROM: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!

    Args: 
        values: numpy.array with data
        quantiles: array-like with many quantiles needed
        sample_weight: array-like of the same length as `array`
        values_sorted: bool, if True, then will avoid sorting of
        initial array
        old_style: if True, will correct output to be consistent
        with numpy.percentile.
    Returns: 
        numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    # weighted_quantiles & values has to be 1d for this function to work
    return np.interp(quantiles, weighted_quantiles, values)


def reweighted_table(samples_list,weights_list,y_truth,file_name):
    """Takes samples & weights and returns a table of metrics after computing
    the weighted median & spread using quantiles

    Args:
        samples_list: shape (n_lenses,n_samples,n_params)
        weights_list: shape (n_lenses,n_samples)
        y_truth: shape(n_lenses,n_params)
        file_name: where to save table to a .txt file. If None, prints to stdout
    
    Returns:
        numpy.arrays y_pred,std_pred

    """

    # initialize final arrays
    y_pred = np.empty((len(samples_list),samples_list[0].shape[1]))
    std_pred = np.empty((len(samples_list),samples_list[0].shape[1]))
    # loop over lenses
    for i in range(len(samples_list)):
        samps = samples_list[i]
        weights = weights_list[i]

        # weighted_quantiles can't do more than one param at a time
        param_medians = []
        param_sigmas = []
        for p in range(0,samps.shape[1]):
            median,low,high = weighted_quantile(samps[:,p],[0.5,0.1586,0.8413],weights)
            sigma = ((high-median)+(median-low))/2
            param_medians.append(median)
            param_sigmas.append(sigma)
        #mus = np.sum(samps*weights,axis=0)/np.sum(weights,axis=0)
        #stds = np.sqrt(np.sum(weights*(samps - mus)**2,axis=0)/np.sum(weights,axis=0))
        y_pred[i,:] = np.asarray(param_medians)
        std_pred[i,:] = np.asarray(param_sigmas)

    visualization_utils.table_metrics(y_pred,y_truth,std_pred,file_name)

    return y_pred,std_pred