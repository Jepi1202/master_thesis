import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def simpleVelocityDistr(sim, display:bool = True, s:bool = True):
    inds = np.arange(sim.shape[0]-1)
    inds2 = inds+1
    
    vel = np.linalg.norm(sim[inds2, :, :] - sim[inds, :, :], axis = -1)
    
    if display:
        plt.grid()
        plt.hist(vel.reshape(-1), bins = 'auto')
        plt.xlabel('Velocity distribution')
        plt.ylabel('Number of occurences')
        
        if s:
            plt.show()
        
    return vel


def plotVelTime(vel, s:bool=True, display:bool = True):
    m = np.mean(vel, axis=-1)
    std = np.std(vel, axis=-1)
    inds = np.arange(len(m))
    
    if display:
        plt.grid()
        plt.xlabel("Timesteps")
        plt.ylabel('Norm of speed')
        plt.plot(m, 'b')
        plt.fill_between(inds, m + std, m - std, where=(m + std) >= (m - std), facecolor='red', alpha=0.4)

        if s:
            plt.show()
        
    return m, std



def computeNodesError(sim1, sim2):
    vals = (sim1-sim2)
    m = np.mean(np.linalg.norm(vals, axis = -1), axis = -1)
    std = np.std(np.linalg.norm(vals, axis = -1), axis = -1)
    
    return m, std




def scaleMagnVel(sim:np.array, bins:np.array, bins2:np.array, display:bool = True)->np.array:
    """ 
    Fucntion to compute the velocity maginutude distr

    Args:
    -----
        - `sim`: simulatioon 
        - `bins`: bins array
        - `bins2`: component histogram bins

    Returns:
    --------
        the tuple of velocities magnitude and components
    """

    # get the velocits

    inds0 = np.arange(sim.shape[0]-1)
    inds = inds0 + 1

    speeds = sim[inds] - sim[inds0]
    vmagn = np.linalg.norm(speeds, axis=-1)
    vx = speeds[:, :, 0]
    vy = speeds[:, :, 1]

    # average speed according to the cells
    avgSpeed = np.mean(vmagn, axis = -1)


    # intitialize the three histograms

    magnDist = np.zeros(len(bins)-1)
    magnDistX = np.zeros(len(bins2)-1)
    magnDistY = np.zeros(len(bins2)-1)


    # loop over cells
    for i in range(speeds.shape[0]):
                
        vdist,_=np.histogram(vmagn[i, :]/avgSpeed[i],bins,density=True)
        magnDist += vdist

        vdistx,_=np.histogram(vx[i, :]/avgSpeed[i],bins2,density=True)
        magnDistX += vdistx


        vdisty,_=np.histogram(vy[i, :]/avgSpeed[i],bins2,density=True)
        magnDistY += vdisty
    
    magnDist = magnDist/speeds.shape[0]
    magnDistX = magnDistX/speeds.shape[0]
    magnDistY = magnDistY/speeds.shape[0]
    


    if display:
        fig=plt.figure()
        db=bins[1]-bins[0]
        plt.grid()
        plt.semilogy(bins[1:]-db/2,magnDist,'r.-',lw=2)
        plt.xlabel('v/<v>')
        plt.ylabel('P(frac{v}{<v>})')
        plt.title('Scaled velocity magnitude distribution')
        plt.show()

        fig=plt.figure()
        db=bins2[1]-bins2[0]
        plt.grid()
        plt.semilogy(bins2[1:]-db/2,magnDistX,'r.-',lw=2)
        plt.semilogy(bins2[1:]-db/2,magnDistY,'k.-',lw=2)
        plt.xlabel('v/<v>')
        plt.ylabel('$P(fracv/<v>)$')
        plt.title('Scaled velocity component (x & y) distribution')
        plt.show()


    return magnDist, magnDistX, magnDistY



def MSD(traj: np.array, display:bool = True)-> np.array:
    """
    Allows to compute the Mean Squared Displacement of the trajectories for all timestamps
    
    Args:
    -----
    - `traj`: np.array of N trajectories of length T [NxT]
    
    Output:
    -------
    Mean Squared Displacement for all timestamps
    """

    def MSD_comp(traj, tau):
        T = traj.shape[0]
        i = np.arange(T - tau)
        j = i + tau

        return np.linalg.norm(traj[j, :, :] - traj[i, :, :], axis=-1)**2

    res = []
    T = traj.shape[0]
    
    for tau in range(1, T):
        val = np.mean(np.mean(MSD_comp(traj, tau), axis=0), axis=0)
        res.append(val)
        
    if display:
        plt.grid()
        plt.loglog(res)
        plt.xlabel(r'$\tau$')
        plt.ylabel('MSD')

    return res





def getVelAuto(mat,display:bool=True):

    inds = np.arange(mat.shape[0]-1)
    inds1 = inds+1
    vel = mat[inds1,:,:] - mat[inds,:,:]
    L = vel.shape[0]
    N = vel.shape[1]

    velauto=np.empty((L,))    

    for t in range(mat.shape[0]-1):
        tmax=L-t

        val = vel[t:, :, 0]*vel[:tmax, :, 0] + vel[t:, :, 1]*vel[:tmax, :, 1]
        velauto[t]=np.sum(np.sum(val,axis=1),axis=0)/(N*tmax)
    #normalising back to velauto = 1 due to different averaging (types 1 & 2 rather than 1) - out by approx 4%                        
    xval=np.linspace(0,round(L*0.01*100),num=L)
    velauto /= velauto[0]
    
    if display:
        plt.grid()
        plt.plot(xval, velauto)
        plt.xlabel('Time')
        plt.ylabel('Velocity autocorrelation')

    return xval, velauto


def strucMeas(sim, thresholds):
    
    res = []
    # for each threshold
    for thresh in tqdm(thresholds):
        # loop on timesteps
        val = 0
        for t in range(sim.shape[0]):

            # loop on nodes
            for i in range(sim.shape[1]):
                
                for j in range(sim.shape[1]):
                    
                    if i != j:
                        d = np.linalg.norm(sim[t, i, :] - sim[t, j, :])

                        if d < thresh:
                            val += 1/d
                            # val += 1
                        
        res.append(val/(sim.shape[0] * sim.shape[1]))
        
    return res


#### self-scattering

def SelfIntermediate(mat,qval,display=True):
    # This is single particle, single q, shifted time step. Equivalent to the MSD, really
    SelfInt=np.empty((data.Nsnap-1,),dtype=complex)
            
    for t in range(data.Nsnap-1):

        smax=data.Nsnap-t
        # get (tracer) rval for up to smax
        frames = np.arange(smax)
        boollabels = np.isin(data.ptype[frames],[1])
        rt = data.rval[frames][boollabels].reshape(smax,-1,2)

        # get (tracer) rval for u to end        
        frames = np.arange(t,data.Nsnap)
        boollabels = np.isin(data.ptype[frames],[1])
        rtplus = data.rval[frames][boollabels].reshape(len(frames),-1,2)
        
        dr  = rt - rtplus

        if periodic:
            SelfInt[t]=np.sum(np.sum(np.exp(1.0j*qval[0]*dr[:,:,0]+ \
                                            1.0j*qval[1]*dr[:,:,1] \
                                        ),axis=1),axis=0)/(data.Ntracers*smax)         
        else:   
            SelfInt[t]=np.sum(np.sum(np.exp(1.0j*qval[0]*(rt[:,:,0]-rtplus[:,:,0]) + \
                                            1.0j*qval[1]*(rt[:,:,1]-rtplus[:,:,1])\
                                        ),axis=1),axis=0)/(data.Ntracers*smax)                    

        
    # Looking at the absolute value of it here
    SelfInt2=(np.real(SelfInt)**2 + np.imag(SelfInt)**2)**0.5
    
    tval=np.linspace(0,round(data.Nsnap*data.param.dt*data.param.output_time),num=data.Nsnap-1)
    if verbose:
        qnorm=np.sqrt(qval[0]**2+qval[1]**2)
        fig=plt.figure()
        plt.semilogy(tval,SelfInt2,'.-r',lw=2)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate, k = ' + str(qnorm))
        plt.ylim([0,1])
        plt.show()
 
    return tval, SelfInt2, SelfInt


#### Fourier


def makeQrad(dq,qmax,nq):
    """ 
    Makes a 2d repartition fo the frequencies

    Args:
    -----
        - `dq`: 2 pi / L
        - `qmax`: maximal q
        - `nq`: qmax/dq
    """
    nq2=int(2**0.5*nq)
    qmax2=2**0.5*qmax
    qx=np.linspace(0,qmax,nq)
    qy=np.linspace(0,qmax,nq)
    qrad=np.linspace(0,qmax2,nq2)
    # do this silly counting once and for all
    binval=np.empty((nq,nq))
    for kx in range(nq):
        for ky in range(nq):
            qval=np.sqrt(qx[kx]**2+qy[ky]**2)
            binval[kx,ky]=round(qval/dq)
    ptsx=[]
    ptsy=[]
    # do the indexing arrays
    for l in range(nq2):
        pts0x=[]
        pts0y=[]
        for kx in range(nq):
            hmm=np.nonzero(binval[kx,:]==l)[0]
            for v in range(len(hmm)):
                pts0y.append(hmm[v])
                pts0x.append(kx)
        ptsx.append(pts0x)
        ptsy.append(pts0y)
    return qx, qy, qrad, ptsx, ptsy



def FourierTrans(mat, L, qmax=0.3,verbose=True, debug=False):
    
    # Note to self: only low q values will be interesting in any case. 
    # The stepping is in multiples of the inverse box size. Assuming a square box.
    dq=2*np.pi/L

    nq=int(qmax/dq)
    if debug:
        print("Fourier transforming positions")
        print("Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps.")
    qx, qy, qrad, ptsx, ptsy=makeQrad(dq,qmax,nq)
    #print " After Qrad" 
    fourierval=np.zeros((nq,nq),dtype=complex)
    
    #index relevant particles (by default we use all of them)
    useparts = data.gettypes(usetype,whichframe)
    N = len(useparts)
    for kx in range(nq):
        for ky in range(nq):
            # And, alas, no FFT since we are most definitely off grid. And averaging is going to kill everything.
            fourierval[kx,ky]=np.sum(np.exp(1j*(qx[kx]*data.rval[whichframe,useparts,0]+qy[ky]*data.rval[whichframe,useparts,1])))/N
    plotval=N*(np.real(fourierval)**2+np.imag(fourierval)**2)
    
    # Produce a radial averaging to see if anything interesting happens
    nq2=int(2**0.5*nq)
    valrad=np.zeros((nq2,))
    for l in range(nq2):
        valrad[l]=np.mean(plotval[ptsx[l],ptsy[l]])#, axis=0)
    
    if debug:
        plt.figure()
        plt.pcolor(qx,qy,plotval, vmin=0, vmax=3,shading='auto' )
        plt.colorbar()
        plt.title('Static structure factor (2d)')
        
        plt.figure()
        plt.plot(qrad,valrad,'.-r',lw=2)
        plt.xlabel('q')
        plt.ylabel('S(q)')
        plt.title('Static structure factor (radial)')
        
    return qrad,valrad








#############################################################

# making plots


def plotMSE(sim1:list, sim2:list, display:bool, path:str = None):

    meanRes = None
    stdRes = None
    for i in range(len(sim1)):
        m, std = computeNodesError(sim1[i][:100, :,:], sim2[i][:100, :, :])

        if meanRes is None:
            meanRes = m
        else:
            meanRes += m

        if stdRes is None:
            stdRes = std
        else:
            stdRes += std


    meanRes /= len(sim1)
    stdRes /= len(sim1)

    plt.plot(meanRes)
    plt.fill_between(np.arange(len(meanRes)), meanRes - stdRes, meanRes+stdRes, where=(meanRes - stdRes) <= (meanRes+stdRes), facecolor='red', alpha=0.4)
    plt.grid()
    plt.ylabel('Distance between $\hat{x}$ and x')
    plt.xlabel('Timesteps')
    if path is not None:
        plt.savefig(path)

    if display:
        plt.show()
    else:
        plt.close()


    return meanRes, stdRes




def plotMSD(sim1, sim2, display, path):

    r1 = None
    r2 = None

    for i in range(len(sim1)):
        if r1 is None:
            r1 = MSD(sim1, display = False)
            r2 = MSD(sim2[:sim1.shape[0]], display = False)
        else:
            r1 += MSD(sim1, display = False)
            r2 += MSD(sim2[:sim1.shape[0]], display = False)

    r1 /= len(sim1)
    r2 /= len(sim1)

    plt.grid()
    plt.loglog(r1, 'orange',label = 'ground truths')
    plt.loglog(r2, 'b',label = 'predictions')
    plt.xlabel(r'$\tau$')
    plt.ylabel('MSD')
    plt.legend()

    if path:
        plt.savefig(path)
    if display:
        plt.show()
    else:
        plt.close()




def plotDiffMSD(sim1, sim2, display, path):
    v = None

    for i in range(len(sim1)):
        r1 = MSD(sim1, display = False)
        r2 = MSD(sim2[:sim1.shape[0]], display = False)
        vals = (np.array(r2) + np.array(r1))/2
        if v is None:
            v = (np.array(r2) - np.array(r1))/vals
        else:
            v += (np.array(r2) - np.array(r1))/vals

    v /= len(sim1)

    plt.grid()
    plt.plot(v)
    plt.xlabel(r'$\tau$')
    plt.ylabel('normalized $\Delta$ MSD')

    if path:
        plt.savefig(path)
    if display:
        plt.show()
    else:
        plt.close()



def plotScaleVelo(sim1, sim2, display, path):

    b1=np.linspace(0,2,100)
    b2=np.linspace(-2,2,100)
    a1, a2, a3 = None, None, None
    c1, c2, c3 = None, None, None
    v = None

    for i in range(len(sim1)):
        if a1 is None:
            a1, a2, a3 = scaleMagnVel(sim1[i].copy(), bins = b1, bins2 = b2, display = False)
            c1, c2, c3 = scaleMagnVel(sim2[i][:sim1[i].shape[0]].copy(), bins = b1, bins2 = b2, display = False)

        else:
            a1b, a2b, a3b = scaleMagnVel(sim1[i].copy(), bins = b1, bins2 = b2, display = False)
            c1b, c2b, c3b = scaleMagnVel(sim2[i][:sim1[i].shape[0]].copy(), bins = b1, bins2 = b2, display = False)

            a1 += a1b
            a2 += a2b
            a3 += a3b
            c1 += c1b
            c2 += c2b
            c3 += c3b

    a1 /= len(sim1)
    a2 /= len(sim1)
    a3 /= len(sim1)
    c1 /= len(sim1)
    c2 /= len(sim1)
    c3 /= len(sim1)

    def plotMeanSVDist(bins, v1, v2, display, path):
        fig=plt.figure()
        db=bins[1]-bins[0]
        plt.grid()
        plt.semilogy(bins[1:]-db/2,v1,'r.-',lw=2, label = 'ground truth')
        plt.semilogy(bins[1:]-db/2,v2,'b.-',lw=2, label = 'predictions')
        plt.legend()
        plt.xlabel('v/<v>')
        plt.ylabel(r'P($\frac{v}{<v>}$)')
        plt.title('Scaled velocity magnitude distribution')
        if path:
            plt.savefig(path)
        if display:
            plt.show()
        else:
            plt.close()

    plotMeanSVDist(b1, a1, c1, display=display, path=path[0])
    plotMeanSVDist(b2, a2, c2, display=display, path=path[1])
    plotMeanSVDist(b2, a3, c3, display=display, path=path[2])








def plotCorrVel(sim1, sim2):
    vCor1L = None
    vCor2L = None
    for i in range(len(sim1)):
        xVal, vCor1 = getVelAuto(sim1[i].copy(), display = False)
        xVal, vCor2 = getVelAuto(sim2[i][:sim1[i].shape[0]].copy(), display = False)

        if vCor1L is None:
            vCor1L = vCor1
            vCor2L = vCor2
        else:
            vCor1L += vCor1
            vCor2L += vCor2


    vCor1L /= len(sim1)
    vCor2L /= len(sim1)

    plt.grid()
    plt.plot(xVal, vCor1L, 'orange', label = 'ground truth')
    plt.plot(xVal, vCor2L, 'b', label = 'predictions', alpha = 0.4)
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity autocorrelation')
    plt.show()
