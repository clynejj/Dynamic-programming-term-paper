import numpy as np
import scipy.optimize as optimize

from EconModel import EconModelClass
from consav.grids import nonlinspace
from consav import linear_interp, linear_interp_1d, linear_interp_2d, linear_interp_3d   
from consav import quadrature
from scipy.optimize import minimize,  NonlinearConstraint

# user-specified functions
import UserFunctions as usr

# set gender indication as globals
woman = 1
man = 2

class HouseholdModelClass(EconModelClass):
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. cpp
        self.cpp_filename = 'cppfuncs/solve.cpp'
        self.cpp_options = {'compiler':'vs'}
        
    def setup(self):
        par = self.par
        
        par.R = 1.03
        par.beta = 1.0/par.R # Discount factor
        
        par.div_A_share = 0.5 # divorce share of wealth to wife
        
       

        # Utility: gender-specific parameters
        par.rho_w = 2.0        # CRRA
        par.rho_m = 2.0        # CRRA
        
        par.alpha1_w = 1.0
        par.alpha1_m = 1.0
        
        par.alpha2_w = 1.0
        par.alpha2_m = 1.0
        
        par.phi_w = 0.2
        par.phi_m = 0.2
        
        par.theta0_w = 0.1 # constant disutility of work for women
        par.theta0_m = 0.1 # constant disutility of work for men
        par.theta1_w = 0.05 # additional disutility of work from children, women 
        par.theta1_m = 0.05 # additional disutility of work from children, men
        par.gamma = 0.5 # disutility of work elasticity

        # state variables
        par.T = 10
        
        # wealth
        par.num_A = 50
        par.max_A = 5.0
        
        # human capital 
        par.num_H = 50
        par.max_H = 5.0

        # income
        par.wage_const_1 = np.log(10_000.0) # constant, men
        par.wage_const_2 = np.log(10_000.0) # constant, women
        par.wage_K_1 = 0.1 # return on human capital, men
        par.wage_K_2 = 0.1 # return on human capital, women
        par.delta = 0.1 # depreciation in human capital

        # child-related transfers
        par.uncon_uni = 0.1
        par.means_level = 0.1
        par.means_slope = 0.001
        par.cond = -0.1
        par.cond_high = -0.1

        # bargaining power
        par.num_power = 21

        # love/match quality
        par.num_love = 41
        par.max_love = 1.0

        par.sigma_love = 0.1
        par.num_shock_love = 5

        # pre-computation
        par.num_Ctot = 100
        par.max_Ctot = par.max_A*2

        par.num_A_pd = par.num_A * 2

        # simulation
        par.seed = 9210
        par.simT = par.T
        par.simN = 50_000

        # grids        
        par.k_max = 20.0 # maximum point in HC grid
        par.num_k = 20 #30 # number of grid points in HC grid
        par.num_n = 2 # maximum number in my grid over children

        #interest rate
        par.r = 0.03
        
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # setup grids
        par.simT = par.T
        self.setup_grids()
        
        # singles
        shape_single = (par.T, par.num_n, par.num_A, par.num_k)
        sol.Vw_single = np.nan + np.ones(shape_single)
        sol.Vm_single = np.nan + np.ones(shape_single)
        sol.Cw_priv_single = np.nan + np.ones(shape_single)
        sol.Cm_priv_single = np.nan + np.ones(shape_single)
        sol.Cw_pub_single = np.nan + np.ones(shape_single)
        sol.Cm_pub_single = np.nan + np.ones(shape_single)
        sol.Cw_tot_single = np.nan + np.ones(shape_single)
        sol.Cm_tot_single = np.nan + np.ones(shape_single)
        sol.Hw_single = np.nan + np.ones(shape_single)
        sol.Hm_single = np.nan + np.ones(shape_single)

        sol.Vw_trans_single = np.nan + np.ones(shape_single)
        sol.Vm_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_priv_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_priv_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_pub_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_pub_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_tot_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_tot_trans_single = np.nan + np.ones(shape_single)
        sol.Hw_trans_single = np.nan + np.ones(shape_single)
        sol.Hm_trans_single = np.nan + np.ones(shape_single)

        # couples
        shape_couple = (par.T,par.num_power,par.num_love,par.num_A, par.num_k, par.num_k)
        sol.Vw_couple = np.nan + np.ones(shape_couple)
        sol.Vm_couple = np.nan + np.ones(shape_couple)
        
        sol.Cw_priv_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_couple = np.nan + np.ones(shape_couple)
        sol.C_pub_couple = np.nan + np.ones(shape_couple)
        sol.C_tot_couple = np.nan + np.ones(shape_couple)

        sol.Vw_remain_couple = np.nan + np.ones(shape_couple)
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple)
        
        sol.Cw_priv_remain_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_remain_couple = np.nan + np.ones(shape_couple)
        sol.C_pub_remain_couple = np.nan + np.ones(shape_couple)
        sol.C_tot_remain_couple = np.nan + np.ones(shape_couple)

        sol.power_idx = np.zeros(shape_couple,dtype=np.int_)
        sol.power = np.zeros(shape_couple)

        # temporary containers
        sol.savings_vec = np.zeros(par.num_shock_love)
        sol.Vw_plus_vec = np.zeros(par.num_shock_love) 
        sol.Vm_plus_vec = np.zeros(par.num_shock_love) 

        # EGM?? I think this is here from previous version... 
        sol.marg_V_couple = np.zeros(shape_couple)
        sol.marg_V_remain_couple = np.zeros(shape_couple)

        shape_egm = (par.num_power,par.num_love,par.num_A_pd)
        sol.EmargU_pd = np.zeros(shape_egm)
        sol.C_tot_pd = np.zeros(shape_egm)
        sol.M_pd = np.zeros(shape_egm)

        # pre-compute optimal consumption allocation - should I add human capital here?
        shape_pre = (par.num_power,par.num_Ctot)
        sol.pre_Ctot_Cw_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_Cm_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_C_pub = np.nan + np.ones(shape_pre)
        
        # simulation
        shape_sim = (par.simN,par.simT)
        sim.Cw_priv = np.nan + np.ones(shape_sim)
        sim.Cm_priv = np.nan + np.ones(shape_sim)
        sim.Cw_pub = np.nan + np.ones(shape_sim)
        sim.Cm_pub = np.nan + np.ones(shape_sim)
        sim.Cw_tot = np.nan + np.ones(shape_sim)
        sim.Cm_tot = np.nan + np.ones(shape_sim)
        sim.C_tot = np.nan + np.ones(shape_sim)
        
        sim.A = np.nan + np.ones(shape_sim)
        sim.Aw = np.nan + np.ones(shape_sim)
        sim.Am = np.nan + np.ones(shape_sim)
        sim.couple = np.nan + np.ones(shape_sim)
        sim.power_idx = np.ones(shape_sim,dtype=np.int_)
        sim.power = np.nan + np.ones(shape_sim)
        sim.love = np.nan + np.ones(shape_sim)

        # shocks
        np.random.seed(par.seed)
        sim.draw_love = np.random.normal(size=shape_sim)

        # initial distribution
        sim.init_A = par.grid_A[0] + np.zeros(par.simN)
        sim.init_Aw = par.div_A_share * sim.init_A #np.zeros(par.simN)
        sim.init_Am = (1.0 - par.div_A_share) * sim.init_A #np.zeros(par.simN)
        sim.init_couple = np.ones(par.simN,dtype=np.bool_)
        sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        sim.init_love = np.zeros(par.simN)
        
    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A = nonlinspace(0.0,par.max_A,par.num_A,1.1)

        par.grid_Aw = par.div_A_share * par.grid_A
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # human capital grid
        par.kw_grid = nonlinspace(0.0,par.k_max,par.num_k,1.1)
        par.km_grid = nonlinspace(0.0,par.k_max,par.num_k,1.1)

        # number of children grid
        par.nw_grid = np.arange(0,par.num_n)
        par.nm_grid = np.arange(0,par.num_n)


        # power. non-linear grid with more mass in both tails.
        odd_num = np.mod(par.num_power,2)
        first_part = nonlinspace(0.0,0.5,(par.num_power+odd_num)//2,1.3)
        last_part = np.flip(1.0 - nonlinspace(0.0,0.5,(par.num_power-odd_num)//2 + 1,1.3))[1:]
        par.grid_power = np.append(first_part,last_part)

        # love grid and shock
        if par.num_love>1:
            par.grid_love = np.linspace(-par.max_love,par.max_love,par.num_love)
        else:
            par.grid_love = np.array([0.0])

        if par.sigma_love<=1.0e-6:
            par.num_shock_love = 1
            par.grid_shock_love,par.grid_weight_love = np.array([0.0]),np.array([1.0])

        else:
            par.grid_shock_love,par.grid_weight_love = quadrature.normal_gauss_hermite(par.sigma_love,par.num_shock_love)

        # pre-computation
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)

        # EGM
        par.grid_util = np.nan + np.ones((par.num_power,par.num_Ctot))
        par.grid_marg_u = np.nan + np.ones(par.grid_util.shape)
        par.grid_inv_marg_u = np.flip(par.grid_Ctot)
        par.grid_marg_u_for_inv = np.nan + np.ones(par.grid_util.shape)

        par.grid_A_pd = nonlinspace(0.0,par.max_A,par.num_A_pd,1.1)

    def solve(self):
        sol = self.sol
        par = self.par 

        # setup grids
        self.setup_grids()

        # precompute the optimal intra-temporal consumption allocation for couples given total consumption
        for iP,power in enumerate(par.grid_power):
            for i,C_tot in enumerate(par.grid_Ctot):
                sol.pre_Ctot_Cw_priv[iP,i], sol.pre_Ctot_Cm_priv[iP,i], sol.pre_Ctot_C_pub[iP,i] = solve_intraperiod_couple(C_tot,power,par)

        # loop backwards
        for t in reversed(range(par.T)):
            self.solve_single(t)
            self.solve_couple(t)

        # total consumption
        sol.C_tot_couple = sol.Cw_priv_couple + sol.Cm_priv_couple + sol.C_pub_couple
        sol.C_tot_remain_couple = sol.Cw_priv_remain_couple + sol.Cm_priv_remain_couple + sol.C_pub_remain_couple
        sol.Cw_tot_single = sol.Cw_priv_single + sol.Cw_pub_single
        sol.Cm_tot_single = sol.Cm_priv_single + sol.Cm_pub_single

        # value of transitioning to singlehood. Done here because absorbing . it is the same as entering period as single.
        sol.Vw_trans_single = sol.Vw_single.copy()
        sol.Vm_trans_single = sol.Vm_single.copy()
        sol.Cw_priv_trans_single = sol.Cw_priv_single.copy()
        sol.Cm_priv_trans_single = sol.Cm_priv_single.copy()
        sol.Cw_pub_trans_single = sol.Cw_pub_single.copy()
        sol.Cm_pub_trans_single = sol.Cm_pub_single.copy()
        sol.Cw_tot_trans_single = sol.Cw_tot_single.copy()
        sol.Cm_tot_trans_single = sol.Cm_tot_single.copy()

    def solve_single(self, t):
        par = self.par
        sol = self.sol
        
        # loop through state variable: wealth
        for iN in range(par.num_n):  # addition of children
            for iA in range(par.num_A):  
                for iK in range(par.num_k):  # addition of human capital
                    
                    # index
                    idx = (t, iN, iA, iK)

                    for gender in ['woman', 'man']:
                        # Resources
                        A = par.grid_Aw[iA] if gender == 'woman' else par.grid_Am[iA]
                        kids = par.nw_grid[iN] if gender == 'woman' else par.nm_grid[iN]
                        K = par.kw_grid[iK] if gender == 'woman' else par.km_grid[iK]

                        if t == (par.T - 1):  # terminal period
                            obj = lambda x: self.obj_last_single(x[0], A, K, gender, kids)

                            # call optimizer
                            hours_min = np.fmax(-A / self.wage_func(K,gender) + 1.0e-5, 0.0)  # minimum amount of hours that ensures positive consumption
                            if iA == 0:
                                init_h = np.maximum(hours_min, 2.0)
                            else:
                                if gender == 'woman':
                                    init_h = np.array([sol.Hw_single[t, iN, iA - 1, iK]])
                                else:
                                    init_h = np.array([sol.Hm_single[t, iN, iA - 1, iK]])
                            
                            res = minimize(obj, init_h, bounds=((hours_min, np.inf),), method='L-BFGS-B')

                            # Store results separately for each gender
                            if gender == 'woman':
                                sol.Cw_priv_single[idx], sol.Cw_pub_single[idx] = self.cons_last_single(res.x[0], A, K, gender)
                                sol.Hw_single[idx] = res.x[0]
                                sol.Vw_single[idx] = -res.fun
                            else:
                                sol.Cm_priv_single[idx], sol.Cm_pub_single[idx] = self.cons_last_single(res.x[0], A, K, gender)
                                sol.Hm_single[idx] = res.x[0]
                                sol.Vm_single[idx] = -res.fun
                            
                        else:  # earlier periods
                            # search over optimal total consumption, C
                            obj = lambda x: -self.value_of_choice_single(x[0], x[1], A, K, kids, gender, t)
                            
                            # bounds on consumption 
                            lb_c = 0.000001  # avoid dividing with zero
                            ub_c = np.inf

                            # bounds on hours
                            lb_h = 0.0
                            ub_h = np.inf 

                            bounds = ((lb_c, ub_c), (lb_h, ub_h))
                
                            # call optimizer
                            idx_last = (t + 1, iN, iA, iK)
                            if gender == 'woman':
                                init = np.array([sol.Cw_priv_single[idx_last], sol.Hw_single[idx_last]])
                            else:
                                init = np.array([sol.Cm_priv_single[idx_last], sol.Hm_single[idx_last]])
                            
                            res = minimize(obj, init, bounds=bounds, method='L-BFGS-B', tol=1.0e-8) 

                            # Store results separately for each gender
                            if gender == 'woman':
                                sol.Cw_priv_single[idx], sol.Cw_pub_single[idx] = self.cons_single(res.x[0], A, K, gender, kids, t)
                                sol.Hw_single[idx] = res.x[1]
                                sol.Vw_single[idx] = -res.fun
                            else:
                                sol.Cm_priv_single[idx], sol.Cm_pub_single[idx] = self.cons_single(res.x[0], A, K, gender, kids, t)
                                sol.Hm_single[idx] = res.x[1]
                                sol.Vm_single[idx] = -res.fun

                                
                        
    def solve_couple(self,t):
        par = self.par
        sol = self.sol

        remain_Vw,remain_Vm,remain_Cw_priv,remain_Cm_priv,remain_C_pub = np.ones(par.num_power),np.ones(par.num_power),np.ones(par.num_power),np.ones(par.num_power),np.ones(par.num_power)
        
        Vw_next = None
        Vm_next = None
        for iL,love in enumerate(par.grid_love):
            for iA,A in enumerate(par.grid_A): # add human capital state variable!
                for iK1, K1 in enumerate(par.k_grid):
                    for iK2, K2 in enumerate(par.k_grid):

                        M_resources = resources_couple(A,par, K1, K2) 
                    
                        starting_val = None
                        for iP,power in enumerate(par.grid_power): # looop over different power levels!
                            # continuation values
                            if t<(par.T-1):
                                Vw_next = self.sol.Vw_couple[t+1,iP]
                                Vm_next = self.sol.Vm_couple[t+1,iP]

                            # starting values
                            if iP>0:
                                C_tot_last = remain_Cw_priv[iP-1] + remain_Cm_priv[iP-1] + remain_C_pub[iP-1]
                                starting_val = np.array([C_tot_last])
                        
                        # solve problem if remaining married
                        remain_Cw_priv[iP], remain_Cm_priv[iP], remain_C_pub[iP], remain_Vw[iP], remain_Vm[iP] = self.solve_remain_couple(t,M_resources,iL,iP,power,Vw_next,Vm_next,starting_val=starting_val)

                        # check the participation constraints - this applies the limited commitment bargaining scheme 
                        idx_single = (t,iA)
                        idx_couple = lambda iP: (t,iP,iL,iA)

                        list_start_as_couple = (sol.Vw_couple,sol.Vm_couple,sol.Cw_priv_couple,sol.Cm_priv_couple,sol.C_pub_couple)
                        list_remain_couple = (remain_Vw,remain_Vm,remain_Cw_priv,remain_Cm_priv,remain_C_pub)
                        list_trans_to_single = (sol.Vw_single,sol.Vm_single,sol.Cw_priv_single,sol.Cm_priv_single,sol.Cw_pub_single) # last input here not important in case of divorce
                    
                        Sw = remain_Vw - sol.Vw_single[idx_single] 
                        Sm = remain_Vm - sol.Vm_single[idx_single] 
                    
                        check_participation_constraints(sol.power_idx,sol.power,Sw,Sm,idx_single,idx_couple,list_start_as_couple,list_remain_couple,list_trans_to_single, par)

                        # save remain values
                        for iP,power in enumerate(par.grid_power): # looop over different power levels!
                            idx = (t,iP,iL,iA)
                            sol.Cw_priv_remain_couple[idx] = remain_Cw_priv[iP] 
                            sol.Cm_priv_remain_couple[idx] = remain_Cm_priv[iP]
                            sol.C_pub_remain_couple[idx] = remain_C_pub[iP]
                            sol.Vw_remain_couple[idx] = remain_Vw[iP]
                            sol.Vm_remain_couple[idx] = remain_Vm[iP]

    def solve_remain_couple(self,t,M_resources,iL,iP,power,Vw_next,Vm_next,starting_val = None):
        par = self.par

        if t==(par.T-1): # Terminal period
            C_tot = M_resources

        else:
            # objective function
            obj = lambda x: - self.value_of_choice_couple(x[0],t,M_resources,iL,iP,power,Vw_next,Vm_next)[0]
            x0 = np.array([M_resources * 0.8]) if starting_val is None else starting_val

            # optimize
            res = optimize.minimize(obj,x0,bounds=((1.0e-6, M_resources - 1.0e-6),) ,method='SLSQP') 
            C_tot = res.x[0]

        # implied consumption allocation (re-calculation)
        _, Cw_priv, Cm_priv, C_pub, Vw,Vm = self.value_of_choice_couple(C_tot,t,M_resources,iL,iP,power,Vw_next,Vm_next) # throw out value?

        # return objects
        return Cw_priv, Cm_priv, C_pub, Vw, Vm


    def value_of_choice_couple(self,C_tot,t,M_resources,iL,iP,power,Vw_next,Vm_next):
        sol = self.sol
        par = self.par

        love = par.grid_love[iL]
        
        # current utility from consumption allocation
        Cw_priv, Cm_priv, C_pub = intraperiod_allocation(C_tot,iP,sol,par)
        Vw = usr.util(Cw_priv,C_pub,woman,par,love)
        Vm = usr.util(Cm_priv,C_pub,man,par,love)
    
        # add continuation value
        if t < (par.T-1):
            # savings_vec = np.ones(par.num_shock_love)
            sol.savings_vec[:] = M_resources - C_tot #np.repeat(M_resources - C_tot,par.num_shock_love) np.tile(M_resources - C_tot,(par.num_shock_love,)) 
            love_next_vec = love + par.grid_shock_love

            linear_interp.interp_2d_vec(par.grid_love,par.grid_A , Vw_next, love_next_vec,sol.savings_vec,sol.Vw_plus_vec)
            linear_interp.interp_2d_vec(par.grid_love,par.grid_A , Vm_next, love_next_vec,sol.savings_vec,sol.Vm_plus_vec)

            EVw_plus = sol.Vw_plus_vec @ par.grid_weight_love
            EVm_plus = sol.Vm_plus_vec @ par.grid_weight_love

            Vw += par.beta*EVw_plus
            Vm += par.beta*EVm_plus

        # return
        Val = power*Vw + (1.0-power)*Vm
        return Val , Cw_priv, Cm_priv, C_pub, Vw,Vm
    
    def value_of_choice_single(self,C_tot,hours,assets,capital,kids,gender,t):

        # a. unpack
        par = self.par
        sol = self.sol

        #b. Specify consumption levels. 
        # flow-utility
        C_priv = usr.cons_priv_single(C_tot,gender,par)
        C_pub = C_tot - C_priv

        # b. penalty for violating bounds. 
        penalty = 0.0
        if C_tot < 0.0:
            penalty += C_tot*1_000.0
            C_tot = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = usr.util(C_priv, C_pub,hours,gender,kids, par)
        
        # d. *expected* continuation value from savings
        income = wage_func(self, capital, gender) * hours
        a_next = (1.0+par.r)*(assets + income - C_tot)
        k_next = capital + hours

        # no birth
        kids_next = kids
        V_next = sol.V[t+1,kids_next]
        V_next_no_birth = linear_interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        # birth
        if (kids>=(par.Nn-1)):
            # cannot have more children
            V_next_birth = V_next_no_birth

        else:
            kids_next = kids + 1
            V_next = sol.V[t+1,kids_next]
            V_next_birth = linear_interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        EV_next = par.p_birth * V_next_birth + (1-par.p_birth)*V_next_no_birth

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty
        
   
    def simulate(self):
        sol = self.sol
        sim = self.sim
        par = self.par

        for i in range(par.simN):
            for t in range(par.simT):

                # state variables
                if t==0:
                    A_lag = sim.init_A[i]
                    Aw_lag = sim.init_Aw[i]
                    Am_lag = sim.init_Am[i]
                    couple_lag = sim.init_couple[i]
                    power_idx_lag = sim.init_power_idx[i]
                    love = sim.love[i,t] = sim.init_love[i]

                else:
                    A_lag = sim.A[i,t-1]
                    Aw_lag = sim.Aw[i,t-1]
                    Am_lag = sim.Am[i,t-1]
                    couple_lag = sim.couple[i,t-1]
                    power_idx_lag = sim.power_idx[i,t-1]
                    love = sim.love[i,t]
                
                power_lag = par.grid_power[power_idx_lag]

                # first check if they want to remain together and what the bargaining power will be if they do.
                if couple_lag:                   

                    # value of transitioning into singlehood
                    Vw_single = linear_interp.interp_1d(par.grid_Aw,sol.Vw_trans_single[t],Aw_lag)
                    Vm_single = linear_interp.interp_1d(par.grid_Am,sol.Vm_trans_single[t],Am_lag)

                    idx = (t,power_idx_lag)
                    Vw_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love,A_lag)
                    Vm_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love,A_lag)

                    if ((Vw_couple_i>=Vw_single) & (Vm_couple_i>=Vm_single)):
                        power_idx = power_idx_lag

                    else:
                        # value of partnerhip for all levels of power
                        Vw_couple = np.zeros(par.num_power)
                        Vm_couple = np.zeros(par.num_power)
                        for iP in range(par.num_power):
                            idx = (t,iP)
                            Vw_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love,A_lag)
                            Vm_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love,A_lag)

                        # check participation constraint
                        Sw = Vw_couple - Vw_single
                        Sm = Vm_couple - Vm_single
                        power_idx = update_bargaining_index(Sw,Sm,power_idx_lag, par)

                    # infer partnership status
                    if power_idx < 0.0: # divorce is coded as -1
                        sim.couple[i,t] = False

                    else:
                        sim.couple[i,t] = True

                else: # remain single

                    sim.couple[i,t] = False

                # update behavior
                if sim.couple[i,t]:
                    
                    # optimal consumption allocation if couple
                    sol_C_tot = sol.C_tot_couple[t,power_idx] 
                    C_tot = linear_interp.interp_2d(par.grid_love,par.grid_A,sol_C_tot,love,A_lag)

                    sim.Cw_priv[i,t], sim.Cm_priv[i,t], C_pub = intraperiod_allocation(C_tot,power_idx,sol,par)
                    sim.Cw_pub[i,t] = C_pub
                    sim.Cm_pub[i,t] = C_pub

                    # update end-of-period states
                    M_resources = usr.resources_couple(A_lag,par) 
                    sim.A[i,t] = M_resources - sim.Cw_priv[i,t] - sim.Cm_priv[i,t] - sim.Cw_pub[i,t]
                    if t<(par.simT-1):
                        sim.love[i,t+1] = love + par.sigma_love*sim.draw_love[i,t+1]

                    # in case of divorce
                    sim.Aw[i,t] = par.div_A_share * sim.A[i,t]
                    sim.Am[i,t] = (1.0-par.div_A_share) * sim.A[i,t]

                    sim.power_idx[i,t] = power_idx
                    sim.power[i,t] = par.grid_power[sim.power_idx[i,t]]

                else: # single

                    # pick relevant solution for single, depending on whether just became single
                    idx_sol_single = t
                    sol_single_w = sol.Cw_tot_trans_single[idx_sol_single]
                    sol_single_m = sol.Cm_tot_trans_single[idx_sol_single]
                    if (power_idx_lag<0):
                        sol_single_w = sol.Cw_tot_single[idx_sol_single]
                        sol_single_m = sol.Cm_tot_single[idx_sol_single]

                    # optimal consumption allocations
                    Cw_tot = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw_lag)
                    Cm_tot = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am_lag)
                    
                    sim.Cw_priv[i,t],sim.Cw_pub[i,t] = intraperiod_allocation_single(Cw_tot,woman,par)
                    sim.Cm_priv[i,t],sim.Cm_pub[i,t] = intraperiod_allocation_single(Cm_tot,man,par)

                    # update end-of-period states
                    Mw = usr.resources_single(Aw_lag,woman,par)
                    Mm = usr.resources_single(Am_lag,man,par) 
                    sim.Aw[i,t] = Mw - sim.Cw_priv[i,t] - sim.Cw_pub[i,t]
                    sim.Am[i,t] = Mm - sim.Cm_priv[i,t] - sim.Cm_pub[i,t]

                    # not updated: nans
                    # sim.power[i,t] = np.nan
                    # sim.love[i,t+1] = np.nan 
                    # sim.A[i,t] = np.nan

                    sim.power_idx[i,t] = -1

        # total consumption
        sim.Cw_tot = sim.Cw_priv + sim.Cw_pub
        sim.Cm_tot = sim.Cm_priv + sim.Cm_pub
        sim.C_tot = sim.Cw_priv + sim.Cm_priv + sim.Cw_pub
    

############
# routines #
############
def intraperiod_allocation_single(C_tot,gender,par):
    C_priv = usr.cons_priv_single(C_tot,gender,par)
    C_pub = C_tot - C_priv
    return C_priv,C_pub
 
def intraperiod_allocation(C_tot,iP,sol,par):

    # interpolate pre-computed solution
    j1 = linear_interp.binary_search(0,par.num_Ctot,par.grid_Ctot,C_tot)
    Cw_priv = linear_interp_1d._interp_1d(par.grid_Ctot,sol.pre_Ctot_Cw_priv[iP],C_tot,j1)
    Cm_priv = linear_interp_1d._interp_1d(par.grid_Ctot,sol.pre_Ctot_Cm_priv[iP],C_tot,j1)
    C_pub = C_tot - Cw_priv - Cm_priv 

    return Cw_priv, Cm_priv, C_pub

def solve_intraperiod_couple(C_tot,power,par,starting_val=None):
    
    # setup estimation. Impose constraint that C_tot = Cw+Cm+C
    bounds = optimize.Bounds(0.0, C_tot, keep_feasible=True)
    obj = lambda x: - (power*usr.util(x[0],C_tot-np.sum(x),woman,par) + (1.0-power)*usr.util(x[1],C_tot-np.sum(x),man,par))
    
    # estimate
    x0 = np.array([C_tot/3,C_tot/3]) if starting_val is None else starting_val
    res = optimize.minimize(obj,x0,bounds=bounds)

    # unpack
    Cw_priv = res.x[0]
    Cm_priv = res.x[1]
    C_pub = C_tot - Cw_priv - Cm_priv

    return Cw_priv,Cm_priv,C_pub

def check_participation_constraints(power_idx,power,Sw,Sm,idx_single,idx_couple,list_couple,list_raw,list_single, par):
    
    # check the participation constraints. Array
    min_Sw = np.min(Sw)
    min_Sm = np.min(Sm)
    max_Sw = np.max(Sw)
    max_Sm = np.max(Sm)

    if (min_Sw >= 0.0) & (min_Sm >= 0.0): # all values are consistent with marriage
        for iP in range(par.num_power):

            # overwrite output for couple
            idx = idx_couple(iP)
            for i,key in enumerate(list_couple):
                list_couple[i][idx] = list_raw[i][iP]

            power_idx[idx] = iP
            power[idx] = par.grid_power[iP]

    elif (max_Sw < 0.0) | (max_Sm < 0.0): # no value is consistent with marriage
        for iP in range(par.num_power):

            # overwrite output for couple
            idx = idx_couple(iP)
            for i,key in enumerate(list_couple):
                list_couple[i][idx] = list_single[i][idx_single]

            power_idx[idx] = -1
            power[idx] = -1.0

    else: 
    
        # find lowest (highest) value with positive surplus for women (men)
        Low_w = 1 #0 # in case there is no crossing, this will be the correct value
        Low_m = par.num_power-1-1 #par.num_power-1 # in case there is no crossing, this will be the correct value
        for iP in range(par.num_power-1):
            if (Sw[iP]<0) & (Sw[iP+1]>=0):
                Low_w = iP+1
                
            if (Sm[iP]>=0) & (Sm[iP+1]<0):
                Low_m = iP

        # b. interpolate the surplus of each member at indifference points
        # women indifference
        id = Low_w-1
        denom = (par.grid_power[id+1] - par.grid_power[id])
        ratio_w = (Sw[id+1] - Sw[id])/denom
        ratio_m = (Sm[id+1] - Sm[id])/denom
        power_at_zero_w = par.grid_power[id] - Sw[id]/ratio_w
        Sm_at_zero_w = Sm[id] + ratio_m*( power_at_zero_w - par.grid_power[id] )

        # men indifference
        id = Low_m
        denom = (par.grid_power[id+1] - par.grid_power[id])
        ratio_w = (Sw[id+1] - Sw[id])/denom
        ratio_m = (Sm[id+1] - Sm[id])/denom
        power_at_zero_m = par.grid_power[id] - Sm[id]/ratio_m
        Sw_at_zero_m = Sw[id] + ratio_w*( power_at_zero_m - par.grid_power[id] )


        # update the outcomes
        for iP in range(par.num_power):

            # index to store solution for couple 
            idx = idx_couple(iP)
    
            # woman wants to leave
            if iP<Low_w: 
                if Sm_at_zero_w > 0: # man happy to shift some bargaining power

                    for i,key in enumerate(list_couple):
                        if iP==0:
                            list_couple[i][idx] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_zero_w,Low_w-1) 
                        else:
                            list_couple[i][idx] = list_couple[i][idx_couple(0)]; # re-use that the interpolated values are identical

                    power_idx[idx] = Low_w
                    power[idx] = power_at_zero_w
                    
                else: # divorce

                    for i,key in enumerate(list_couple):
                        list_couple[i][idx] = list_single[i][idx_single]

                    power_idx[idx] = -1
                    power[idx] = -1.0
                
            # man wants to leave
            elif iP>Low_m: 
                if Sw_at_zero_m > 0: # woman happy to shift some bargaining power
                    
                    for i,key in enumerate(list_couple):
                        if (iP==(Low_m+1)):
                            list_couple[i][idx] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_zero_m,Low_m) 
                        else:
                            list_couple[i][idx] = list_couple[i][idx_couple(Low_m+1)]; # re-use that the interpolated values are identical

                    power_idx[idx] = Low_m
                    power[idx] = power_at_zero_m
                    
                else: # divorce

                    for i,key in enumerate(list_couple):
                        list_couple[i][idx] = list_single[i][idx_single]

                    power_idx[idx] = -1
                    power[idx] = -1.0

            else: # no-one wants to leave

                for i,key in enumerate(list_couple):
                    list_couple[i][idx] = list_raw[i][iP]

                power_idx[idx] = iP
                power[idx] = par.grid_power[iP]

def update_bargaining_index(Sw,Sm,iP, par):
    
    # check the participation constraints. Array
    min_Sw = np.min(Sw)
    min_Sm = np.min(Sm)
    max_Sw = np.max(Sw)
    max_Sm = np.max(Sm)

    if (min_Sw >= 0.0) & (min_Sm >= 0.0): # all values are consistent with marriage
        return iP

    elif (max_Sw < 0.0) | (max_Sm < 0.0): # no value is consistent with marriage
        return -1

    else: 
    
        # find lowest (highest) value with positive surplus for women (men)
        Low_w = 0 # in case there is no crossing, this will be the correct value
        Low_m = par.num_power-1 # in case there is no crossing, this will be the correct value
        for _iP in range(par.num_power-1):
            if (Sw[_iP]<0) & (Sw[_iP+1]>=0):
                Low_w = _iP+1
                
            if (Sm[_iP]>=0) & (Sm[_iP+1]<0):
                Low_m = iP

        # update the outcomes
        # woman wants to leave
        if iP<Low_w: 
            if Sm[Low_w] > 0: # man happy to shift some bargaining power
                return Low_w
                
            else: # divorce
                return -1
            
        # man wants to leave
        elif iP>Low_m: 
            if Sw[Low_m] > 0: # woman happy to shift some bargaining power
                return Low_m
                
            else: # divorce
                return -1

        else: # no-one wants to leave
            return iP

def wage_func(self,capital,gender):
    # before tax wage rate
    par = self.par

    if gender == woman:
        constant = par.wage_const_1
        return_K = par.wage_K_1
    else:
        constant = par.wage_const_2
        return_K = par.wage_K_2
    return np.exp(constant + return_K * capital)

def child_tran(self,hours1,hours2,income_hh,kids):
    par = self.par
    if kids<1:
        return 0.0
    
    else:
        C1 = par.uncon_uni                           #unconditional, universal transfer (>0)
        C2 = np.fmax(par.means_level - par.means_slope*income_hh , 0.0) #means-tested transfer (>0)
        # child-care related (net-of-subsidy costs)
        both_work = (hours1>0) * (hours2>0)
        C3 = par.cond*both_work                      #all working couples has this net cost (<0)
        C4 = par.cond_high*both_work*(income_hh>0.5) #low-income couples do not have this net-cost (<0)

    return C1+C2+C3+C4

def resources_couple(self,par,A, capital1, capital2, hours1, hours2):
    par = par.self
    return par.R*A + wage_func(self,capital1,sex = 1)*hours1 + wage_func(self, capital2, sex = 2)*hours2

def resources_single(self,capital, hours, A, gender,par):
    income = wage_func(self, capital, sex = 1)*hours
    if gender == man:
        income = wage_func(self, capital, sex = 2)*hours

    return par.R*A + income

def obj_last_single(self,hours,assets,capital,gender,kids): #remember to add kids!
    par = self.par
    conspriv = self.cons_last_single(hours,assets,capital)[0]
    conspub = self.cons_last_single(hours,assets,capital)[1]
    
    return - usr.util(conspriv,conspub, hours, gender, kids, par)

def cons_last_single(self,hours,assets,capital, gender):
    #This returns C_priv, C_pub for singles in the last period
    par = self.par
    income = self.wage_func(self,capital,gender)*hours
    #Consume everything in the last period
    C_tot = assets + income
    conspriv = usr.cons_priv_single(C_tot, gender, par)
    conspub = C_tot - conspriv

    return conspriv, conspub
    
