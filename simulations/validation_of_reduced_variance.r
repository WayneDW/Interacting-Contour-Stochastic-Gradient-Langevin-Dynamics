library(scales)

# set your folder
setwd("//Users//wayne//Documents//My research//0. Pop CSGLD//")


# total number of iterations
total_iters = 1e6
# learning rate for gradient Langevin dynamics
eta = 0.1 
# temperature
T = 1
# hyperparameter
zeta = 0.9
# Total number of partitions
parts = 100
# Energy gap
div = 1
# Minimum energy for partition
min_energy = 0.15

# build the energy function and the gradient
energy = function(x) return(-log(0.4 * dnorm(x, -6, 1) + 0.6 * dnorm(x, 4, 1)))
grad_f = function(x) return(numDeriv::grad(energy, x) + 0.03 * rnorm(1, 0, 1))

# find the index for new samples
find_idx = function(sample) {
    stochastic_energy = energy(sample) + 0.03 * rnorm(1, 0, 1)
    return(min((as.integer((energy(sample) - min_energy) / div) + 1), parts-1))
}

random_field = function(theta, idx) {
    sub_random_field = -theta
    sub_random_field[idx] = sub_random_field[idx] + 1
    return(theta[idx] * sub_random_field)
}

# print process
print_progress = function(iter, total_iters) {
    if (iter %% (total_iters / 10) == 0) {
        print(paste('Iter ', scientific(iter, digits = 3), '/',  total_iters,' completed'))
    }
}


# Since we are converging to a biased equilibrium that approximates the ground truth, we will not compare it and will only
# focus on validate the reduction of variance
#############################################     Ground truth density of states  #########################################
parts_for_presentation = 11
# simulate samples from the Gaussian mixture distribution
real_samples = c(rnorm(1e6 * 0.4, -6, 1), rnorm(1e6 * 0.6, 4, 1))
# obtain the energy value for these samples
energy_each_sample = mapply(energy, real_samples)

# establish energy partition
grids = seq(1, parts_for_presentation) * div + min_energy
# compute density of states (ignore the first partition)
real_density_of_states = rep(0, (parts_for_presentation-1))
for (i in 1:(parts_for_presentation-1)) {
    real_density_of_states[i]  = (sum(energy_each_sample <= grids[i+1]) - sum(energy_each_sample <= grids[i])) / 1e6
}



chains = 10
#########################################    ICSGLD  (many short chains)    ##############################################
theta_ICSGLD = c()
candidate_seeds = seq(1, 10)
for (seeds in candidate_seeds) {
    set.seed(seeds) 
    print(paste("Run ICSGLD (many short chains) based on the random seed", seeds))
    theta = rep(1, parts) / parts
    samples = rep(0, chains)
    indexs = rep(1, chains)
    for (iter in 1: total_iters) { 
        decay = 1 / (iter^0.6 + 100)
        print_progress(iter, total_iters)
        # adaptive sampling part (parallelizable)
        for (cidx in 1: chains) {
            multiplier = 1 + zeta * T * (log(theta[indexs[cidx]+1]) - log(theta[indexs[cidx]])) / div 
            samples[cidx] = samples[cidx] - eta * multiplier * grad_f(samples[cidx])  + sqrt(2 * eta * T) * rnorm(1, 0, 1)
            indexs[cidx] = find_idx(samples[cidx])
        }
        # interacting stochastic approximation
        interacting_random_field = 0
        for (cidx in 1: chains) {
            interacting_random_field = interacting_random_field + random_field(theta, indexs[cidx]) / chains
        }
        theta = theta + decay * interacting_random_field
    }
    # ignore the first empty partition where the minimum energy is around 1.42
    theta_ICSGLD = cbind(theta_ICSGLD, theta[2:parts_for_presentation])
}

#########################################    CSGLD  (a single long chain)    ##############################################
theta_CSGLD = c()
for (seeds in candidate_seeds) {
    set.seed(seeds) 
    print(paste("Run CSGLD (a single long chain) based on the random seed", seeds))
    theta = rep(1, parts) / parts
    sample = 0
    idx = 1
    for (iter in 1: (total_iters*chains)) { 
        decay = 1 / (iter^0.6 + 100)
        print_progress(iter, total_iters*chains)
        # adaptive sampling part
        multiplier = 1 + zeta * T * (log(theta[idx+1]) - log(theta[idx])) / div 
        sample = sample - eta * multiplier * grad_f(sample)  + sqrt(2 * eta * T) * rnorm(1, 0, 1)
        idx = find_idx(sample)
        theta = theta + decay * random_field(theta, idx)
    }
    # ignore the first empty partition where the minimum energy is around 1.42
    theta_CSGLD = cbind(theta_CSGLD, theta[2:parts_for_presentation])
}



library(ggplot2)

ICSGLD_estimated_density_of_states = rowMeans(theta_ICSGLD)^zeta / sum(rowMeans(theta_ICSGLD)^zeta)
CSGLD_estimated_density_of_states = rowMeans(theta_CSGLD)^zeta / sum(rowMeans(theta_CSGLD)^zeta)

# we don't know the real equilibrium, we approximate it instead
biased_equilibrium = (ICSGLD_estimated_density_of_states + CSGLD_estimated_density_of_states) / 2

wdata = cbind(CSGLD_estimated_density_of_states, ICSGLD_estimated_density_of_states, biased_equilibrium)
ddf = as.data.frame(wdata)
ddf$X = seq(1, length(ICSGLD_estimated_density_of_states))
ddf$ICSGLD_theta_upper = ddf$ICSGLD_estimated_density_of_states + 2 * apply(theta_ICSGLD, 1, sd) 
ddf$ICSGLD_theta_lower = ddf$ICSGLD_estimated_density_of_states - 2 * apply(theta_ICSGLD, 1, sd)
ddf$CSGLD_theta_upper = ddf$CSGLD_estimated_density_of_states + 2 * apply(theta_CSGLD, 1, sd) 
ddf$CSGLD_theta_lower = ddf$CSGLD_estimated_density_of_states - 2 * apply(theta_CSGLD, 1, sd) 


sub_p = ggplot(ddf, aes(X)) + 
    geom_line(aes(y=ICSGLD_estimated_density_of_states, col="#7570B3"), size=0.5) + 
    geom_line(aes(y=CSGLD_estimated_density_of_states, col="#FF9999"), size=0.5) + 
    geom_line(aes(y=biased_equilibrium, col="black"), size=1) +  
    geom_ribbon(aes(ymin=ICSGLD_theta_lower,ymax=ICSGLD_theta_upper, x=X),alpha=0.3, fill="#7570B3") + 
    geom_ribbon(aes(ymin=CSGLD_theta_lower,ymax=CSGLD_theta_upper, x=X),alpha=0.3, fill="#FF9999") + 
    scale_x_continuous(name="Energy") +
    scale_y_continuous(name="PDF in Energy (density of states)") + 
    scale_color_identity(name = "Multipliers",
                       breaks = c("#7570B3", "#FF9999", "black"),
                       labels = c("ICSGLD xP10", "CSGLD xT10", bquote(hat(theta)["*"])),
                       guide = "legend")  +
    theme(legend.position=c(0.8, 0.2),
        legend.title = element_blank(),
        axis.title=element_text(size=20),
        legend.text = element_text(colour="grey15", size=24),
        legend.key.size = unit(2,"line"),
        legend.key = element_blank(),
        legend.background = element_rect(fill=alpha('grey', 0.1)),
        axis.text.y = element_text(size=16),
        axis.text.x = element_text(size=16))

print(sub_p)


