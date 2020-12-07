import numpy as np
import networkx as nx
import random
import copy
import matplotlib
matplotlib.use('agg')
import pylab as pl
import pickle as pkl
import configparser
from multiprocessing import Pool, Process, sharedctypes, current_process
from functools import partial
import math
import time
import datetime
import os.path
import matplotlib.pyplot as plt
import RAMchecker
import os.path
import os
import sys
import gc
from types import ModuleType, FunctionType
from gc import get_referents
from telepy import telegram_bot_sendtext
import itertools


class Params:
    """
    Parameter object, containing all simulation/model parameters
    """

    def __init__(self, L=None, N=4, bouts=1, poolsize=1, outstep=10,
                 res_value=2.0,
                 init_c_rep=0.0, init_c_upt=0.0, init_c_sig=0.0, base_upt_rate=1, base_sig_prob=0.0,
                 init_pheno_distr=1, inputpath='./',
                 factor_rep=1.0, factor_upt=1.0, factor_sig=0.0, sigma_mut=0.1, no_gens=10, traitmin=0.0, traitmax=1.0,
                 str_of_selection=0.9, init_placement=0, sig_range=2,
                 res_persistence_time=1., fraction_covered=0.5, normalize=0, gkernel_range=1.0,
                 prob_of_fight=1.0, beta_competition=5.0, maxiterations=50, debug=False, finding_bias=1.0,
                 connectivity=0.2, visual_debug=0, mobility=0, evolution_rate=5, resource_amount=150):
        self.L = L  # lattice size (L x L)
        if self.L is None:
            self.L = 20

        self.N = N  # number of agents / phenotypes to be evolved
        self.bouts = bouts  # number of bouts per generation
        self.poolsize = poolsize  # size of the parallel processing pool (number of threads to be run in parallel)
        self.outstep = outstep  # output frequency (generations)
        self.res_value = res_value  # resource value (increase in fitness benefits per unit resource)
        self.init_c_rep = init_c_rep  # initial investment in repulsion
        self.init_c_upt = init_c_upt  # initial investment in uptake
        self.init_c_sig = init_c_sig  # initial investment in signaling
        self.factor_rep = factor_rep  # factor for increase in rep. strength per unit investment
        self.factor_upt = factor_upt  # factor for increase in upt. strength per unit investment
        self.factor_sig = factor_sig  # factor for increase in sig. strength per unit investment
        self.base_upt_rate = base_upt_rate  # base uptake rate
        self.base_sig_prob = base_sig_prob  # base signaling probability
        self.init_pheno_distr = init_pheno_distr  # flag setting initial phenotype distribution
        self.inputpath = inputpath  # path for reading in phenotype distribution (if init_pheno_distr=99)
        self.traitmin = traitmin  # minimum allowed value of traits (investments)
        self.traitmax = traitmax  # maximum allowed value of traits (investments)
        self.str_of_selection = str_of_selection  # strength of selection parameter
        self.init_placement = init_placement  # flag setting initial placement (not used currently)
        self.sigma_mut = sigma_mut  # mutation strength (std deviation of trait mutation)
        self.no_gens = no_gens  # total number of generations
        self.fraction_covered = fraction_covered  # fraction of area covered by resources
        self.finding_bias = finding_bias  # bias to be placed on resource patch (between 0 - random and 1 - only in
        # resource patches)
        self.normalize = normalize  # normalization of the resource amount with respect to area
        self.sig_range = sig_range
        if self.L < gkernel_range:  # Gaussian kernel size for generation of spatially correlated res. fields
            print("gkernel range to large setting to L")
            self.gkernel_range = self.L
        else:
            self.gkernel_range = gkernel_range

        self.res_persistence_time = res_persistence_time  # resource lifetime
        self.prob_of_fight = prob_of_fight  # probability of fight
        self.debug = debug  # debug flag for additional output
        self.maxiterations = maxiterations  # max. iterations for calculating displacement due to repulsion
        self.beta_competition = beta_competition  # steepness of the sigmoidal function for displacement
        self.connectivity = connectivity
        self.visual_debug = visual_debug
        self.mobility = mobility
        self.evolution_rate = evolution_rate
        self.resource_amount = resource_amount

    def __repr__(self):
        return "Params()"


class Agent:
    """
    Agent object, simulated individuum with all characteristics
    """

    def __init__(self, idx=0, cost_rep=0.0, cost_upt=0.0, cost_sig=0.0, rep_strength=0, sig_prob=0.0, sig_range=2,
                 uptake_rate=0.0, benefits=0.0, perception=0.25, age=1):

        self.idx = idx  # identification index of the agent
        self.pos = -1   # identifier of the patch the agent is currently located on
        self.cost_rep = cost_rep  # investment in repulsion
        self.cost_upt = cost_upt  # investment in uptake
        self.cost_sig = cost_sig  # investment in signaling
        self.rep_strength = rep_strength  # repulsion strength
        self.sig_prob = sig_prob  # probability of signalling
        self.sig_range = sig_range  # signaling range
        self.perception = perception
        self.uptake_rate = uptake_rate  # uptake rates
        self.benefits = benefits  # fitness benefits
        self.age = age  # Lifetime of the agent counted in actions the agent performed
        self.fitness_rate = 0  # Quotient of benefits divided by age

    def __repr__(self):
        return 'ag%d:  %2.2f,\t %2.2f,\t %2.2f,\t %2.2f' % (
            self.idx, self.cost_rep, self.cost_sig,
            self.cost_upt, self.benefits)

    def UpdatePhenotype(self, params_):
        """
        Update phenotype, by renormalizing the investment vector, and updating corresponding behavioral parameters
        :param params_:
        :return:
        """
        traitsum = 0.0

        if params_.factor_rep > 0.0:
            traitsum += self.cost_rep
        else:
            self.cost_rep = 0.0

        if params_.factor_upt > 0.0:
            traitsum += self.cost_upt
        else:
            self.cost_upt = 0.0

        if params_.factor_sig > 0.0:
            traitsum += self.cost_sig
        else:
            self.cost_sig = 0.0

        if traitsum > 0.0:
            self.cost_rep /= traitsum
            self.cost_upt /= traitsum
            self.cost_sig /= traitsum

        self.rep_strength = self.cost_rep * params_.factor_rep
        self.uptake_rate = self.cost_upt * params_.factor_upt + params_.base_upt_rate
        self.sig_prob = self.cost_sig * params_.factor_sig + params_.base_sig_prob
        self.benefits = 0.0
        self.age = 1
        self.fitness_rate = 0


class Patches:
    """
    Patches obejct, containing resources, agents and signal strength.
    """

    def __init__(self, idp, ressource=0, signal=0):
        self.idp = idp
        self.agentlist = set()
        self.ressource = ressource
        self.signal = signal

    def __repr__(self):
        return 'ptch-%d' % (self.idp)

    def info(self):
        return 'ptch-%d, %2.2f, %2.2f' % (self.idp, self.ressource, self.signal)


class Universe:
    """
    World object, contains the network and generates and simulates the agents on the network.
    All actions for the patches and agents and their interaction are defined here.
    """


    def __init__(self, agents=None, L=10, N=4, bouts=1, poolsize=1, outstep=10,
                 res_value=2.0,
                 init_c_rep=0.0, init_c_upt=0.0, init_c_sig=0.0, base_upt_rate=1, base_sig_prob=0.0,
                 init_pheno_distr=1, inputpath='./',
                 factor_rep=1.0, factor_upt=1.0, factor_sig=0.0, sigma_mut=0.1, no_gens=10, traitmin=0.0, traitmax=1.0,
                 str_of_selection=0.9, init_placement=0,
                 res_persistence_time=1., fraction_covered=0.2, normalize=0,
                 prob_of_fight=1.0, beta_competition=5.0, maxiterations=50, debug=False, finding_bias=1.0,
                 patches=None, adjmatrix=None, params_=None, connectivity=15, pos=None, visual_debug=0, g=0, pid=0,
                 mobility=0, safepath="./", evolution_rate=5, sig_range=2, resource_amount=150):

        if not params_ is None:

            self.P = params_.L  # number of patches

            self.N = params_.N  # number of agents / phenotypes to be evolved
            self.bouts = params_.bouts  # number of bouts per generation
            self.poolsize = params_.poolsize  # size of the parallel processing pool (number of threads to be run in
            # parallel)
            self.outstep = params_.outstep  # output frequency (generations)
            self.res_value = params_.res_value  # resource value (increase in fitness benefits per unit resource)
            self.init_c_rep = params_.init_c_rep  # initial investment in repulsion
            self.init_c_upt = params_.init_c_upt  # initial investment in uptake
            self.init_c_sig = params_.init_c_sig  # initial investment in signaling
            self.factor_rep = params_.factor_rep  # factor for increase in rep. strength per unit investment
            self.factor_upt = params_.factor_upt  # factor for increase in upt. strength per unit investment
            self.factor_sig = params_.factor_sig  # factor for increase in sig. strength per unit investment
            self.base_upt_rate = params_.base_upt_rate  # base uptake rate
            self.base_sig_prob = params_.base_sig_prob  # base signaling probability
            self.init_pheno_distr = params_.init_pheno_distr  # flag setting initial phenotype distribution
            self.inputpath = params_.inputpath  # path for reading in phenotype distribution (if init_pheno_distr=99)
            self.traitmin = params_.traitmin  # minimum allowed value of traits (investments)
            self.traitmax = params_.traitmax  # maximum allowed value of traits (investments)
            self.str_of_selection = params_.str_of_selection  # strength of selection parameter
            self.init_placement = params_.init_placement  # flag setting initial placementn (not used currently)
            self.sigma_mut = params_.sigma_mut  # mutation strength (std deviation of trait mutation)
            self.no_gens = params_.no_gens  # total number of generations
            self.fraction_covered = params_.fraction_covered  # fraction of area covered by resources
            self.finding_bias = params_.finding_bias  # bias to be placed on resource patch (between 0 - random and 1
            # - only in resource patches)
            self.normalize = params_.normalize  # normalization of the resource amount with respect to area
            self.sig_range = params_.sig_range

            self.res_persistence_time = params_.res_persistence_time  # resource lifetime
            self.prob_of_fight = params_.prob_of_fight  # probability of fight
            self.debug = params_.debug  # debug flag for additional output
            self.maxiterations = params_.maxiterations  # max. iterations for calculating displacement due to repulsion
            self.beta_competition = params_.beta_competition  # steepness of the sigmoidal function for displacement
            self.connectivity = params_.connectivity
            self.evolution_rate = params_.evolution_rate
            self.resource_amount = params_.resource_amount

            self.visual_debug = params_.visual_debug
            self.params_ = params_

        else:
            self.P = L
            self.N = N  # number of agents / phenotypes to be evolved
            self.bouts = bouts  # number of bouts per generation
            self.poolsize = poolsize  # size of the parallel processing pool (number of threads to be run in parallel)
            self.outstep = outstep  # output frequency (generations)
            self.res_value = res_value  # resource value (increase in fitness benefits per unit resource)
            self.init_c_rep = init_c_rep  # initial investment in repulsion
            self.init_c_upt = init_c_upt  # initial investment in uptake
            self.init_c_sig = init_c_sig  # initial investment in signaling
            self.factor_rep = factor_rep  # factor for increase in rep. strength per unit investment
            self.factor_upt = factor_upt  # factor for increase in upt. strength per unit investment
            self.factor_sig = factor_sig  # factor for increase in sig. strength per unit investment
            self.base_upt_rate = base_upt_rate  # base uptake rate
            self.base_sig_prob = base_sig_prob  # base signaling probability
            self.init_pheno_distr = init_pheno_distr  # flag setting initial phenotype distribution
            self.inputpath = inputpath  # path for reading in phenotype distribution (if init_pheno_distr=99)
            self.traitmin = traitmin  # minimum allowed value of traits (investments)
            self.traitmax = traitmax  # maximum allowed value of traits (investments)
            self.str_of_selection = str_of_selection  # strength of selection parameter
            self.init_placement = init_placement  # flag setting initial placement (not used currently)
            self.sigma_mut = sigma_mut  # mutation strength (std deviation of trait mutation)
            self.no_gens = no_gens  # total number of generations
            self.fraction_covered = fraction_covered  # fraction of area covered by resources
            self.finding_bias = finding_bias  # bias to be placed on resource patch (between 0 - random and 1 - only
            # in resource patches)
            self.normalize = normalize  # normalization of the resource amount with respect to area
            self.sig_range = sig_range



            self.res_persistence_time = res_persistence_time  # resource lifetime
            self.prob_of_fight = prob_of_fight  # probability of fight
            self.debug = debug  # debug flag for additional output
            self.maxiterations = maxiterations  # max. iterations for calculating displacement due to repulsion
            self.beta_competition = beta_competition  # steepness of the sigmoid function for displacement
            self.connectivity = connectivity
            self.visual_debug = visual_debug
            self.mobility = mobility
            self.evolution_rate = evolution_rate
            self.memory_usage = 0

            self.resource_amount = 200

        self.hist_array = np.zeros((self.N))

        self.adjmatrix = adjmatrix
        if self.adjmatrix is None:
            self.adjmatrix = create_adjacency_matrix(self.connectivity, self.P)

        self.pos = pos
        if self.pos is None:
            self.pos = nx.spring_layout(nx.from_numpy_matrix(np.transpose(self.adjmatrix)))

        self.g = g
        self.pid = pid

        self.debug_string = ""
        self.rumble_count = 0
        self.max_rumble_count = self.N * 10

        self.use_debug_file = False

        self.safepath = safepath

        # assign the passed patches to this universe or initiate new patches if no patches were passed
        self.patches = patches
        if self.patches is None:
            self.patches = []
            for ip in range(self.P):
                self.patches.append(Patches(ip, ressource=0))

        self.list_idps = list(patch.idp for patch in self.patches)

        # dictionary{patch-id: patch object}
        self.id_to_patch = dict.fromkeys(self.list_idps)
        for patch in self.patches:
            self.id_to_patch[patch.idp] = patch

        self.replenish_patches()

        # in case of non-symmetric transpose adjancy matrix to get the allowed paths
        self.inv_adj_matrix = np.transpose(self.adjmatrix)

        # create Graph from adjancy matrix
        self.world_graph = nx.from_numpy_matrix(np.transpose(self.adjmatrix))

        # calculate all distances between all patches and safe them to a dictionary
        self.distance_dictionary = dict(nx.all_pairs_shortest_path_length(self.world_graph))

        # assign the passed agents to this Universe or initiate new agents if no agents were passed
        self.agents = agents
        if self.agents is None:
            self.agents = self.Initiate_Agents()

        # create dictionary to access agents by their ID
        self.agentdict = dict.fromkeys([agent.idx for agent in self.agents])

        for agent in self.agents:
            self.agentdict[agent.idx] = agent

        # dictionary{patch: List of valid patch-ids Agents can move to from this patch}
        self.neighbor_list = dict.fromkeys(self.patches)
        for idn in range(self.P):
            neighbors = (np.where(self.adjmatrix[:, idn] == 1)[0]).tolist()
            self.neighbor_list[self.id_to_patch[idn]] = neighbors

        # Dictionary that declares not allowed paths in non-symmetric network
        self.anti_neighbor_list = dict.fromkeys(self.list_idps)
        for idp in self.list_idps:
            anti_neighbors = (np.where(self.inv_adj_matrix[:, idp] == 1)[0]).tolist()
            self.anti_neighbor_list[idp] = anti_neighbors

        # places the agents initially, either random or predefined
        if self.init_placement in ["random", 1]:
            self.random_agent_placement()
        else:

            self.defined_agent_placement()

        #create a folder to store information about the agents
        date = datetime.datetime.now()
        date = str(date)[:16].replace(" ", "_").replace(":", "_")

        date = "/home/winkler/Pictures/masterproject/agents/" + unique_file(date, "/")[:-2] + "/"
        #print(self.init_pheno_distr)

        os.mkdir(date)

        self.agent_path = date

        self.fig1, self.ax1 = plt.subplots(1, 1, figsize=(15, 15))
        self.ax1.set_axis_off()

        print(self.resource_amount)

    def replenish_patches(self):
        """
        Function to chose random patches to put resources on
        """

        number_of_res_patches = math.ceil(self.P * self.fraction_covered)

        patch_ids = [patchx.idp for patchx in self.patches]

        for i in range(number_of_res_patches):
            patchid = random.choice(patch_ids)

            self.id_to_patch[patchid].ressource = self.resource_amount

            patch_ids.remove(patchid)

    def random_agent_placement(self):
        """
        Removes all agents from all patches and places all agents on random patches again
        """
        for patch in self.patches:
            patch.agentlist = set()

        for agentx in self.agents:
            random_patch = random.choice(self.patches)
            random_patch.agentlist.add(agentx)
            agentx.pos = random_patch.idp

    def defined_agent_placement(self):
        """
        Places all the agents on patch 0
        """
        for agentx in self.agents:
            self.patches[0].agentlist.add(agentx)
            agentx.pos = self.patches[0].idp

    def update_patch_agentlists(self):
        """
        Function to update the lists of agents for all patches
        """
        for patch in self.patches:
            patch.agentlist = set()

        for agent in self.agents:
            self.id_to_patch[agent.pos].agentlist.add(agent)

    def move_one_agent_random(self, agentx: Agent):
        """
        Moves an agent to a random valid patch, if another agent is on the patch and other conditions meet as well,
        the agent fights all other agents on that patch

        :param agentx: agent that gets moved

        """

        self.debug_string += "MOVING RANDOM \n"

        pos = agentx.pos
        # noinspection PyTypeChecker

        # check if agent moves at all (mobility not implemented yet) and if the agent can move
        if random.random() > 0 and len(self.neighbor_list[self.id_to_patch[pos]]) >= 1:

            # get new position of the agent
            new_pos = random.choice(self.neighbor_list[self.id_to_patch[pos]])

            # change the position of the agent
            agentx.pos = new_pos

            # remove the agent from the agentlist of the old position and reassign it to the new position
            if agentx in self.id_to_patch[pos].agentlist:
                self.id_to_patch[pos].agentlist.remove(agentx)
            self.id_to_patch[new_pos].agentlist.add(agentx)

            self.debug_string += "ag" + str(agentx.idx) + " moved to patch" + str(agentx.pos) + "\n"

            # if another agent is allrdy on the new patch and it contains ressources initiate fighting
            if len(self.id_to_patch[agentx.pos].agentlist) > 1 and self.id_to_patch[agentx.pos].ressource > 0:
                self.debug_string += "new patch ressource: " + str(self.id_to_patch[agentx.pos].ressource) + "\n"
                self.fighting(agentx, self.id_to_patch[agentx.pos].agentlist)


        # if conditions are met agent signals other agents to come to the patch
        if agentx.sig_prob > random.random() and self.id_to_patch[agentx.pos].ressource > 0 \
                and self.id_to_patch[agentx.pos].signal < 1:
            self.signalling(agentx)

    def move_one_agent_defined(self, agentx: Agent, patchx: Patches):
        """
        moves an agent to the desired patch

        :param agentx: agent that will be moved
        :param patchx: patch the agents moves to
        """

        self.debug_string += "MOVING DEFINED \n"

        pos = agentx.pos

        new_pos = patchx.idp

        agentx.pos = new_pos

        self.id_to_patch[pos].agentlist.remove(agentx)
        self.id_to_patch[new_pos].agentlist.add(agentx)

        self.debug_string += "ag" + str(agentx.idx) + " moved to patch" + str(agentx.pos) + "\n"

    def uptake(self, agentx: Agent):
        """
        Calculates the amount of ressource the given agent harvests from the patch he is on.
        Reduce the patches ressource by that amount and improve the agents fitness by:
        (ressource value) x (harvested amount)

        :param agentx:
        """

        self.debug_string += "UPTAKING \n"

        patch = self.id_to_patch[agentx.pos]

        if patch.ressource > agentx.uptake_rate:
            ressource_change = agentx.uptake_rate
        else:
            ressource_change = patch.ressource

        agentx.benefits += self.res_value * ressource_change
        patch.ressource -= ressource_change

        agentx.fitness_rate = agentx.benefits / agentx.age
        # //TODO: change it back on

        self.debug_string += "ag" + str(agentx.idx) + " uptaking: " + str(ressource_change) + " ressources" + \
                             "\n" + "agent got {} benefits".format(agentx.benefits) + "\n"

    def fighting(self, agentx: Agent, agentlist: list):
        """
        Decides which agents the given agent has to fight of the agents given in the list.

        :param agentx: Agent
        :param agentlist: List of Agents
        """

        # create call-by-value list to avoid moving or removing agents from the original list
        fightlist = [agent.idx for agent in agentlist]

        if agentx.idx in fightlist:
            fightlist.remove(agentx.idx)

        self.debug_string += "FIGHTING \n" + "ag" + str(agentx.idx) + " fights agents: " + str(fightlist) + \
                             " on patch" + str(agentx.pos) + "\n"

        # let agentx fight all agents in the fightlist
        for id_ in fightlist:

            agenty = self.agentdict[id_]

            # check if fight even starts (avoid fighting of signaling agents)
            eff_prob_of_fight = self.prob_of_fight - 0.5 * (agenty.sig_prob + agentx.sig_prob)

            if eff_prob_of_fight > random.random() and agenty.pos == agentx.pos:

                # calculate strength difference of fighting agents
                delta = agentx.rep_strength - agenty.rep_strength

                self.debug_string += "ag" + str(agentx.idx) + "fights " + "ag" + str(agenty.idx) + "\n"

                # assign agents depending on their strength relative to each other to determine the winning probability
                if delta > 0:
                    higher_str = agentx
                    lower_str = agenty
                else:
                    higher_str = agenty
                    lower_str = agentx

                # roll winning prob and check for the value it has to overcome so the weaker agent wins
                # the beta
                winning_prob = random.random()
                beta_value = competition_function(np.abs(delta), self.beta_competition)

                # if winning probability is higher than beta value, the weaker agent wins
                if winning_prob > beta_value:
                    winner = lower_str
                    loser = higher_str
                else:
                    winner = higher_str
                    loser = lower_str

                self.debug_string += "ag" + str(winner.idx) + " won over ag" + str(loser.idx) + "\n"

                random_patch = random.choice([patchx for patchx in self.patches if patchx.ressource == 0])

                #self.move_one_agent_defined(loser, random_patch)

                self.move_one_agent_random(loser)

                # if agentx loses a fight, all other following fights should not happen
                if loser.idx == agentx.idx:
                    break

            else:

                self.debug_string += "NO Fighting \n"

    def initial_fighting(self):
        """
        Checks all occupied patches with 2 or more agents and let these agents fight against each other initially.
        """

        self.debug_string += "------- Initial Fighting ------- \n \n"

        # get all patches with more than one agent
        fight_patches = [patch for patch in self.patches if len(patch.agentlist) > 1 and patch.ressource > 0]

        self.debug_string += "These are the patches with more than 1 agent and ressources: \n"

        for patchy in fight_patches:
            self.debug_string += patchy.info() + "\n \n"

        for patchy in fight_patches:
            self.rumble_patch(patchy)

        self.rumble_count = 0

    def rumble_patch(self, patchx: Patches):
        """
        Function that let all agents fight against all other agents on the given patch
        :param patchx: Type = Patches
        """

        # create a list of agent-ids of agents that can fight
        fight_list = [agent.idx for agent in patchx.agentlist]

        # shuffle the list to avoid that some agents will always fight before others do and get a disadvantage
        random.shuffle(fight_list)

        # create a list where all agent-ids are stored of agents that allrdy did fought against all agents
        list_of_used_agents = []

        # pick an agent and let it fight against
        for ida in fight_list:

            if not self.agentdict[ida].pos == patchx.idp:
                continue

            # avoid that the fighting agent fights against itself
            if ida in fight_list:
                fight_list.remove(ida)

            # create list of agents that have to fight against agent with id == 'ida'
            agent_fight_list = [self.agentdict[idb] for idb in fight_list]

            # the actual fighting!
            self.fighting(self.agentdict[ida], agent_fight_list)

            # update the list of agents that fought against all others
            list_of_used_agents.append(ida)

            # update fightlist according to results from last fight round
            fight_list = [agent.idx for agent in patchx.agentlist]

            for idc in list_of_used_agents:
                if idc in fight_list:
                    fight_list.remove(idc)

    def signalling(self, agentx: Agent):
        """

        :param agentx: agent that sends out the signal to other agent, that he has ressources.
        :return:
        """

        self.debug_string += "SIGNALLING \n" + "ag" + str(agentx.idx) + " signals" + "\n"

        sig_range = agentx.sig_range

        # get list of signal receiving patches
        sig_receiving_patches = [k for k, v in self.distance_dictionary[agentx.pos].items() if float(v) <= sig_range]

        # set signal strength of original patch to 1
        self.id_to_patch[agentx.pos].signal = 1

        # for every patch in the list of signal receiving patches: set signal strength depending on the distance
        for i in range(len(sig_receiving_patches)):
            patchy = self.id_to_patch[sig_receiving_patches[i]]
            patchy.signal = 2 ** (-(self.distance_dictionary[agentx.pos][patchy.idp]))

        self.debug_string += "on these patches: " + str(sig_receiving_patches) + "\n"

    def initial_signalling(self):
        """
        Initial signaling for all agents with high enough signal probability
        :return:
        """

        self.debug_string += "------- Initial signalling ------- \n \n"

        # for every agent, check if signalling conditions are met and if its the case -> spread signal
        for agenty in self.agents:

            patchy = self.id_to_patch[agenty.pos]

            if random.random() < agenty.sig_prob and patchy.signal < 1 and patchy.ressource > 0:
                self.signalling(agenty)

        # for every agent, follow the signal if your own patch doesnt contain ressources but you sense a signal
        for agenty in self.agents:

            patchy = self.id_to_patch[agenty.pos]

            if not patchy.ressource > 0 and patchy.signal > 0:
                self.follow_signal(agenty)

    def follow_signal(self, agentx: Agent):
        """

        :type agentx: Agent
        :param agentx: agent that travels against the signal gradient to find the ressource patch.
        """

        patchx = self.id_to_patch[agentx.pos]

        self.debug_string += "FOLLOWING \n" + "ag" + str(agentx.idx) + " AT: ptch" + str(patchx.idp) + "\n"

        neighborlist = []

        # get list of
        for patch in self.neighbor_list[patchx]:
            if self.id_to_patch[patch].signal > patchx.signal:
                neighborlist.append(patch)

        if len(neighborlist) > 0:
            next_patch_id = random.choice(neighborlist)
            next_patch = self.id_to_patch[next_patch_id]

            self.move_one_agent_defined(agentx, next_patch)

            patchx = self.id_to_patch[agentx.pos]

            self.debug_string += "ag" + str(agentx.idx) + "now AT: " + "ptch" + str(patchx.idp) + "\n"

            if agentx.perception <= patchx.signal < 1 and not patchx.ressource > 0:
                self.follow_signal(agentx)

                # patchx = self.id_to_patch[agentx.pos]
        else:

            if len(patchx.agentlist) > 1 and patchx.ressource > 0:
                self.fighting(agentx, patchx.agentlist)

    def single_agent_action(self):
        """
        picks a random agent and lets him perform an action depending on his situation

        """

        agentx = random.choice(self.agents)
        patchx = self.id_to_patch[agentx.pos]

        agentx.age += 1

        if patchx.ressource > 0:
            self.uptake(agentx)

        else:
            if 1 > patchx.signal >= agentx.perception:
                self.follow_signal(agentx)

            else:
                if random.random() < mobility:
                    self.move_one_agent_random(agentx)

    def diffuse_signal(self):
        """
        reduces strength of signal on patches

        """
        for patchy in self.patches:
            if patchy.signal > 0:
                patchy.signal /= 2

                if patchy.signal < 2 ** (-self.sig_range):
                    patchy.signal = 0

    def dissipate_signal(self):
        """
        removes the signal from every patch
        :return:
        """

        for patchy in self.patches:
            patchy.signal = 0

    def diffuse_ressource(self):
        """
        reduces the amount of ressources left on patches
        :return:
        """
        for patch in self.patches:
            if patch.ressource > self.resource_amount/50:
                patch.ressource -= self.resource_amount/50
            elif patch.ressource > 0:
                patch.ressource = 0

    def resource_empty(self):
        """
        Function to check if all patches do not contain resources anymore
        :return: Boolean, True if all patches are empty
        """
        really_empty = True

        for patchx in self.patches:
            if patchx.ressource > 1:
                really_empty = False

        return really_empty

    def constant_resource_replenish(self):
        """
        Function to replenish a random patch with resources after emptying a patch
        :return: False, (legacy code and lazyness reasons)
        """
        number_of_res_patches = math.ceil(self.P * self.fraction_covered)

        current_number_of_res_patches = len([patchx for patchx in self.patches if patchx.ressource > 0])

        diff = number_of_res_patches - current_number_of_res_patches

        if diff > 0:
            empty_patches = [patchy for patchy in self.patches if patchy.ressource < 1]

            selected_patches = random.choices(empty_patches, k=diff)

            for patchz in selected_patches:
                patchz.ressource = self.resource_amount

        return False

    def Initiate_Agents(self):
        """Initate agents at the start of the simulation
            depending on different parameters definded in Params

            :return: list of Agents
        """

        agents = []

        for n in range(self.N):
            if self.init_pheno_distr == 1:

                agents.append(
                    Agent(idx=n, cost_rep=self.init_c_rep, cost_upt=self.init_c_upt, cost_sig=self.init_c_sig,
                          uptake_rate=self.base_upt_rate, sig_range=self.sig_range))

            else:

                rep = random.choices([0.3, 0.7], [0.3, 0.7], k=1)[0]

                rnd_trait = [rep, 1-rep, 0]
                agents.append(Agent(idx=n, cost_rep=rnd_trait[0], cost_upt=rnd_trait[1], cost_sig=rnd_trait[2],
                                    uptake_rate=self.base_upt_rate, sig_range=self.sig_range))

            agents[n].UpdatePhenotype(self.params_)

        return agents

    def Evolve_Agents(self, agents):
        """
        Take all agents and evolve a subset according to a roulette wheel algorithm depending on the subset
        agents fitness

        returns:
        """
        self.debug_string += "\nEvolving! \n \n "

        # determine the amount of agents that should be evolved; arbitarily set to 10 % - 33 %
        agNr = len(agents)
        subset_size = int(agNr / 3)

        # the minimal subset sample size is 2
        if subset_size < 2:
            subset_size = 2

        self.debug_string += f"{subset_size} agents got picked \n"

        # prepare the probabilities to be picked depending on the agents age, higher age = higher probability
        age_P = []

        for agent in agents:
            age_P.append(agent.age)

        age_P /= np.sum(age_P)

        self.debug_string += f"probabilities to get picked: \n{age_P} \n \n"

        # choose a subset of agents to evolve
        subset_agents = np.random.choice(agents, p=age_P, size=subset_size, replace=False)

        subset_IDs = [agent_.idx for agent_ in subset_agents]

        self.debug_string += f"agents that got picked: \n{subset_IDs} \n"
        benifs = [self.agentdict[ida].benefits for ida in subset_IDs]
        fitsrates = [self.agentdict[ida].fitness_rate for ida in subset_IDs]
        self.debug_string += f"benefits: \n{benifs} \n"
        self.debug_string += f"benefits: \n{fitsrates} \n"

        remaining_agents = []

        for agent in agents:
            if agent.idx not in subset_IDs:
                remaining_agents.append(agent)

        agents = remaining_agents[:]

        fitness = 1.0 + self.str_of_selection * self.calculate_payoff(subset_agents)

        #print(fitness)

        # get the references for the new agents, multiples possible! care for call-by-reference errors
        refereced_subset = random.choices(subset_agents, weights=fitness, k=len(subset_agents))

        self.debug_string += f"agents that got procreated: \n{refereced_subset} \n \n"

        # create new objects by deepcopying the picked agents
        subset = [copy.deepcopy(agent) for agent in refereced_subset]

        subset = self.mutate_agents(subset, subset_IDs)

        agents = agents + subset

        return agents, subset_IDs

    def mutate_agents(self, mutated_agents, ids):
        """
        Function to mutate the newly generated agents and update their IDs with the ones of the agent that got selected
        for the evolution process
        :param mutated_agents: List of generated agents that get placed in the world after the evolution process
        :param ids: List of Ids of the agent selected for the evolution process
        :return: List of the mutated agents
        """
        for idx, id_ in enumerate(ids):
            mutated_agents[idx].idx = id_

            pre_trait_vec = mutated_agents[idx].cost_rep, mutated_agents[idx].cost_sig, mutated_agents[idx].cost_upt

            mutated_agents[idx].cost_rep, mutated_agents[idx].cost_sig, mutated_agents[
                idx].cost_upt = self.TraitMutation(pre_trait_vec, 0.05)

            mutated_agents[idx].UpdatePhenotype(self.params_)

        # random displacement for agents after birth
        for i_ in range(1):
            for agent_ in mutated_agents:
                self.move_one_agent_random(agent_)

        #for agent_ in mutated_agents:
        #    random_patch = random.choice([patchx for patchx in self.patches if patchx.ressource == 0])
        #    agent_.pos = random_patch.idp

        self.update_patch_agentlists()

        return mutated_agents

    def TraitMutation(self, trait, sigma_mut_, traitmin=0.0, traitmax=1.0):
        """
        Function to generate 3 random values from a Gaussian distribution with µ = 0 and sigma = sigma_mut and to add
        thse values to the trait vector of an agent
        :param trait: trait vector of an agent
        :param sigma_mut_: standard deviation of the gaussian distribution
        :param traitmin: min value of the trait values
        :param traitmax: max values of the trait values
        :return: altered trait
        """
        # Generate mutation as normal distr. random number with std-dev. sigma_mut
        mutnoise = np.random.normal(0.0, 0.2, len(trait)) # sigma_mut_

        trait += mutnoise # FromContinousToDiscrete(mutnoise, 0.2)
        # Clip values larger (smaller) traitmax (traimin) to traitmax (traitmin)
        trait = np.clip(trait, traitmin, traitmax)

        return trait

    def calculate_payoff(self, agents):
        """
        Function to calulate the relative advantage that the agents generated by harvesting resources
        """
        pop_fitness = np.zeros(len(agents))
        pop_costs = np.zeros(len(agents))

        for n, agent in enumerate(agents):

            if self.factor_rep > 0.0:
                pop_costs[n] += agent.cost_rep
            if self.factor_sig > 0.0:
                pop_costs[n] += agent.cost_sig
            if self.factor_upt > 0.0:
                pop_costs[n] += agent.cost_upt

            pop_fitness[n] = agent.benefits / agent.age

        payoff = pop_fitness - pop_costs

        self.debug_string += f"payoffs: {payoff} \n"

        delta_payoffs = (np.max(payoff) - np.min(payoff))

        if delta_payoffs > 0.0:
            differential_fitness = 2.0 * (payoff - np.min(payoff)) / delta_payoffs - 1
        else:
            differential_fitness = np.zeros(np.shape(payoff))

        self.debug_string += f"fitness: {differential_fitness} \n"

        return differential_fitness

    def calculate_average_payoff(self):
        """
        Function to calculate the average advantage generated by the selected agents.
        :return:
        """
        pop_fitness = np.zeros(len(self.agents))
        pop_costs = np.zeros(len(self.agents))

        for n, agent in enumerate(self.agents):

            if self.factor_rep > 0.0:
                pop_costs[n] += agent.cost_rep
            if self.factor_sig > 0.0:
                pop_costs[n] += agent.cost_sig
            if self.factor_upt > 0.0:
                pop_costs[n] += agent.cost_upt

            pop_fitness[n] = agent.fitness_rate

        payoff = pop_fitness - pop_costs

        return np.mean(payoff)

    def debugging(self, step):
        """
        Function to track whats going on in a separat debug file (WIP!)

        """

        if step == "Initialising":
            # create file to record debug string

            self.debug_string += "Agents: ------------ \n \n"

            self.debug_string += "id \t pos \t rep \t sig \t upt \n \n"

            for agentx in self.agents:
                self.debug_string += str(agentx) + "\n \n"

            self.debug_string += "Patches with ressources: ------------ \n \n"

            for patchx in [patch for patch in self.patches if patch.ressource > 0]:
                self.debug_string += str(patchx) + "\n \n"

    def simulate(self, process_id):
        """
        Function that runs the actual simulation (agent actions + evolution)
        :return: average payoff for every 100th generation and average phenotype trait every 100th generation
        """

        self.memory_usage = RAMchecker.memory()

        if self.use_debug_file:
            debug_file_name = '/home/winkler/Pictures/masterproject/Debugging_Neolution/{}_{}_{date:%Y-%m-%d ' \
                              '%H:%M:%S}.txt'.format(self.g,
                                                     self.pid,
                                                     date=datetime.datetime.now())
            debug_file = open(debug_file_name, "w+")

            self.debugging("Initialising")

        # print the default look of the world without any actions happened
        # if self.visual_debug:
        #    self.print_world_graph()

        # initial signalling of the agents to reproduce the behaviour of Pawels model
        self.initial_signalling()

        # print the default look of the world after the signalling step
        # if self.visual_debug:
        #    self.print_world_graph()

        # initial fighting of all agents to give territorial agents the chance to be alone on a patch
        self.initial_fighting()

        # Write debug string into the file
        if self.g % 10 == 0 and self.use_debug_file:
            debug_file.write(self.debug_string)

        # print the world after initial steps to rearrange agents according to their specs
        # if self.visual_debug:
        #    self.print_world_graph()

        # variable to prevent infinite loops happening and to order world graph images
        step_counter = 0

        self.debug_string += "\n \n ------- Start of Simulation ------- \n \n"

        avgpayoffs_vs_gens = []
        phenotype_vs_gens = []

        hist_list = []

        # actual simulation of the world
        while step_counter <= self.no_gens:

            # if step_counter%(self.no_gens/10) == 0 and not step_counter == 0:
            #    telegram_bot_sendtext(f'{str(self.no_gens/step_counter)}')

            if False:
                print(process_id, RAMchecker.memory(self.memory_usage))
                self.memory_usage = RAMchecker.memory()

            if self.visual_debug and any([step_counter in [2, self.no_gens - 1,
                                                           #1000000, 1500000,
                                                           #2000000, 2500000,
                                                           #3000000, 3500000,
                                                            ]]):

                abort = self.print_world_graph_simulation(step_counter)
                if abort == "abort!":
                    break

            # increment of the step_counter
            step_counter += 1

            # if step_counter == self.no_gens/2:
            #     self.fraction_covered += 0.5
            #     if self.fraction_covered > 1:
            #         self.fraction_covered = 1

            # reset debug string for every step to really only see changes from one step to the next
            self.debug_string = "\n{0} _________ \n".format(step_counter)

            self.single_agent_action()

            # if step_counter % self.evolution_rate == 0:
            #
            #      for agentx in self.agents:
            #          self.hist_array[agentx.idx] += agentx.benefits
            #
            #      hist_list.append(self.hist_array)
            #      print(self.hist_array)
            #      self.hist_array = np.zeros(self.N)

            if step_counter % self.res_persistence_time == 0:
                self.diffuse_ressource()

            if step_counter % 20 * len(self.agents) == 0:
                self.diffuse_signal()

            if step_counter % (self.no_gens/10) == 0:
                print("write agents")
                self.write_agents(step_counter)

            if step_counter % self.evolution_rate == 0:
                self.agents, IDs = self.Evolve_Agents(self.agents)

                # update the agent dictionary
                for agent in self.agents:
                    self.agentdict[agent.idx] = agent

                self.update_patch_agentlists()

            resources_empty_ = self.constant_resource_replenish()

            # if step_counter % (self.no_gens / 10) == 0:
            #     self.print_correlation_fitness()

            if resources_empty_:
                self.debug_string += f"\n Resources replenished -- \n \n"
                self.replenish_patches()

            avgpayoffs_vs_gens.append(self.calculate_average_payoff())
            phenotype_vs_gens = UpdatePhenotypeOutput(step_counter, self.agents, phenotype_vs_gens, self.no_gens)

            if self.g % 10 == 0 and self.use_debug_file:
                debug_file.write(self.debug_string)

        plt.close()

        if self.g % 10 == 0 and self.use_debug_file:
            debug_file.close()

        if False:
            print(process_id, RAMchecker.memory(self.memory_usage))
            self.memory_usage = RAMchecker.memory()

        return [np.mean(np.array(avgpayoffs_vs_gens)), np.mean(np.array(phenotype_vs_gens[1000::100]), axis=1)]

    """Print Functions:"""

    # ----------------------

    def print_occupation_list(self):
        """
        Prints "patch: agents" with agents being all agents on this patch

        """
        for key in self.patches:
            keystr = "patch-" + str(key.idp) + " contains: "
            for agentx in key.agentlist:
                keystr += str(agentx.idx) + ", "

            print(keystr)

    def print_agents(self):
        """
        Prints the representation for all agents and a table-header for orientation
        """
        print("agent,  pos,  rep-str, sig-prob, uptake, fitness")
        for agentx in self.agents:
            print(agentx)

    def print_patches(self):
        """
        Prints all patches
        """
        for patchy in self.patches:
            if len(patchy.agentlist) > 0:
                print("{} has {} agents sitting on it with {} ressources left".format(patchy,
                                                                                      len(patchy.agentlist),
                                                                                      patchy.ressource))

    def print_world_graph_simulation(self, stepcounter):
        """
        Plots the used graph of the simulation
        """

        self.fig1.add_axes(self.ax1)
        self.ax1.set_axis_off()

        if False:
            print(stepcounter, RAMchecker.memory(self.memory_usage))
            self.memory_usage = RAMchecker.memory()
            if self.memory_usage > 10000000000:
                return "abort!"

        nx.draw_networkx_edges(self.world_graph, self.pos, width=2.0, alpha=0.5, ax=self.ax1)

        for patchy in self.patches:
            if patchy.signal > 0 and patchy.ressource == 0:
                nx.draw_networkx_nodes(self.world_graph,
                                       self.pos,
                                       [patchy.idp],
                                       node_color='lightblue',
                                       node_size=650,
                                       ax=self.ax1,
                                       edgecolors=[0.9, 1 - (patchy.signal/2), 1 - (patchy.signal/2)],
                                       linewidths=4)

            elif patchy.signal == 0 and patchy.ressource == 0:
                nx.draw_networkx_nodes(self.world_graph,
                                       self.pos,
                                       [patchy.idp],
                                       node_color='lightblue',
                                       node_size=500,
                                       alpha=0.8,
                                       ax=self.ax1,
                                       )

            else:
                nx.draw_networkx_nodes(self.world_graph,
                                       ax=self.ax1,
                                       pos=self.pos,
                                       nodelist=[patchy.idp],
                                       node_color='g',
                                       node_size=500,
                                       alpha=(patchy.ressource / self.resource_amount),
                                       )

        agentdrawing_list = []

        for patch in self.patches:
            for agentx in patch.agentlist:
                if agentx.cost_rep > 0.5:
                    agentdrawing_list.append([self.pos[patch.idp], 'b', 'black' if agentx.age > 2 else 'r'])
                elif agentx.cost_upt > 0.5:
                    agentdrawing_list.append([self.pos[patch.idp], 'orange', 'black' if agentx.age > 2 else 'r'])
                elif agentx.cost_sig > 0.5:
                    agentdrawing_list.append([self.pos[patch.idp], 'g', 'black' if agentx.age > 2 else 'r'])
                else:
                    agentdrawing_list.append([self.pos[patch.idp], 'grey', 'black' if agentx.age > 2 else 'r'])

        for agenty in agentdrawing_list:
            self.ax1.plot(agenty[0][0] + random.uniform(-0.017, 0.017), agenty[0][1] + random.uniform(-0.017, 0.017),
                          color=agenty[1], marker='.', markersize=15, markeredgecolor=agenty[2],
                          markeredgewidth=0.8)

        filename = unique_file(self.agent_path + str(stepcounter), "png")

        self.fig1.savefig(filename, dpi=1000)

        plt.clf()
        plt.cla()

    def print_correlation_fitness(self):
        """
        Plot the fitness_rate over the cost_rep of every agent (debug function)
        """
        correlation_array = np.zeros((self.N, 2))

        for i in range(self.N):
            correlation_array[i, 0] = self.agents[i].fitness_rate
            correlation_array[i, 1] = self.agents[i].cost_rep

        corr_fig, cor_ax = plt.subplots(1, 1, figsize=(10, 10))

        cor_ax.scatter(correlation_array[:, 1], correlation_array[:, 0])
        plt.show()

    def create_histograms(self, hist_list):
        """
        Function to create histograms of how many agents harvested how many resources over the course of one evolution
        cycle
        :param hist_list: List of all resource values of all agents
        """
        safepath = self.safepath

        if not os.path.exists(safepath):
            os.mkdir(safepath)

        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))

        some_value = self.res_value * self.evolution_rate

        bins = [0,
                self.res_value,
                3 * self.res_value,
                10 * self.res_value,
                20 * self.res_value,
                30 * self.res_value,
                40 * self.res_value,
                50 * self.res_value,
                60 * self.res_value,
                ]

        print(bins)

        ax1.hist(hist_list, bins=bins, histtype='barstacked', color=len(hist_list) * ['lightblue'])

        fig1.savefig(safepath + f'hist_test2_{int(self.evolution_rate / self.N)}.png', dpi=300)

    def write_agents(self, step):
        """
        Write all current agents in a pickle file for later inspection
        :return: Null
        """
        filename = unique_file(self.agent_path + str(step), "pkl")

        if not "-" in filename.split("/")[-1]:
            telegram_bot_sendtext(f"Progress: {step/self.no_gens}")

        if os.path.exists(f"{self.agent_path}"):

            with open(f"{filename}", 'wb') as safefile:
                pkl.dump(self.agents, safefile)


def unique_file(basename, ext, kwargs=None):
    """
    Helper function to create unique filenames to prevent overwrites
    :param basename: name of the file w/o the extension
    :param ext: extension od the file (type)
    :param kwargs: dictionary, parameter: value pairs
    :return: String, unique filename
    """

    if not kwargs is None:
        for key, value in kwargs.items():
            basename += f'_{key}_{value}'
    actualname = f"{basename}.{ext}"
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = f"{basename}-{next(c)}.{ext}"
    return actualname


def FromContinousToDiscrete(trait, delta_trait=0.1):
    """
    Function to discretize the trait of an agent
    :param trait: trait of the agent
    :param delta_trait: size of the discretized chunks
    :return: discretized trait
    """
    tmp = trait / delta_trait
    trait_discrete = np.round(tmp) * delta_trait

    return trait_discrete


def ReadConfig(configfile="./elm.cfg"):
    """
    Reading params object from configuration file. containing all information about the simulation

    :param configfile: String, Path to config file to be read
    :return: Params, parameter object
    """

    config = configparser.RawConfigParser()
    config.read(configfile)
    L = config.getint('General_Parameters', 'size')
    N = config.getint('General_Parameters', 'no_agents')
    init_placement = config.getint('General_Parameters', 'init_placement')
    finding_bias = config.getfloat('General_Parameters', 'finding_bias')
    res_persistence_time = config.getfloat('General_Parameters', 'res_persistence_time')
    fraction_covered = config.getfloat('General_Parameters', 'fraction_covered')
    # normalize = config.getint('General_Parameters', 'normalize_ressource')
    gkernel_range = config.getfloat('General_Parameters', 'res_smoothing_range')
    prob_of_fight = config.getfloat('General_Parameters', 'prob_of_fight')
    beta_competition = config.getfloat('General_Parameters', 'beta_competition')
    connectivity = config.getfloat('General_Parameters', 'connectivity')
    mobility = config.getfloat('General_Parameters', 'mobility')
    sig_range = config.getint('General_Parameters', 'sig_range')
    resource_amount = config.getint('General_Parameters', 'resource_amount')

    no_gens = config.getint('Evol_Parameters', 'no_generations')
    bouts = config.getint('Evol_Parameters', 'no_bouts_per_gen')
    init_pheno_distr = config.getint('Evol_Parameters', 'init_phenotype_distr')

    # noinspection PyBroadException
    try:
        inputpath = config.get('Evol_Parameters', 'inputpath')
    except:
        inputpath = './'

    res_value = config.getfloat('Evol_Parameters', 'res_value')
    init_c_rep = config.getfloat('Evol_Parameters', 'init_cost_repulsion')
    init_c_upt = config.getfloat('Evol_Parameters', 'init_cost_uptake')
    init_c_sig = config.getfloat('Evol_Parameters', 'init_cost_signal')
    base_upt_rate = config.getfloat('Evol_Parameters', 'base_uptake_rate')
    base_sig_prob = config.getfloat('Evol_Parameters', 'base_signal_prob')
    factor_rep = config.getfloat('Evol_Parameters', 'factor_repulsion')
    factor_upt = config.getfloat('Evol_Parameters', 'factor_uptake')
    factor_sig = config.getfloat('Evol_Parameters', 'factor_signal')
    sigma_mut = config.getfloat('Evol_Parameters', 'sigma_mutation_noise')
    traitmin = config.getfloat('Evol_Parameters', 'trait_min')
    traitmax = config.getfloat('Evol_Parameters', 'trait_max')
    str_of_selection = config.getfloat('Evol_Parameters', 'strength_of_selection')
    evolution_rate = config.getint('Evol_Parameters', 'evolution_rate')

    outstep = config.getint('Comp_Parameters', 'outstep')
    poolsize = config.getint('Comp_Parameters', 'no_processors')
    maxiterations = config.getint('Comp_Parameters', 'max_iter_placement')
    debug = config.getint('Comp_Parameters', 'debug')
    visual_debug = config.getint('Comp_Parameters', 'visual_debug')

    # print("debug = ", debug)

    params_ = Params(N=N,
                     L=L,
                     bouts=bouts,
                     init_placement=init_placement,
                     finding_bias=finding_bias,
                     res_persistence_time=res_persistence_time,
                     fraction_covered=fraction_covered,
                     gkernel_range=gkernel_range,
                     no_gens=no_gens,
                     init_pheno_distr=init_pheno_distr,
                     inputpath=inputpath,
                     res_value=res_value,
                     init_c_rep=init_c_rep,
                     init_c_upt=init_c_upt,
                     init_c_sig=init_c_sig,
                     sigma_mut=sigma_mut,
                     prob_of_fight=prob_of_fight,
                     factor_rep=factor_rep, factor_upt=factor_upt, factor_sig=factor_sig,
                     base_upt_rate=base_upt_rate,
                     base_sig_prob=base_sig_prob,
                     beta_competition=beta_competition,
                     str_of_selection=str_of_selection,
                     traitmin=traitmin,
                     traitmax=traitmax,
                     outstep=outstep,
                     debug=debug,
                     poolsize=poolsize,
                     maxiterations=maxiterations,
                     visual_debug=visual_debug,
                     connectivity=connectivity,
                     mobility=mobility,
                     evolution_rate=evolution_rate,
                     sig_range=sig_range,
                     resource_amount=resource_amount)

    return params_


def WriteConfigFile(params_, configfile='elm.cfg'):
    """

    :param params_: parameter class object
    :param configfile: String, name of the file to be wrote in
    :return: /
    """

    config = configparser.RawConfigParser()

    config.add_section('General_Parameters')
    config.set('General_Parameters', 'size', '%d' % params_.L),
    config.set('General_Parameters', 'no_agents', '%d' % params_.N),
    config.set('General_Parameters', 'init_placement', '%d' % params_.init_placement)
    config.set('General_Parameters', 'finding_bias', '%g' % params_.finding_bias)
    config.set('General_Parameters', 'res_persistence_time', '%g' % params_.res_persistence_time)
    config.set('General_Parameters', 'fraction_covered', '%g' % params_.fraction_covered)
    config.set('General_Parameters', 'normalize_ressource', '%d' % params_.normalize)
    config.set('General_Parameters', 'res_smoothing_range', '%g' % params_.gkernel_range)
    config.set('General_Parameters', 'prob_of_fight', '%g' % params_.prob_of_fight)
    config.set('General_Parameters', 'beta_competition', '%g' % params_.beta_competition)
    config.set('General_Parameters', 'connectivity', '%g' % params_.connectivity)
    config.set('General_Parameters', 'mobility', '%g' % params_.mobility)
    config.set('General_Parameters', 'sig_range', '%g' % params_.sig_range)
    config.set('General_Parameters', 'resource_amount', '%d' % params_.resource_amount)

    config.add_section('Evol_Parameters')
    config.set('Evol_Parameters', 'no_generations', '%d' % params_.no_gens)
    config.set('Evol_Parameters', 'no_bouts_per_gen', '%d' % params_.bouts)
    config.set('Evol_Parameters', 'init_phenotype_distr', '%d' % params_.init_pheno_distr)
    config.set('Evol_Parameters', 'inputpath', params_.inputpath)
    config.set('Evol_Parameters', 'res_value', '%g' % params_.res_value)
    config.set('Evol_Parameters', 'init_cost_repulsion', '%g' % params_.init_c_rep)
    config.set('Evol_Parameters', 'init_cost_uptake', '%g' % params_.init_c_upt)
    config.set('Evol_Parameters', 'init_cost_signal', '%g' % params_.init_c_sig)
    config.set('Evol_Parameters', 'base_uptake_rate', '%g' % params_.base_upt_rate)
    config.set('Evol_Parameters', 'base_signal_prob', '%g' % params_.base_sig_prob)
    config.set('Evol_Parameters', 'factor_repulsion', '%g' % params_.factor_rep)
    config.set('Evol_Parameters', 'factor_uptake', '%g' % params_.factor_upt)
    config.set('Evol_Parameters', 'factor_signal', '%g' % params_.factor_sig)
    config.set('Evol_Parameters', 'sigma_mutation_noise', '%g' % params_.sigma_mut)
    config.set('Evol_Parameters', 'trait_min', '%g' % params_.traitmin)
    config.set('Evol_Parameters', 'trait_max', '%g' % params_.traitmax)
    config.set('Evol_Parameters', 'strength_of_selection', '%g' % params_.str_of_selection)
    config.set('Evol_Parameters', 'evolution_rate', '%g' % params_.evolution_rate)

    config.add_section('Comp_Parameters')
    config.set('Comp_Parameters', 'outstep', '%d' % params_.outstep)
    config.set('Comp_Parameters', 'no_processors', '%d' % params_.poolsize)
    config.set('Comp_Parameters', 'max_iter_placement', '%d' % params_.maxiterations)
    config.set('Comp_Parameters', 'debug', '%d' % params_.debug)
    config.set('Comp_Parameters', 'visual_debug', '%d' % params_.visual_debug)

    # Writing our configuration file to 'example.cfg'
    with open(configfile, 'w') as configfile:
        config.write(configfile)


def competition_function(delta: float, beta=1.0):
    """
    sigmoid competition function
    :param delta: difference in repulsion strength
    :param beta: beta_competition parameter, steepness of the function
    :return: winning probability for the stronger agent
    """

    return 1. / (1 + np.exp(-beta * delta))


def create_adjacency_matrix(parameter, L):
    """
    Taking different paremeters to create adjacency matrices of different shapes
    :param parameter:
    :param L: Amount of patches in the world
    :return: adjacency matrix for the world network
    """

    # debug_file = open("/home/winkler/Pictures/Evol_04/00.txt", "w+")

    # debug_file.write(str(parameter) + " :parameter, L: " + str(L))

    if parameter in [None, "cyclic"]:
        adjmatrix = np.zeros((L, L))

        # creation of a cyclic network
        for i in range(L):
            for j in range(L):
                if i == j + 1:
                    adjmatrix[i, j] = 1
                if i == 0 and j == L - 1:
                    adjmatrix[i, j] = 1

    elif type(parameter) in [int, float] and parameter > 0:

        connectivity = parameter

        adjmatrix = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if i == j:
                    adjmatrix[i, j] = 0
                else:
                    if random.randint(0, 101) < connectivity:
                        adjmatrix[i, j] = 1
                        adjmatrix[j, i] = 1

    elif parameter == -99 and L == 9:

        adjmatrix = np.zeros((9, 9))

        adjmatrix[2, 1] = 1
        adjmatrix[1, 2] = 1
        adjmatrix[2, 3] = 1
        adjmatrix[3, 2] = 1
        adjmatrix[3, 4] = 1
        adjmatrix[4, 3] = 1
        adjmatrix[4, 5] = 1
        adjmatrix[5, 4] = 1
        adjmatrix[3, 8] = 1
        adjmatrix[8, 3] = 1
        adjmatrix[0, 8] = 1
        adjmatrix[8, 0] = 1
        adjmatrix[7, 3] = 1
        adjmatrix[3, 7] = 1
        adjmatrix[6, 7] = 1
        adjmatrix[7, 6] = 1

    elif parameter == -98 and L == 6:

        adjmatrix = np.zeros((6, 6))

        adjmatrix[0, 1] = 1
        adjmatrix[1, 0] = 1
        adjmatrix[1, 2] = 1
        adjmatrix[2, 1] = 1
        adjmatrix[2, 3] = 1
        adjmatrix[3, 2] = 1
        adjmatrix[3, 4] = 1
        adjmatrix[4, 3] = 1
        adjmatrix[4, 5] = 1
        adjmatrix[5, 4] = 1

    elif parameter == -97 and L == 5:

        adjmatrix = np.zeros((L, L))

        adjmatrix[0, 1] = 1
        adjmatrix[1, 0] = 1
        adjmatrix[1, 2] = 1
        adjmatrix[2, 1] = 1
        adjmatrix[2, 3] = 1
        adjmatrix[3, 2] = 1
        adjmatrix[3, 4] = 1
        adjmatrix[4, 3] = 1

    elif parameter == -96 and L == 5:

        adjmatrix = np.zeros((L, L))

        adjmatrix[0, 2] = 1
        adjmatrix[2, 0] = 1
        adjmatrix[0, 3] = 1
        adjmatrix[3, 0] = 1
        adjmatrix[1, 2] = 1
        adjmatrix[2, 1] = 1
        adjmatrix[3, 4] = 1
        adjmatrix[4, 3] = 1

    elif parameter in range(-1, -90, -1):

        parameter = int(-parameter)

        size = L

        adjmatrix = np.zeros((size, size))

        for i in range(size):

            for j in range(size):

                if i == 0 and not j == 0 and j < parameter + 1:

                    adjmatrix[i, j] = 1
                    adjmatrix[j, i] = 1

                else:
                    if np.sum(adjmatrix[i]) == 1 and np.sum(adjmatrix[:, j]) == 0:
                        adjmatrix[i, j] = 1
                        adjmatrix[j, i] = 1

    elif parameter in ["quadratic", "cubic"]:
        adjmatrix = np.zeros((L, L))
        connectivity = 2 * L / (math.factorial(L - 1))
        for i in range(L):
            for j in range(L):
                if i == j:
                    adjmatrix[i, j] = 0
                else:
                    if np.random.rand() < connectivity:
                        adjmatrix[i, j] = 1
                        adjmatrix[j, i] = 1

    else:
        adjmatrix = np.random.rand(L, L)
        for i in range(len(adjmatrix)):
            for j in range(len(adjmatrix[i])):
                adjmatrix[i, j] = round(adjmatrix[i, j])

    # debug_file.close()

    return adjmatrix


def UpdatePhenotypeOutput(g, agents, phenotype_vs_gens, no_gens):
    """
    Function to append the results of the current generation to the output array
    """

    if g % (no_gens / 10) == 0:
        print(f'{g / 100} of {no_gens / 100}')

    rep_costs = np.reshape(np.array([a.cost_rep for a in agents]), (-1, 1))
    # if g % (no_gens / 10) == 0:
    #    print("rep_costs")
    #    print(rep_costs)
    sig_costs = np.reshape(np.array([a.cost_sig for a in agents]), (-1, 1))
    upt_costs = np.reshape(np.array([a.cost_upt for a in agents]), (-1, 1))

    phenotype_data = np.hstack((rep_costs, upt_costs, sig_costs))
    phenotype_vs_gens.append(phenotype_data)

    return phenotype_vs_gens


# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.

BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def PerformSimulation(safepath="./", configfile="./elm.cfg", adjmatrix=None):
    """
    Function performing the generation and dividing the processes if multiprocessing is enabled
    :param safepath: path to the directory where the results are stored
    :param configfile: path to the config file containing all parameter values
    :param adjmatrix: adjacency matrix of the used graph
    """

    params = ReadConfig(configfile=configfile)

    if not adjmatrix is None:
        adjmatrix = adjmatrix
    else:
        adjmatrix = create_adjacency_matrix(params.connectivity, params.L)

    pos = nx.drawing.layout.kamada_kawai_layout(nx.from_numpy_matrix(np.transpose(adjmatrix)))

    universe = Universe(params_=params, adjmatrix=adjmatrix, pos=pos, safepath=safepath)

    pool = Pool(processes=params.poolsize)

    results_dict = universe.simulate(process_id=1)

    #results_dict = pool.map(universe.simulate, range(params.poolsize))

    pool.close()

    sharedPayoffs = np.array([x[0] for x in results_dict])

    sharedPhenos = np.array([x[1] for x in results_dict])

    # np.savetxt(out_dir + "shared_phenos_{date:%Y-%m-%d %H:%M:%S}.txt".format(date=datetime.datetime.now()),
    # sharedPhenos)
    date = datetime.datetime.now()
    date = str(date).replace(" ", "_").replace(":", "_")
    outfile_phenos = unique_file(safepath + f"shared_phenos_", "pkl", {"perstime": params.res_persistence_time,
                                                                       "mobi": params.mobility,
                                                                       "frac": params.fraction_covered,
                                                                       "resam": params.resource_amount,
                                                                       "rep": factor_rep,
                                                                       "upt": factor_upt,
                                                                       "sig": factor_sig
                                                                       })

    with open(outfile_phenos, "wb") as shared_f:
        pkl.dump(sharedPhenos, shared_f)
    # phenotypes = np.concatenate(sharedPhenos, axis=1)

    data = dict()
    data['agents'] = universe.agents

    # 1,2,0 so that uptake is red, signalling green and repulsion blue
    data['pheno_vs_gens'] = np.mean(sharedPhenos, axis=1)
    data['avgpayoffs_vs_gens'] = sharedPayoffs
    # data['payoffs_last_gen'] = payoffs
    outfile = "./evo_results.pkl"

    with open(outfile, "wb") as results_file:
        pkl.dump(data, results_file)


if __name__ == "__main__":


    # Initialize simulation parameters
    L = np.int32(400)
    N = np.int32(200)
    no_gens = 300  # 500
    sigma_mut = 0.05
    prob_of_fight = 1
    str_of_selection = 0.8

    out_dir = "/home/winkler/Pictures/masterproject/20201110/"

    # connectivity=-5
    init_placement = 1  # 1 == random, else all agents on patch0
    init_pheno_distr = 1
    no_bouts_per_gen = 20

    connectivity = 0

    evolution_rate = 70 * N  # a good value is 170 * N

    no_gens = no_gens * evolution_rate

    result_list = []
    pheno_type_list = []
    size_results = getsize(result_list)

    telegram_bot_sendtext("Simulation started")

    param_dict = { 1: {"i": 20, "j": 720, "k": 600, "l": 8},
                   2: {"i": 20, "j": 220, "k": 600, "l": 8},
                   3: {"i": 50, "j": 720, "k": 600, "l": 8},
    }

    try:
        for i in [3, 3]:
            #for j in [120, 720]:
                fraction_covered = param_dict[i]["l"] / 100
                res_persistence_time = param_dict[i]["j"]
                res_value = 5
                beta_competition = 2.1  # 3.1
                base_upt_rate = 2.4  # 2.4
                mobility = param_dict[i]["i"] / 100
                sig_range = 4
                resource_amount = param_dict[i]["k"]

                # res_value=15.0
                init_c_rep = 0.5
                init_c_sig = 0.5
                init_c_upt = 0.5

                factor_rep = 1
                factor_upt = 1
                factor_sig = 1
                poolsize = 1

                # print(f"evolution after every {evolution_rate} generations \n{no_gens} of generations in total")

                params = Params(L=L, N=N, no_gens=no_gens, sigma_mut=sigma_mut, poolsize=poolsize,
                                prob_of_fight=prob_of_fight, str_of_selection=str_of_selection,
                                res_persistence_time=res_persistence_time, bouts=no_bouts_per_gen,
                                init_pheno_distr=init_pheno_distr,
                                base_upt_rate=base_upt_rate, res_value=res_value,
                                init_c_rep=init_c_rep, init_c_sig=init_c_sig, init_c_upt=init_c_upt,
                                factor_rep=factor_rep, factor_upt=factor_upt, factor_sig=factor_sig, debug=0,
                                init_placement=init_placement, visual_debug=1, connectivity=connectivity,
                                fraction_covered=fraction_covered, beta_competition=beta_competition, outstep=5,
                                mobility=mobility, evolution_rate=evolution_rate, sig_range=sig_range,
                                resource_amount=resource_amount
                                )

                adj = np.loadtxt("/home/winkler/git_repos/EvolSociality/Kai/adj_matrix_20x20", dtype=int)

                WriteConfigFile(params)

                WriteConfigFile(params, unique_file(f"{out_dir}elm",
                                                    "cfg",
                                                    {"perstime": res_persistence_time,
                                                     "res_am": resource_amount,
                                                     "mobi": mobility,
                                                     "frac": fraction_covered,
                                                     "rep": factor_rep,
                                                     "upt": factor_upt,
                                                     "sig": factor_sig}))

                PerformSimulation(
                    safepath=out_dir,
                    adjmatrix=adj)

                data = pkl.load(open('./evo_results.pkl', 'rb'))

                # print(np.array(data['pheno_vs_gens'][0]).shape)

                result_list.append([fraction_covered,
                                    res_persistence_time,
                                    ])

                pheno_type_list.append(data['pheno_vs_gens'])

                # print(getsize(result_list) - size_results)

                telegram_bot_sendtext(f'{i}')

                size_results = getsize(result_list)

                del data

        result_array = np.array(result_list)
        np.savetxt(out_dir + "frac_6pc_res_170.txt", result_array)
        pkl.dump(pheno_type_list, open(out_dir + "frac_6pc_res_170.pkl", "wb"))
        telegram_bot_sendtext("Simulation finished completely")

        print("Bye Bye Poggers!")

    except:
        print(sys.exc_info()[0])
        telegram_bot_sendtext(str(sys.exc_info()[0]))
        raise
