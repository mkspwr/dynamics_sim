__author__ = 'elubin'
from abc import ABCMeta, abstractmethod
import heapq
import math
import numpy as np
import random
from itertools import chain
# The precision of the decimal comparison operations this should not need any changing
DECIMAL_PRECISION = 5


class DynamicsSimulator(object):
    """
    An abstract class that is used as the super class for all dynamics simulations, both stochastic and deterministic.
    This class roughly corresponds to the replicator rules aspect of evolutionary game theory, as outlined on
    U{Wikipedia<http://en.wikipedia.org/wiki/Evolutionary_game_theory#Models>}
    """
    __metaclass__ = ABCMeta

    def __init__(self, payoff_matrix, player_frequencies, number_groups=1, pop_size=100, rate=0.01, stochastic=True, uniDist=False, fitness_func= None, selection_strengthI=0.8, selection_strengthG=0.8):
        """
        The constructor for the abstract class. This doesn't need to be called directly, as it is called by @see
        L{GameDynamicsWrapper}

        @param payoff_matrix: the L{PayoffMatrix} object that defines the game
        @type payoff_matrix: PayoffMatrix
        @param player_frequencies: a list indicating the relative frequency of each type of player, must sum to 1
        @type player_frequencies: iterable
        @param pop_size: the size of the population to use. If zero, infinite population size is used, or else a finite
            population size is used and players are never divided into parts.
        @type pop_size: int
        @param number_groups: Number of groups in the system
        @type number_groups: int
        @param rate: Rate at which groups reproduce.
        @type rate: float
        @param stochastic: whether or not the simulation is stochastic (True) or deterministic (False)
        @type stochastic: bool
        @param uniDist: whether or not the initial distribution is generated by a uniform (True) or multinomial (False) distribution
        @type uniDist: bool
        @param fitness_func: A function that maps payoff to fitness
        @type fitness_func: lambda that takes in two arguments, the payoff and the selection strength, and returns the fitness
        @param selection_strengthI: the selection strength that will be used in the fitness function at the individual level
        @type selection_strength: float
        @param selection_strengthG: the selection strength that will be used in the fitness function at the group level
        @type selection_strength: float
        """
        assert math.fsum(player_frequencies) == 1.0

        assert pop_size >= 0
        # The total population gets divided equally among the groups only when the population size is larger than
        # the number of groups.
        if pop_size > number_groups:
            pop_size = int(pop_size/number_groups)

        if pop_size > 0:
            self.num_players = self.round_individuals([pop_size * x for x in player_frequencies])
            assert sum(self.num_players) == pop_size
            self.infinite_pop_size = False
            self.uniDist = uniDist
        else:
            self.num_players = player_frequencies
            self.infinite_pop_size = True

        self.pop_size = pop_size
        self.number_groups = int(number_groups)
        self.rate = rate
        self.pm = payoff_matrix
        self.stochastic = stochastic
        self.selection_strengthI = selection_strengthI
        self.selection_strengthG = selection_strengthG
        if fitness_func is None:
            if stochastic:
                fitness_func = lambda p, w: math.e**(p*w)
            else:
                fitness_func = lambda p, w: p*w
        self.fitness_func = lambda payoff, w: float(fitness_func(payoff, w))


    @abstractmethod
    def next_generation(self, previous, group_selection, rate):
        """
        An abstract method that must be overridden assuming the default implementation of the simulate() method in
        order to define the state transition from generation i to generation i + 1.

        @param previous: The state of the population(s) before the transition, represented as an array of arrays. Each
            entry in the parent array refers to a player type, each entry in each sublist refers to the number or
            frequency of players playing that strategy.
        @type previous: list
        @param group_selection: Where the simulation is incorporating group selection
        @type group_selection: Boolean
        @param rate: Rate of group selection
        @type rate: float
        @return: the subsequent distribution of players by player type and fitnesses, after the state transition
        @rtype: list
        """
        return [],[]

    def validate_state(self, s):
        """
        Verifies validity of state, each state is an array of numpy arrays, one for every player type
        Also needs to coerce any arrays to numpy arrays

        @param s: the array representation of the state
        @type s: list(list())
        @return: whether or not the state is valid
        @rtype: bool
        """
        assert len(s) == self.pm.num_player_types
        for i, (p, expected, n_strats) in enumerate(zip(s, self.num_players, self.pm.num_strats)):
            if isinstance(p, (list, tuple)):
                p = np.array(p)
                s[i] = p

            assert isinstance(p, np.ndarray)
            assert p.sum() == expected
            assert len(p) == n_strats

        return s

    def simulate(self, num_gens=100, start_state=None, fixation_probability = False, strategy_indx = 0):
        """
        Group theory simulation for a given number of generations, optionally starting at a provided state.
        To Do: Better way to deal with population in each group

        @param num_gens: the number of iterations of the simulation.
        @type num_gens: int
        @param start_state: An optional list of distributions of strategies for each player.
        @type start_state: list or None
        @param fixation_probability: Whether the given simulation is to compute the fixation probability for a strategy.
        @type: bool
        @param strategy_indx: Strategy whose fixation probability is to be computed.
        @type: int
        @return: the list of states that the simulation steps through in each generation
        @rtype: list(list(nxm array)) where n:number of player types,m:number of strategies, list over the number of groups.
        """
        if self.number_groups > 1:
            group_selection = True
        else:
            group_selection = False

        # Set up the start state if not specified and validate it otherwise
        if start_state is None:
            start_state=[]
            if not self.infinite_pop_size:
                if self.uniDist:
                    distribution_for_player = lambda n_p, n_s: np.random.uniform(0, 1, n_s)
                else:
                    distribution_for_player = lambda n_p, n_s: np.random.multinomial(n_p, [1./n_s] * n_s)
            else:
                distribution_for_player = lambda n_p, n_s: np.random.dirichlet([1] * n_s) * n_p
            for i in range(self.number_groups):
                start_state.append([distribution_for_player(n_p, n_s) for n_p, n_s in zip(self.num_players, self.pm.num_strats)])

            # Modify the start state in order to compute the fixation probability
            if fixation_probability:
                assert self.number_groups ==1, ('Fixation probability can only be computed for one group')
                players_to_distribute = start_state[0][0][strategy_indx] - 1

                # There is only one individual playing the strategy whose fixation probability has to be computed.
                start_state[0][0][strategy_indx] = 1

                # Distribute rest of the players amongst other strategies making sure the total population size is fixed.
                player_dist = []
                number_divisions = len(start_state[0][0]) - 1
                count = 0
                for i in range(number_divisions):
                    if i != number_divisions - 1 :
                        div = int(players_to_distribute/number_divisions)
                        count += div
                        player_dist.append(div)
                    else:
                        player_dist.append(players_to_distribute-count)
                player_dist.insert(strategy_indx,0)
                for i in range(len(start_state[0][0])):
                    if i != strategy_indx:
                        start_state[0][0][i] += player_dist[i]
                        players_to_distribute -= player_dist

        else:
            assert len(start_state)==self.number_groups
            for i in range(self.number_groups):
                start_state[i] = self.validate_state(start_state[i])

        # Setting the initial payoffs to zero
        initial_payoffs = []
        for i in range(self.number_groups):
            initial_payoffs.append([0*np.random.uniform(0, 1, n_s) for n_p, n_s in zip(self.num_players, self.pm.num_strats)])

        # Store the strategy frequencies and payoffs received at each time step
        strategies=[start_state]
        payoffs=[initial_payoffs]

        # Actual simulation consisting of two levels of dynamics, one at the level of the group and one in between the groups.
        for i in range(num_gens):
            r,p=self.next_generation(start_state,group_selection,self.rate)
            strategies.append(np.array([r[j] for j in range(self.number_groups)],dtype=object))
            payoffs.append(np.array([p[j] for j in range(self.number_groups)],dtype=object))
            start_state=strategies[i+1]
            for j in range(self.number_groups):
                start_state[j] = self.validate_state(start_state[j])


        # Create lists that contain the total normalized frequency and payoffs associated with each player type across groups per time step
        strategies_total=[np.array([sum(np.array(strategies[i][j][k]) for j in range(self.number_groups))/self.number_groups for i in range(num_gens)]) for k in range(self.pm.num_player_types)]
        payoffs_total=[np.array([sum(np.array(payoffs[i][j][k]) for j in range(self.number_groups))/self.number_groups for i in range(num_gens)]) for k in range(self.pm.num_player_types)]

        return strategies_total, payoffs_total


    @staticmethod
    def round_individuals(unrounded_frequencies):
        """
        Due to integer cutoffs, the number of senders and receivers might not be consistent. This take the integer part
        of each of the inputs and then assign the remaining few leftovers (so that the sum is the sum of the original
        floats) in a way such that the numbers with higher decimal parts will get the extra int before those with lower.

        @param unrounded_frequencies: an iterable of floats representing the unrounded frequencies of players playing
            each strategy
        @type unrounded_frequencies: iterable
        @return: an iterable of frequencies of the same length, that sums to the same total.
        @rtype: iterable
        """
        unrounded_total = math.fsum(unrounded_frequencies)
        total = int(round(unrounded_total, DECIMAL_PRECISION))

        int_num_senders = [int(x) for x in unrounded_frequencies]

        diff = total - sum(int_num_senders)
        if diff > 0:
            # note the difference needs to be negative, because heapq's only implement a minimum priority queue but
            # we want max priority queue
            thresh = [((x - y), i) for i, (x, y) in enumerate(zip(int_num_senders, unrounded_frequencies))]
            heapq.heapify(thresh)
            while diff > 0:
                v, i = heapq.heappop(thresh)
                int_num_senders[i] += 1
                diff -= 1
        assert sum(int_num_senders) == total, "the total number of individuals after rounding must be the same as before rounding"
        return int_num_senders

    def calculate_payoffs(self, state):
        """
        """
        payoff = [[self.pm.get_expected_payoff(p_idx, s_idx, state)
                       for s_idx in range(num_strats_i)]
                      for p_idx, num_strats_i in enumerate(self.pm.num_strats)]


        # Average payoff of all members in a group. Lengths could be written better?
        avg_group_payoff = sum([payoff[i][j]*state[i][j] for i in range(len(payoff)) for j in range(len(payoff[0]))])/sum(chain(*state))

        return payoff, avg_group_payoff

    def calculate_fitnesses(self, payoff, selection_strength):
        """
        Given the payoff matrix for the dynamics simulation, calculate the expected payoff of playing each strategy
        for each player, as a function of the payoff for that strategy and the relative frequency of players playing
        every other strategy. Then use the fitness function to convert these payoffs to fitnesses.

        @param state: The current distribution of players playing each strategy for each player
        @type state: list(list())
        @return: a 2D array representing the fitness of playing each strategy for each player
        @rtype: list(list())
        """
        # Calculate fitness for each individual in the population (based on what strategy they are playing)
        fitness = [[self.fitness_func(p,selection_strength) for p in j] for j in payoff]


        return fitness
