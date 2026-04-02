import numpy as np
import math
from collections import deque


class Time_calc:
    """
    Exchanged np.random.RandomState with np.random.default_rng to avoid
    issues with overlapping random seeds, as previous implementation had
    a cap of 100 different random seeds.
    """
    def __init__(self, parameters, episode):
        self.parameters = parameters

        num_machines = parameters['NUM_MACHINES']
        num_sources = parameters['NUM_SOURCES']
        num_transp = parameters['NUM_TRANSP_AGENTS']
        """Random Seed for random numbers"""
        np.random.seed(parameters['SEED'] + episode)

        # Change seed randomStreams for reproducibility
        stream_seed = None
        self.randomStreams = {}
        self.randomStreams["process_time"] = [
            np.random.default_rng(stream_seed) for _ in range(num_machines)
        ]

        self.randomStreams["machine_failure"] = [
            np.random.default_rng(stream_seed) for _ in range(num_machines)
        ]

        self.randomStreams["repair_time"] = [
            np.random.default_rng(stream_seed) for _ in range(num_machines)
        ]

        self.randomStreams["order_generation"] = [
            np.random.default_rng(stream_seed) for _ in range(num_sources)
        ]

        self.randomStreams["order_sequence"] = np.random.default_rng(
            stream_seed
        )

        self.randomStreams["transp_agent"] = [
            np.random.default_rng(stream_seed) for _ in range(num_transp)
        ]

        self.randomStreams["filled_initial_system"] = np.random.default_rng(
            stream_seed
        )

    """
    Helper functions
    """
    @staticmethod
    def clamp(value, lo, hi):
        """Clamp value between lo and hi."""
        return min(hi, max(lo, value))

    """
    Utility procedures.
    """
    def get_inventory_level(self, statistics):
        return statistics['stat_inv_episode'][-1][1]

    def processing_time(self, machine, statistics, parameters, order):
        """Return actual processing time for a concrete part."""
        max_time = parameters['MAX_PROCESS_TIME'][machine.id]
        min_time = parameters['MIN_PROCESS_TIME'][machine.id]
        avg_time = parameters['AVERAGE_PROCESS_TIME'][machine.id]
        rand_time = self.randomStreams["process_time"][machine.id].exponential(
            scale=avg_time
        )

        result_time = self.clamp(rand_time, min_time, max_time)
        statistics['stat_machines_working'][machine.id] += result_time
        return result_time

    def transp_time(self, start, end, transp, statistics, parameters):
        """Return actual transportation time for a concrete part."""
        result_time = parameters['TRANSP_TIME'][start.id][end.id]
        statistics['stat_transp_walking'][transp.id] += result_time
        statistics['stat_transp_working'][transp.id] += result_time
        return result_time

    def handling_time(self, MachineOrSource, LoadOrUnload,
                      transp, statistics, parameters):
        """Return actual handling time for a concrete part."""
        result_time = 0
        if MachineOrSource == "machine":
            if LoadOrUnload == "load":
                result_time += parameters['TIME_TO_LOAD_MACHINE']
            elif LoadOrUnload == "unload":
                result_time += parameters['TIME_TO_UNLOAD_MACHINE']
        elif MachineOrSource == "source":
            if LoadOrUnload == "load":
                result_time += parameters['TIME_TO_LOAD_SOURCE']
            elif LoadOrUnload == "unload":
                result_time += parameters['TIME_TO_UNLOAD_SOURCE']
        statistics['stat_transp_handling'][transp.id] += result_time
        statistics['stat_transp_working'][transp.id] += result_time
        return result_time

    def changeover_time(self, machine, current_variant,
                        next_variant, statistics, parameters):
        """Return actual changeover time for a machine."""
        result_time = parameters['CHANGEOVER_TIME']
        statistics['stat_machines_changeover'][machine.id] += result_time
        return result_time

    def time_to_failure(self, machine, statistics, parameters):
        """Return time until next failure for a machine."""
        MTBF = parameters['MTBF'][machine.id]
        machine_failure = self.randomStreams["machine_failure"][machine.id]

        result_time = machine_failure.exponential(scale=MTBF)
        return result_time

    def repair_time(self, machine, statistics, parameters):
        """Return time until next failure for a machine."""
        MTOL = parameters['MTOL'][machine.id]
        repair_time = self.randomStreams["repair_time"][machine.id]

        result_time = repair_time.exponential(scale=MTOL)
        statistics['stat_machines_broken'][machine.id] += result_time
        return result_time

    def time_to_order_generation(self, source, statistics, parameters):
        """Return time until next failure for a machine."""
        MTOG = parameters['MTOG'][source.id]
        order_generation = self.randomStreams["order_generation"][
            source.id - self.parameters['NUM_MACHINES']
        ]

        result_time = order_generation.exponential(scale=MTOG)
        return result_time

    def create_intermediate_production_steps_and_variant(self, statistics,
                                                         parameters, resources,
                                                         at_resource):
        """
        Return random sequence of production steps based on the distribution
        that is defined over all machines.
        """
        if at_resource.type == "source":
            result_prod_steps = [at_resource]
            while True:
                # Choose first machine based on the distribution of order
                # generation at sources, if the machine is not in the
                # responsible area of the source, repeat the process
                order_sequence = self.randomStreams["order_sequence"]
                first_machine = order_sequence.choice(
                    resources['machines'],
                    1,
                    p=parameters['ORDER_DISTRIBUTION']
                )[0]
                if first_machine.id in at_resource.resp_area:
                    break
            result_prod_steps.append(first_machine)
            result_prod_steps.extend(order_sequence.choice(
                resources['machines'],
                parameters['NUM_PROD_STEPS'] - 1,
                p=parameters['ORDER_DISTRIBUTION']
            ))
        else:
            filled_initial_system = self.randomStreams["filled_initial_system"]
            order_sequence = self.randomStreams["order_sequence"]
            first_machine = at_resource

            result_prod_steps = [x for x in resources['sources']
                                 if first_machine.id in x.resp_area]
            result_prod_steps.append(first_machine)

            actual_step = filled_initial_system.randint(
                1, high=parameters['NUM_PROD_STEPS']
            )

            result_prod_steps.extend(order_sequence.choice(
                resources['machines'],
                parameters['NUM_PROD_STEPS'] - 1 - actual_step,
                p=parameters['ORDER_DISTRIBUTION']
            ))
        if result_prod_steps[-1].id < 2:
            result_prod_steps.append(resources['sinks'][0])
        elif result_prod_steps[-1].id < 5:
            result_prod_steps.append(resources['sinks'][1])
        elif result_prod_steps[-1].id < 8:
            result_prod_steps.append(resources['sinks'][2])
        result_variant = order_sequence.choice(
            parameters['NUM_PROD_VARIANTS'],
            1,
            p=parameters['VARIANT_DISTRIBUTION']
        )

        return result_prod_steps, result_variant


def update_mov_avg(**kwargs):
    """ Function to iteratively calculate moving average with a given window"""
    kwargs['cont'].appendleft(kwargs['value'])
    new_mean = sum(kwargs['cont']) / len(kwargs['cont'])
    return new_mean


def update_mov_std(**kwargs):
    """ Function to iteratively calculate moving std with a given window"""
    kwargs['cont_sq'].appendleft(kwargs['value'] ** 2)
    len_wd = len(kwargs['cont_sq'])
    sum_sq = sum(kwargs['cont_sq'])
    sum_cont = sum(kwargs['cont'])
    new_var = ((1 / len_wd) * sum_sq) - ((1 / len_wd) * sum_cont) ** 2

    if new_var < 0:
        new_var = 0
    return math.sqrt(new_var)


def update_exp_weighted_mean(**kwargs):
    """ Function to iteratively calculate exponentially weighted mean"""
    alpha = kwargs['alpha']
    oldMean = kwargs['oldMean']
    value = kwargs['value']
    new_mean = (1 - alpha) * oldMean + alpha * value
    return new_mean


def update_exp_weightes_std(**kwargs):
    """ Function to iteratively calculate exponentially weighted std"""
    diff = kwargs['value'] - kwargs['oldMean']
    incr = kwargs['alpha'] * diff
    alpha = kwargs['alpha']
    oldStd = kwargs['oldStd']
    new_var = (1 - alpha) * (oldStd ** 2 + diff * incr)
    return math.sqrt(new_var)


class ZScoreNormalization(object):
    running_mean = dict(
        exp=update_exp_weighted_mean,
        mov=update_mov_avg
    )
    running_std = dict(
        exp=update_exp_weightes_std,
        mov=update_mov_std
    )

    def __init__(self, type, **config):
        """
        Calculates the z score normalization for a given method of
        calculating the mean and std.
        """
        self.update_mean = ZScoreNormalization.running_mean[type]
        self.update_std = ZScoreNormalization.running_std[type]
        self.config = dict(**config)
        self.type = type
        self.counter = 0
        self.mean = 0
        self.std = 0
        self.setup()

    def __call__(self, value):
        """
        By calling this function calculates the mean, std and z score
        normalization of the next iteration with given value.
        """
        self.counter += 1
        update = dict(
            oldMean=self.mean,
            value=value,
            oldStd=self.std,
            counter=self.counter,
            **self.attr_alg
        )
        self.mean = self.update_mean(**update)
        self.std = self.update_std(**update)

    def setup(self):
        """ Setup function to set all parameters for calculations"""
        if self.type == 'mov':
            size_wd = self.config['window']
            self.attr_alg = dict(
                cont_sq=deque([], maxlen=size_wd),
                cont=deque([], maxlen=size_wd), **self.config
            )
        elif self.type == 'exp':
            self.attr_alg = dict(**self.config)

    def get_z_score_normalization(self, value):
        if self.std != 0 and value is not None:
            normalized = (value - self.mean) / self.std
            return normalized
        elif value is None:
            return -1
        else:
            return 0

    def reset(self):
        self.counter = 0
        self.mean = 0
        self.std = 0
        self.setup()
