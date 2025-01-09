#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
import numpy as np
import pandas as pd
from numbers import Real
import idaes.logger as idaeslog

_logger = idaeslog.getLogger(__name__)
# This script is the backcaster for the price-taker optimization


class PricetakerBackcaster:
    """
    Use historical price signals as scenarios for the price-taker optimization
    """

    def __init__(self, price_signal, scenario, pointer=0, horizon=24, planning_horizon=36):
        """
        Initialize the PricetakerBackcaster

        Arguments:
            price_signal: pandas dataframe or array-like data. The electricity price signal.

            scenario: int, number of scenarios.

            pointer: int, for the rolling horizon optimization, wew need to know where we are.

            horizion: int, the length of the scenario in the price-taker model .

            planning_horizon: int, the length of the planning horizon.
        """

        self.price_signal = price_signal
        self.scenario = scenario
        self.pointer = pointer
        self.horizon = horizon
        self.planning_horizon = planning_horizon

        self._check_inputs()
        self.reshaped_price_signals = self._reshape_price_signals()


    def _check_inputs(self):
        """
        check the inputs for the class is valid.
        """
        if not isinstance(self.horizon, int):
            raise TypeError(
                "Given horizon is not an int object. Horizon as int is expected."
            ) 
        
        if not isinstance(self.planning_horizon, int):
            raise TypeError(
                "Given planning_horizon is not an int object. Planning_horizon as int is expected."
            )  

        if not isinstance(self.scenario, int):
            raise TypeError(
                "Given scenario is not an int object. Scenario as int is expected."
            )    

        price_length = len(self.price_signal)

        if price_length < 24:
            raise ValueError(
                f"The length of the price should be greater than 24, but the give price signal is of length {price_length}"
            )

        if price_length < self.scenario * self.horizon:
            raise ValueError(
                f"The length of the price signal is not {price_length}, but at least {self.scenario * self.horizon} is needed for the price-taker optimization" 
            )
        
        if self.planning_horizon < self.horizon:
            raise ValueError(
                "The planning horizon should be greater or equal to the horizon"
            )

    @property
    def price_signal(self):
        """
        Property getter for price_signal.

        Returns:
            int: max historical days
        """

        return self._price_signal
    
    @price_signal.setter
    def price_signal(self, value):
        """
        Property setter for price_signal.

        Arguments:
            value: intended value for price_signal.

        Returns:
            None
        """

        self._price_signal = value
    
    @property
    def horizon(self):
        """
        Property getter for horizon.

        Returns:
            int: horizon
        """

        return self._horizon
    
    @horizon.setter
    def horizon(self, value):
        """
        Property setter for horizon.

        Arguments:
            value: intended value for horizon.

        Returns:
            None
        """

        self._horizon = value

    @property
    def scenario(self):
        """
        Property getter for scenario.

        Returns:
            int: scenario
        """

        return self._scenario
    
    @scenario.setter
    def scenario(self, value):
        """
        Property setter for scenario.

        Arguments:
            value: intended value for scenario.

        Returns:
            None
        """

        self._scenario = value

    @property
    def planning_horizon(self):
        """
        Property getter for planning_horizon.

        Returns:
            int: planning_horizon
        """

        return self._planning_horizon
    
    @planning_horizon.setter
    def planning_horizon(self, value):
        """
        Property setter for planning_horizon.

        Arguments:
            value: intended value for planning_horizon.

        Returns:
            None
        """

        self._planning_horizon = value

    def _reshape_price_signals(self):
        """
        reshape the price signal according to the given scenarios and horizons.

        Returns:
            reshaped price signals.
        """

        signal_len = len(self.price_signal)

        if signal_len % self.horizon != 0:
            _logger.warning(
                f"The length of the price signal is not divisible by {self.horizon}. Drop the last {signal_len % self.horizon} data."
            )
            drop_index = signal_len - signal_len % self.horizon
            self.price_signal = self.price_signal[0:drop_index]

        # reshape the data by horizons.
        # use list to store the price data
        reshaped_price_signal = []
        num_periods = len(self.price_signal) // self.horizon
        for i in range(num_periods):
            start_index = i*self.horizon
            end_index = i*self.horizon + self.planning_horizon
            
            # if the planning_horizon > horizon, the last period will overflow the index. In this case, we use the first x periods of the price signal.
            if end_index > len(self.price_signal):
                new_end_index = end_index - len(self.price_signal)
                # connect two pieces of the data
                p1 = list(self.price_signal[start_index:])
                p2 = list(self.price_signal[:new_end_index])
                reshaped_price_signal.append(p1+p2)
            else:    
                reshaped_price_signal.append(list(self.price_signal[start_index:end_index]))
        
        return reshaped_price_signal
    
    def generate_price_scenarios(self):
        """
        Generate the price scenarios.

        Returns:
            price_scenarios: list object 
        """    
        price_scenarios = []
        # backcast the past self.scenario signals.

        # when the pointer is less than the scenario, use the data from the reserse order
        # for example. If pointer is at 1 and we have 10 scenarios. Use the data from the day 365 to 356
        if self.pointer+1 < self.scenario:
            num_periods_lack = self.scenario - (self.pointer + 1)
            for i in range(num_periods_lack, 0, -1):
                price_scenarios.append(self.reshaped_price_signals[-i])
            for j in range(0, self.pointer+1):
                price_scenarios.append(self.reshaped_price_signals[j])
        else:
            for i in range(self.pointer + 1 - self.scenario, self.pointer + 1, 1):
                price_scenarios.append(self.reshaped_price_signals[i])

        return price_scenarios


class PriceBackcaster:
    """
    This backcaster is for running "wind_battery_price_taker_new_uncertainty"
    """

    def __init__(self, price_signals, scenario, pointer, horizon, future_horizon):
        self.price_signals = price_signals
        self.scenario = scenario
        self.pointer = pointer
        self.horizon = horizon
        self.future_horizon = future_horizon

        self._check_inputs()
        self.reshaped_price_signals  = self._reshape_price_signals()

    def _check_inputs(self):
        """
        check the inputs for the class is valid.
        """
        if not isinstance(self.horizon, int):
            raise TypeError(
                "Given horizon is not an int object. Horizon as int is expected."
            ) 
        
        if not isinstance(self.future_horizon, int):
            raise TypeError(
                "Given future_horizon is not an int object. future_horizon as int is expected."
            )  

        if not isinstance(self.scenario, int):
            raise TypeError(
                "Given scenario is not an int object. Scenario as int is expected."
            )    

        price_length = len(self.price_signals)

        if price_length < self.horizon + self.future_horizon:
            raise ValueError(
                f"The length of the price should be greater than {sum(self.future_horizon + self.horizon)}, but the give price signal is of length {price_length}"
            )

        if price_length < self.horizon*self.scenario + 24:
            raise ValueError(
                f"The length of the price signal is not enough long for the price-taker optimization with {self.scenario} scenarios" 
            )
        
        if self.horizon % 24 != 0 or self.future_horizon % 24 != 0:
            raise ValueError(
                "The horizon and future horizon should be multiple of 24."
            )
        
        if self.future_horizon < self.horizon:
            raise ValueError(
                "The planning horizon should be greater or equal to the horizon"
            )
        
    @property
    def price_signals(self):
        """
        Property getter for price_signal.

        Returns:
            int: max historical days
        """

        return self._price_signals
    
    @price_signals.setter
    def price_signals(self, value):
        """
        Property setter for price_signal.

        Arguments:
            value: intended value for price_signal.

        Returns:
            None
        """

        self._price_signals = value
    
    @property
    def horizon(self):
        """
        Property getter for horizon.

        Returns:
            int: horizon
        """

        return self._horizon
    
    @horizon.setter
    def horizon(self, value):
        """
        Property setter for horizon.

        Arguments:
            value: intended value for horizon.

        Returns:
            None
        """

        self._horizon = value

    @property
    def scenario(self):
        """
        Property getter for scenario.

        Returns:
            int: scenario
        """

        return self._scenario
    
    @scenario.setter
    def scenario(self, value):
        """
        Property setter for scenario.

        Arguments:
            value: intended value for scenario.

        Returns:
            None
        """

        self._scenario = value

    @property
    def future_horizon(self):
        """
        Property getter for future_horizon.

        Returns:
            int: future_horizon
        """

        return self._future_horizon
    
    @future_horizon.setter
    def future_horizon(self, value):
        """
        Property setter for future_horizon.

        Arguments:
            value: intended value for future_horizon.

        Returns:
            None
        """

        self._future_horizon = value

    
    def _reshape_price_signals(self):
        """
        reshape the price signal according to the given scenarios and horizons.

        Returns:
            reshaped price signals.
        """

        signal_len = len(self.price_signals)

        if signal_len % self.horizon != 0:
            _logger.warning(
                f"The length of the price signal is not divisible by {self.horizon}. Drop the last {signal_len % self.horizon} data."
            )
            drop_index = signal_len - signal_len % self.horizon
            self.price_signals = self.price_signals[0:drop_index]

        # reshape the data by horizons.
        # use list to store the price data
        reshaped_price_signal = []
        num_periods = len(self.price_signals) // self.horizon
        for i in range(num_periods):
            start_index = i*self.horizon
            end_index = i*self.horizon + self.horizon
            
            # if the planning_horizon > horizon, the last period will overflow the index. In this case, we use the first x periods of the price signal.
            if end_index > len(self.price_signals):
                new_end_index = end_index - len(self.price_signals)
                # connect two pieces of the data
                p1 = list(self.price_signals[start_index:])
                p2 = list(self.price_signals[:new_end_index])
                reshaped_price_signal.append(p1+p2)
            else:    
                reshaped_price_signal.append(list(self.price_signals[start_index:end_index]))
        
        return reshaped_price_signal
    
    def generate_stage_1_price_signals(self):
        """
        stage_1 price_signal is in the length of self.horizon
        """
        stage_1_price_signal = self.reshaped_price_signals[self.pointer]
        
        return stage_1_price_signal
    

    def generate_stage_2_price_signals(self):
        """
        stage_2 price_signal is in the length of self.future_horizon
        """
        multipler = self.future_horizon // self.horizon
        if self.pointer >= self.scenario + 1:
            stage_2_price_signal = []
            _signals = self.reshaped_price_signals[self.pointer-self.scenario-1:self.pointer]

        else:
            # if not enough days to backcast (beginning of the year), use the end-of-year data.
            stage_2_price_signal = []
            lack_number = self.scenario + 1 - self.pointer
            _signal_1 = self.reshaped_price_signals[:self.pointer+1]
            _signal_2 = self.reshaped_price_signals[-lack_number:]
            _signals = _signal_2 + _signal_1
            
        for i in range(self.scenario):
            scenario_signals = []
            for j in range(multipler):
                scenario_signals += _signals[i+j]
            stage_2_price_signal.append(scenario_signals)
        
        return stage_2_price_signal