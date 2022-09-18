#  Model parallel implementation interview question

#  Inputs: Shock file with variable scenario values,
#          Metadata file with variable descriptions.

#  Initial import - to be extended if needed
import os
import time
from multiprocessing import Pool
import threading
import concurrent.futures
import numpy as np
import pandas as pd
# import time
# import random
import matplotlib.pyplot as plt

#  Coding framework:
# ############## Parallel class #################
#  Code here:


class Parallel:
    def __init__(self,
                 func,
                 args,
                 workers):
        """
        Constructor of parallel class.
        Args:
            func: function, function to be run via Parallel
            args: args, arguments to the function
            workers: int, number of workers to run
        Returns:
            -
        """
        self.func = func
        self.args = args
        self.workers = workers

    def multithreading(self):
        """
        Multithreading function.
        Args:
            -
        Returns:
            res: list with results from func
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            future = []
            for i in self.args:
                future.append(executor.submit(self.func, i))
            return_value = [i.result() for i in future]
        return return_value

    def multiprocessing(self):
        """
        Multiprocessing function.
        Args:
            -
        Returns:
            res: list with results from func
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        with Pool(self.workers) as _p:
            return_value = _p.map(self.func, self.args)
        return return_value

    def run_parallel(self):
        """
        Run function for chosen parallel type.
        Args:
            -
        Returns:
            -
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        pool = Pool()
        result_async = [pool.apply_async(self.func, args=(i,)) for i in
                        self.args]
        results = [r.get() for r in result_async]
        return results

    @staticmethod
    def visualize_runtimes(results, title):
        """
        Method to visualize parallel runtimes.
        Args:
            results, results from parallel run in form of list
            title, string with plot title
        Returns:
            -
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        start, stop = results
        print(start, stop)
        plt.bar(np.asarray(start), np.asarray(stop))
        plt.grid(axis='x')
        plt.ylabel("Tasks")
        plt.xlabel("Seconds")
        plt.title(title)


# ###################### Virtual (parent) class ####################


class MacroVariable:
    def __init__(self,
                 my_path,
                 metadata_file,
                 scenario):
        """
        Constructor of parent class.
        Args:
            my_path: string, location of the data folder
            metadata_file: string, file with variables description
            scenario: string, name of scenario data file
        Returns:
        -
        """
        self.my_path = my_path
        self.metadata_file = metadata_file
        self.scenario = scenario

    def get_value(self, variable):
        """
        Re-Implement this method in child class. Gets
        quarterly predictions for variables.
        Args:
            variable: string with variable name
        Returns:
            matrix: matrix with predicted variable values
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        return None

    @staticmethod
    def get_description():
        """
        Prints variable description.
        Args:
            -
        Returns:
            String: string with variable description
        Author: John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        return 'Macro Variable'

    def get_initial_values(self, variables):
        """
        Loads the initial scenario values for given variables.
        Args:
            variables: string vector with variable names required from scenario
        Returns:
            scen_values: Matrix with scenario values of required variables
        Author: John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        # 1. Get data
        scenario_file = pd.read_csv(os.path.join(self.my_path, self.scenario))
        # 2. Select the variables, store the forecasting period and change date format
        scenario_file = pd.concat(
            [
                scenario_file[scenario_file['Variable'] == x]
                for x in variables
                ]
            ).reset_index(drop=True)
        # 2. Change reshape. Ease the later concat with historical data
        scen_values = scenario_file.pivot(index='TimeStamp',
                                          columns='Variable',
                                          values='Value').reset_index(drop=True)
        return scen_values

    def get_meta_data(self):
        """
        Loads meta data from metadata_file, indexed by variable name.
        Args:
            -
        Returns:
            metadata: table with variable description, units, etc.
        Author: John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        return pd.read_csv(os.path.join(self.my_path, self.metadata_file))

    @staticmethod
    def transform_values_to_log(x):
        """
        Transforms values to log values.
        Args:
            x: matrix with values of required variables
        Returns:
            log(x): matrix with log values of required variables
        Author: John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        return np.log(x)

    @staticmethod
    def get_return_type():
        """
        Gets the return type of the variable.
        Args:
            -
        Returns:
            String: string with variable return type
        Author: John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        return None

############################################ Child class ##############################################################
# Code here:


class VIX(MacroVariable):
    def __init__(self,
                 my_path,
                 metadata_file,
                 scenario):
        """ Constructor of child class.
        Args:
            my_path: string, location of the data folder
            metadata_file: string, file with variables description
            scenario: string, name of scenario data file
        Returns:
        -
        """
        super().__init__(my_path, metadata_file, scenario)
        self.indep_variable = ('SP500', 'LIBOR3M', 'HPI', 'CDS5Y', 'CPI')  # examples of variables

    @staticmethod
    def get_description():
        """
        Prints VIX variable description.
        Args:
            -
        Returns:
            String: string with variable description
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        return 'United States VIX Index (end of period)'

    @staticmethod
    def get_return_type():
        """
        Gets the return type of the VIX variable.
        Args:
            -
        Returns:
          String: string with variable return type
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        return 'level'

    def get_coefficients(self):
        """
        Create random coefficients of the model for established variables.
        Args:
            -
        Returns:
            coefficients_matrix: matrix with coefficient names and values
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        coefficients_matrix = pd.DataFrame(
                data=np.random.normal(
                    0, 1, size=(len(self.indep_variable), 1)
                    ),
                index=self.indep_variable,
                columns=['Coefficient']
                )
        return coefficients_matrix

    def get_value(self, variable):
        """
        Gets quarterly predictions for given variables.
        The final output is in the matrix format consisting of
        predicted variable values and time periods in scenario
        horizon, i.e. variable name as column name and periods
        as row names.
        Args:
          variable: string with explained variable name
        Returns:
          Matrix: matrix with predicted variable values
        Author:   John Doe, 2021/12/06
        Reviewer: Jack Doe, 2021/12/06
        """
        # Fetch scenario initial values
        initial_values = self.get_initial_values(self.indep_variable)
        # Fetch metadata info
        metadata = self.get_meta_data()
        # Apply transformation by ForecastType in metadata.
        for column_vars in initial_values.columns:
            idx_obs = metadata['Variable'] == column_vars
            for_type = metadata[idx_obs]['ForecastType'].values[0]
            if for_type.lower() == 'relative':
                initial_values.loc[:, column_vars] = (
                        self
                        .transform_values_to_log(
                            initial_values
                            .loc[:, column_vars]
                            )
                        .diff()
                        )
            elif for_type.lower() == 'absolute':
                initial_values.loc[:, column_vars] = (
                        self
                        .transform_values_to_log(
                            initial_values.loc[:, column_vars]
                            )
                        )
            else:
                raise ValueError
        # Take projections
        scenario_values = initial_values.loc[1:, :]
        # Fetch model "coefficients"
        coefficients_matrix = self.get_coefficients()
        # Run "model"
        model_values = pd.DataFrame(
                np.dot(scenario_values, coefficients_matrix),
                index=scenario_values.index,
                columns=[variable])
        return model_values

#######################################################################################################################
############################################ Code run #################################################################
# Code here:


if __name__ == '__main__':
    my_path = ''  # folder path
    metadata_file = 'Metadata.csv'
    scenario = 'EconomicScenario.csv'
    vix_model = VIX(my_path, metadata_file, scenario)
    # Example run
    print(vix_model.get_value('VIX'))
    # Parallelism
    models_parallel = ['VIX' + str(i) for i in range(500)]
    # Analysis:
    workers = 4
    parallel = Parallel(vix_model.get_value, models_parallel, workers)
    # Multiprocessing:
    #####
    start_time = time.time()
    ret = parallel.multiprocessing()
    mp_time = time.time() - start_time
    print(f'MultiProcessing: {mp_time}')

    # Multithreading:
    #####
    start_time = time.time()
    ret = parallel.multithreading()
    mt_time = time.time() - start_time
    print(f'Multithreading: {mt_time}')

    # Graph
    Parallel.visualize_runtimes([['MultiProcessing','MultiThreading'],[mt_time,  mp_time]],
                                'MultiProcessing vs MultiThreading')


