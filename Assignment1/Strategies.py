"""
Strategies.py contains functions and classes used in assignment 1.
"""

import traceback  # traceback used for printing stack trace for exceptions


def get_headers(dataset):
    """
    Obtains header values from first row of dataset and adds the values
    into a header_key dict with corresponding column number keys.
    :param dataset: Stock dataset
    :return header_key: dict which can be used as a key for header names
    and the column to which they correspond
    """
    header_key = {}

    for column in range(0, len(dataset[0])):
        header_key[column] = dataset[0][column]

    return header_key


def strategy_1(dataset):
    """
    Assume for each day you know if your stock will be up or down.
    Assume you start with $100 on 1/1/2014.
    For each day, strategy is as follows:
        1. If stock goes down next day, sell all your shares today
            (at adj closing price)
        2. If stock goes up next day & you already own shares, do nothing
        3. If stock goes up next day & you don't own shares, buy (using all
            money to buy maximum number of shares at adjusted closing price)

    :param dataset: Stock dataset
    """

    funds = 100.00   # funds start at $100.00
    shares = 0       # start with no shares

    try:

        # dataset[1] is first day
        # option 2 implied
        for i in range(1, len(dataset)-1):

            current_price = float(dataset[i][12])
            next_day_price = float(dataset[i+1][12])

            # option 1
            if current_price > next_day_price and shares > 0:
                funds += current_price * shares
                shares = 0

            # option 3 (option already implied)
            elif current_price < next_day_price and shares == 0:
                shares = funds / current_price
                funds = 0

        # Display funds equivalent to shares
        if shares > 0:
            final_value = float(dataset[len(dataset)-1][12]) * shares
        # or just display the funds
        else:
            final_value = float(funds)

        print(f'\nStrategy 1 Final Funds: {final_value:.2f}')

    except Exception as e:
        print(e)
        traceback.print_stack()


def strategy_2(dataset):
    """
    Start with $100
    For each day:
        1. If Adj Close > L_MA, then buy
        2. If Adj close < L_MA, then sell (if you have stock)

    * Assuming that this strategy refers to that day's Adj Close
    * and L_MA
    *
    * Assumes that no stock is purchased on first day, since
    * Adj Close is equal to L_MA

    :param dataset: Stock dataset
    """

    funds = 100
    shares = 0

    try:

        for i in range(1, len(dataset)):
            current_price = float(dataset[i][12])
            long_ma = float(dataset[i][15])

            # option 1
            if current_price > long_ma and funds > 0:
                # buy
                shares = funds / current_price
                funds = 0

            # option 2
            elif current_price < long_ma and shares > 0:
                # sell
                funds += current_price * shares
                shares = 0

        # display funds equivalent to shares
        if shares > 0:
            # multiply final number of shares by final price
            final_value = float(dataset[len(dataset)-1][12]) * shares

        # or just display the funds
        else:
            final_value = float(funds)

        print(f'\nStrategy 2 Final Funds: {final_value:.2f}')

    except Exception as e:
        print(e)
        traceback.print_stack()


class SubSet:
    """
    Class that can hold/return min, max, avg, and median values
    for a SubSet of the complete stock DataSet. For example, it can
    be used for a particular day of the week or month of the year.
    """

    def __init__(self, name):
        self._name = name
        self.min = None
        self._max = None
        self._all_returns = []

    # min, max and name getters
    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def name(self):
        return self._name

    # min and max setters
    @min.setter
    def min(self, value):
        self._min = value

    @max.setter
    def max(self, value):
        self._max = value

    @name.setter
    def name(self, value):
        self._name = value

    def add_return_value(self, value):
        # appends new return value to all_returns list
        self._all_returns.append(value)

    def update(self, value):
        # updates SubSet with new return value and updates min/max as needed
        self.add_return_value(value)

        if self._max is None and self._min is None:
            self._max = value
            self._min = value

        elif value > self._max:
            self._max = value

        elif value < self._min:
            self._min = value

    def get_avg(self):
        # returns average of all daily return values
        if len(self._all_returns) > 0:

            total = 0
            for value in self._all_returns:
                total += value

            return total / len(self._all_returns)

        # return None if no data points stored
        else:
            return None

    def get_median(self):
        # returns median of all daily return values
        if len(self._all_returns) > 0:

            sorted_returns = sorted(self._all_returns)

            # number of return values is even
            if len(sorted_returns) % 2 is 0:
                med_1 = sorted_returns[len(sorted_returns) // 2]
                med_2 = sorted_returns[(len(sorted_returns) // 2) - 1]
                return (med_1 + med_2) / 2

            # number of return values is odd
            else:
                return sorted_returns[len(sorted_returns) // 2]

        # returns None if no data points stored
        else:
            return None


def strategy_3(dataset):
    """
    If you decide to hold a stock for 1 day, what is the best day of the
    week to do so?

    This function computes the average, minimum, and maximum daily returns
    for each day of the week. Findings are printed to the console.
    :param dataset:
    """

    weekday_data = {}

    try:
        # iterate through dataset (skipping header row)
        for i in range(1, len(dataset)):

            day = dataset[i][4]     # Weekday column from DataSet
            daily_return = float(dataset[i][13])    # daily return value

            # update daily return values of corresponding SubSet object
            try:
                if day not in weekday_data.keys():
                    weekday_data[day] = SubSet(day)

                weekday_data[day].update(daily_return)

            except KeyError as e:
                print(e)
                print("Invalid Weekday String")

        print("\nStrategy 3")

        # print formatted Weekday values to console
        print("Day of the week    | min       | max       | average   "
              "| median    |")
        for weekday in weekday_data.values():
            print(format(weekday.name, "<19s") + "| " +
                  format(weekday.min * 100, "<10.2f") + "| " +
                  format(weekday.max * 100, "<10.2f") + "| " +
                  format(weekday.get_avg() * 100, "<10.2f") + "| " +
                  format(weekday.get_median() * 100, "<10.2f") + "|")

    except Exception as e:
        print(e)
        traceback.print_stack()


def strategy_4(dataset):
    """
    If you decide to buy a stock for one month, what would be the best/
    worst month?

    This function computes the average, minimum, maximum, and median of daily
    returns. Findings printed to the console.
    :param dataset:
    :return:
    """

    month_names = {'1': "January",
                   '2': "February",
                   '3': "March",
                   '4': "April",
                   '5': "May",
                   '6': "June",
                   '7': "July",
                   '8': "August",
                   '9': "September",
                   '10': "October",
                   '11': "November",
                   '12': "December"}

    month_data = {}

    try:
        # iterate through dataset (skipping header row)
        for i in range(1, len(dataset)):

            month = dataset[i][2]     # Month column from DataSet
            daily_return = float(dataset[i][13])    # daily return value

            # update daily return values of corresponding SubSet object
            try:

                # check if month SubSet has already been added to month_data
                if month not in month_data.keys() and \
                        month in month_names:

                    # Create new month SubSet and add it to month_data
                    month_data[month] = SubSet(month_names[month])

                # update daily return values of corresponding Subset object
                month_data[month].update(daily_return)

            except KeyError as e:
                print(e)
                print("Invalid Month String")

        print("\nStrategy 4")

        # print formatted Weekday values to console
        print("Month              | min       | max       | average   "
              "| median    |")
        for month_obj in month_data.values():
            print(format(month_obj.name, "<19s") + "| " +
                  format(month_obj.min * 100, "<10.2f") + "| " +
                  format(month_obj.max * 100, "<10.2f") + "| " +
                  format(month_obj.get_avg() * 100, "<10.2f") + "| " +
                  format(month_obj.get_median() * 100, "<10.2f") + "|")

    except Exception as e:
        print(e)
        traceback.print_stack()


def buy_and_hold(dataset):

    initial_funds = 100.00   # funds start at $100.00
    initial_price = float(dataset[1][12])
    final_price = float(dataset[len(dataset)-1][12])
    final_funds = final_price * (initial_funds / initial_price)

    print(f'\nBuy and Hold Strategy Final Funds: {final_funds:.2f}')
