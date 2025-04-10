# IMC3

## Project Description
IMC3 is a trading simulation project that includes backtesting and trading strategies. The project is designed to help users develop and test trading algorithms using historical data. The repository contains several Python files that implement different components of the project, including data models, backtesting, and trading strategies.

## Installation
To install the project, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/Jiaweiwang04/IMC3.git
   ```
2. Navigate to the project directory:
   ```
   cd IMC3
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To use the project, follow these steps:
1. Prepare the historical data files and place them in the project directory.
2. Run the backtesting script:
   ```
   python BackTest.py
   ```
3. Analyze the backtesting results and adjust the trading strategies as needed.
4. Implement your trading strategies in the `Trader_2.py` file.
5. Run the trading simulation:
   ```
   python Trader_2.py
   ```

## File Descriptions
- `BackTest.py`: This file contains the backtesting logic for the trading strategies. It loads historical price and trade data, performs backtesting, and plots the results.
- `datamodel.py`: This file defines the data models used in the project, including classes for listings, observations, orders, order depths, trades, and the trading state.
- `Trader_2.py`: This file contains the implementation of the trading strategies. It includes logic for different trading strategies, such as RSI-based strategies and trend-following strategies.
