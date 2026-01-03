from functools import partial
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)
from module.tools.strategy_agents import StrategyAgents

class Agent():
    def __init__(self,
                 *args,
                 strategy_number=1,
                 **kwargs
                 ):

        super(Agent, self).__init__()


        self.strategy_agent=StrategyAgents()
        self.strategy_number=strategy_number

    def decision(self, *input,obs,params, **kwargs):
        return self.strategy_agent.wrapper(self.strategy_number,data=obs,params=params)

