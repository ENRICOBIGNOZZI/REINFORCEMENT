import torch
from price_simulator import PriceSimulator
from option_hedging_env import OptionHedgingEnv
from ppo_agent import PPOAgent
from evaluation import Evaluator

# Impostazione dei parametri di simulazione
S0 = 100.0        # prezzo iniziale dell'asset sottostante
mu = 0.0          # drift (nel risk-neutral può essere 0 o r)
sigma = 0.2       # volatilità dell'asset
r = 0.01          # tasso privo di rischio
T = 10.0           # anni fino alla scadenza
n_steps = 1000      # numero di intervalli di tempo (es. 50 step)
strike = 100.0    # strike price dell'opzione
option_type = 'call'  # tipo di opzione ('call' o 'put')

# Istanzia il simulatore di prezzi e l'ambiente di hedging
price_simulator = PriceSimulator(model='gbm', S0=S0, mu=mu, sigma=sigma, r=r, T=T, n_steps=n_steps)
env = OptionHedgingEnv(price_simulator=price_simulator, option_type=option_type, strike=strike)

# Istanzia l'agente PPO
agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)

# Esegui il training PPO per un certo numero di episodi
training_episodes = 1000
agent.train(env, episodes=training_episodes)

# Salva il modello addestrato su file
torch.save(agent, "ppo_hedging_agent.pth")
print(f"Modello PPO salvato su ppo_hedging_agent.pth")

# Valutazione della strategia RL vs la strategia Delta Hedging
evaluator = Evaluator()
evaluation_episodes = 100  # numero di episodi di test per la valutazione
evaluator.evaluate(agent, env, episodes=evaluation_episodes)
# Visualizza (e salva) l'evoluzione del P&L nel tempo per un singolo episodio
evaluator.evaluate_with_path(agent, env)

# Visualizza (e salva) l'andamento medio del P&L nel tempo su più episodi
evaluator.evaluate_average_pnl_over_time(agent, env, episodes=50)

# Se desideri visualizzare anche i grafici a schermo (in modalità interattiva),

