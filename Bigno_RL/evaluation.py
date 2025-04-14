import math
import numpy as np
import matplotlib.pyplot as plt
from price_simulator import PriceSimulator

class Evaluator:
    """
    Valutatore delle performance di hedging.
    Confronta la strategia RL appresa dall'agente con la strategia di Delta Hedging classica.
    Produce statistiche (media, varianza, percentili del P&L) e salva grafici comparativi in file PNG.
    """
    def evaluate(self, agent, env, episodes=100):
        """
        Esegue la valutazione della strategia RL e della strategia Delta Hedging su un certo numero di episodi.
        Calcola statistiche sui P&L finali e salva un istogramma comparativo in PNG.
        
        :param agent: Agente addestrato (istanza di PPOAgent).
        :param env: Ambiente OptionHedgingEnv su cui effettuare la valutazione.
        :param episodes: Numero di episodi di test da simulare.
        """
        rl_pnls = []
        delta_pnls = []
        for episode in range(episodes):
            # --- Strategia RL ---
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)  # azione deterministica dall'agente RL
                state, reward, done, info = env.step(action)
            rl_pnls.append(info.get('pnl', 0.0))
            
            # --- Strategia Delta Hedging ---
            path = env.path
            n_steps = env.n_steps
            K = env.strike
            r = env.r
            T = env.T
            # Usa la volatilità del modello; se regime switching usa la prima volatilità come approssimazione
            sigma = env.price_simulator.sigma
            if env.price_simulator.model == 'regime_switching' and env.price_simulator.sigma2 is not None:
                sigma = env.price_simulator.sigma
            cash = env.initial_option_price  # premio opzione incassato
            holding = 0.0
            dt = env.dt
            for t in range(n_steps):
                S = path[t]
                time_to_mat = T - t * dt
                # Calcola la delta teorica dell'opzione
                delta = PriceSimulator.black_scholes_delta(S, K, r, sigma, time_to_mat, option_type=env.option_type)
                delta_holding = delta - holding
                cash -= delta_holding * S
                holding = delta
                cash *= math.exp(r * dt)
            S_T = path[n_steps]
            final_portfolio = cash + holding * S_T
            if env.option_type == 'call':
                payoff = max(S_T - K, 0.0)
            else:
                payoff = max(K - S_T, 0.0)
            delta_pnls.append(final_portfolio - payoff)
        
        # Converte le liste in numpy array per analisi statistica
        rl_pnls = np.array(rl_pnls)
        delta_pnls = np.array(delta_pnls)
        
        # Calcola statistiche
        rl_mean = rl_pnls.mean()
        rl_var = rl_pnls.var()
        rl_p5, rl_p50, rl_p95 = np.percentile(rl_pnls, [5, 50, 95])
        delta_mean = delta_pnls.mean()
        delta_var = delta_pnls.var()
        delta_p5, delta_p50, delta_p95 = np.percentile(delta_pnls, [5, 50, 95])
        
        print(f"Strategia RL - P&L medio: {rl_mean:.4f}, Varianza: {rl_var:.4f}, 5° percentile: {rl_p5:.4f}, 95° percentile: {rl_p95:.4f}")
        print(f"Strategia Delta Hedging - P&L medio: {delta_mean:.4f}, Varianza: {delta_var:.4f}, 5° percentile: {delta_p5:.4f}, 95° percentile: {delta_p95:.4f}")
        
        # Salva il grafico istogramma dei P&L in un file PNG
        plt.figure(figsize=(8, 6))
        plt.hist(delta_pnls, bins=50, alpha=0.5, label='Delta Hedging')
        plt.hist(rl_pnls, bins=50, alpha=0.5, label='RL Hedging')
        plt.xlabel("P&L Finale")
        plt.ylabel("Frequenza")
        plt.title("Distribuzione dei P&L: RL vs Delta Hedging")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pnls_hist.png", dpi=300)
        plt.close()
        print("Istogramma dei P&L salvato in 'pnls_hist.png'.")
    
    def evaluate_with_path(self, agent, env):
        """
        Salva un grafico dell'andamento del P&L lungo il tempo per un singolo episodio,
        confrontando la strategia RL e quella di Delta Hedging.
        
        :param agent: Agente addestrato (istanza di PPOAgent).
        :param env: Ambiente OptionHedgingEnv.
        """
        # --- Strategia RL su un singolo episodio ---
        state = env.reset()
        pnl_rl = [0.0]
        time_axis = [0.0]
        t = 0
        while True:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            # Valore corrente del portafoglio per RL
            current_value_rl = env.cash + env.holding * env.current_price
            # Prezzo teorico dell'opzione
            opt_price = PriceSimulator.black_scholes_price(
                env.current_price, env.strike, env.r, env.price_simulator.sigma, env.T - t * env.dt, env.option_type)
            pnl_rl.append(current_value_rl - opt_price)
            t += 1
            time_axis.append(t * env.dt)
            if done:
                break

        # --- Strategia Delta Hedging su un singolo episodio ---
        path = env.path
        n_steps = env.n_steps
        K = env.strike
        r = env.r
        T = env.T
        sigma = env.price_simulator.sigma
        if env.price_simulator.model == 'regime_switching' and env.price_simulator.sigma2 is not None:
            sigma = env.price_simulator.sigma
        cash = env.initial_option_price
        holding = 0.0
        pnl_delta = [0.0]
        for t in range(n_steps):
            S = path[t]
            time_to_mat = T - t * env.dt
            delta = PriceSimulator.black_scholes_delta(S, K, r, sigma, time_to_mat, option_type=env.option_type)
            delta_holding = delta - holding
            cash -= delta_holding * S
            holding = delta
            cash *= math.exp(r * env.dt)
            current_value_delta = cash + holding * S
            opt_price = PriceSimulator.black_scholes_price(S, K, r, sigma, time_to_mat, env.option_type)
            pnl_delta.append(current_value_delta - opt_price)
        
        plt.figure(figsize=(8, 6))
        plt.plot(time_axis, pnl_rl, label='RL Hedging', marker='o')
        plt.plot(time_axis, pnl_delta, label='Delta Hedging', marker='x')
        plt.xlabel("Tempo")
        plt.ylabel("P&L Corrente")
        plt.title("Evoluzione del P&L lungo il tempo")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pnl_evolution.png", dpi=300)
        plt.close()
        print("Grafico dell'evoluzione del P&L salvato in 'pnl_evolution.png'.")
    
    def evaluate_average_pnl_over_time(self, agent, env, episodes=50):
        """
        Esegue più episodi e calcola l'evoluzione media del P&L lungo il tempo per entrambe le strategie
        (RL Hedging e Delta Hedging). Salva il grafico risultante in un file PNG.
        
        :param agent: Agente addestrato (istanza di PPOAgent).
        :param env: Ambiente OptionHedgingEnv.
        :param episodes: Numero di episodi da simulare per il calcolo della media.
        """
        n_steps = env.n_steps + 1  # includiamo il passo iniziale
        pnl_rl_all = np.zeros((episodes, n_steps))
        pnl_delta_all = np.zeros((episodes, n_steps))
        time_axis = np.linspace(0, env.T, n_steps)
        
        for ep in range(episodes):
            # --- Simulazione RL ---
            state = env.reset()
            pnl_rl = [0.0]
            while True:
                action = agent.act(state)
                state, reward, done, info = env.step(action)
                current_value_rl = env.cash + env.holding * env.current_price
                opt_price = PriceSimulator.black_scholes_price(
                    env.current_price, env.strike, env.r, env.price_simulator.sigma, env.T - len(pnl_rl)*env.dt, env.option_type)
                pnl_rl.append(current_value_rl - opt_price)
                if done:
                    break
            # Se l'episodio termina con meno di n_steps, riempiamo con l'ultimo valore
            if len(pnl_rl) < n_steps:
                pnl_rl += [pnl_rl[-1]] * (n_steps - len(pnl_rl))
            pnl_rl_all[ep, :] = np.array(pnl_rl)
            
            # --- Simulazione Delta Hedging ---
            path = env.path
            K = env.strike
            r = env.r
            T = env.T
            sigma = env.price_simulator.sigma
            if env.price_simulator.model == 'regime_switching' and env.price_simulator.sigma2 is not None:
                sigma = env.price_simulator.sigma
            cash = env.initial_option_price
            holding = 0.0
            pnl_delta = [0.0]
            for t in range(env.n_steps):
                S = path[t]
                time_to_mat = T - t * env.dt
                delta = PriceSimulator.black_scholes_delta(S, K, r, sigma, time_to_mat, option_type=env.option_type)
                delta_holding = delta - holding
                cash -= delta_holding * S
                holding = delta
                cash *= math.exp(r * env.dt)
                current_value_delta = cash + holding * S
                opt_price = PriceSimulator.black_scholes_price(S, K, r, sigma, time_to_mat, env.option_type)
                pnl_delta.append(current_value_delta - opt_price)
            if len(pnl_delta) < n_steps:
                pnl_delta += [pnl_delta[-1]] * (n_steps - len(pnl_delta))
            pnl_delta_all[ep, :] = np.array(pnl_delta)
        
        # Calcola la media e la deviazione standard sui P&L per ogni step
        mean_rl = np.mean(pnl_rl_all, axis=0)
        std_rl = np.std(pnl_rl_all, axis=0)
        mean_delta = np.mean(pnl_delta_all, axis=0)
        std_delta = np.std(pnl_delta_all, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(time_axis, mean_rl, label='RL Hedging (media)', color='C0')
        plt.fill_between(time_axis, mean_rl - std_rl, mean_rl + std_rl, color='C0', alpha=0.3)
        plt.plot(time_axis, mean_delta, label='Delta Hedging (media)', color='C1')
        plt.fill_between(time_axis, mean_delta - std_delta, mean_delta + std_delta, color='C1', alpha=0.3)
        plt.xlabel("Tempo")
        plt.ylabel("P&L Corrente (Media ± Deviazione Standard)")
        plt.title("Evoluzione Media del P&L lungo il Tempo")
        plt.legend()
        plt.tight_layout()
        plt.savefig("average_pnl_over_time.png", dpi=300)
        plt.close()
        print("Grafico dell'evoluzione media del P&L salvato in 'average_pnl_over_time.png'.")
