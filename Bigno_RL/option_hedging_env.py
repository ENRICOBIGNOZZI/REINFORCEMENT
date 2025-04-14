import math
import numpy as np
from price_simulator import PriceSimulator

class OptionHedgingEnv:
    """
    Ambiente per il Reinforcement Learning dell'hedging di opzioni.
    Simula il portafoglio di un agente che effettua hedging dinamico di un'opzione venduta.
    Lo stato include il prezzo corrente dell'asset sottostante e il tempo residuo, 
    e la ricompensa finale dipende dal P&L rispetto al payoff dell'opzione.
    """
    def __init__(self, price_simulator, option_type='call', strike=100.0):
        """
        Inizializza l'ambiente con un simulatore di prezzi e i parametri dell'opzione.
        :param price_simulator: Istanza di PriceSimulator per generare scenari di prezzo.
        :param option_type: Tipo di opzione ('call' o 'put').
        :param strike: Prezzo di esercizio (strike) dell'opzione.
        """
        self.price_simulator = price_simulator
        self.option_type = option_type
        self.strike = strike
        # Parametri derivati dal simulatore di prezzi
        self.T = price_simulator.T
        self.n_steps = price_simulator.n_steps
        self.dt = price_simulator.dt
        self.r = price_simulator.r
        # Dimensioni dello stato e dell'azione
        self.state_dim = 2   # stato = [prezzo_corrente, tempo_rimanente]
        self.action_dim = 1  # azione = quantità di asset sottostante da detenere
        # Stato interno del portafoglio
        self.current_step = 0
        self.current_price = None
        self.path = None
        self.cash = 0.0
        self.holding = 0.0
        self.initial_option_price = None

    def reset(self):
        """
        Reset dell'ambiente (inizio di un nuovo episodio).
        Genera un nuovo percorso di prezzo e inizializza il portafoglio con la cassa pari al premio incassato vendendo l'opzione.
        :return: Stato iniziale (np.array contenente [prezzo_iniziale, tempo_a_scadenza]).
        """
        # Genera un nuovo scenario di prezzo per l'asset sottostante
        self.path = self.price_simulator.simulate_path()
        self.current_step = 0
        self.current_price = self.path[0]
        remaining_time = self.T  # tempo rimanente (anni)
        # Calcola il prezzo iniziale teorico dell'opzione per impostare la cassa iniziale
        sigma_for_price = self.price_simulator.sigma
        if self.price_simulator.model == 'regime_switching' and self.price_simulator.sigma2 is not None:
            # usa la volatilità del primo regime come approssimazione
            sigma_for_price = self.price_simulator.sigma
        self.initial_option_price = PriceSimulator.black_scholes_price(
            self.current_price, self.strike, self.r, sigma_for_price, self.T, self.option_type
        )
        # Inizializza il portafoglio: nessuna posizione iniziale in underlying, cassa = premio incassato
        self.holding = 0.0
        self.cash = self.initial_option_price
        # Stato iniziale: prezzo corrente e tempo rimanente
        return np.array([self.current_price, remaining_time], dtype=np.float32)

    def step(self, action):
        """
        Esegue un passo temporale nell'ambiente dato l'azione specificata (posizione da detenere).
        Aggiorna lo stato del portafoglio e il prezzo, e calcola la ricompensa dello step.
        :param action: Quantità di asset sottostante da detenere fino al prossimo passo (può essere float).
        :return: Tuple (next_state, reward, done, info):
                 - next_state: nuovo stato [prezzo, tempo_rimanente] dopo aver applicato l'azione
                 - reward: ricompensa ottenuta in questo step (0 tranne che a fine episodio)
                 - done: True se si è raggiunta la scadenza dell'opzione (episodio terminato)
                 - info: dizionario con informazioni aggiuntive (es. 'pnl' finale a fine episodio)
        """
        target_holding = float(action)
        # Esegui il trade per raggiungere la posizione target: aggiorna cassa e posizione
        delta_holding = target_holding - self.holding
        # Se delta_holding > 0, compriamo asset (diminuisce la cassa); se < 0 vendiamo asset (aumenta la cassa)
        self.cash -= delta_holding * self.current_price
        self.holding = target_holding
        # Avanza al passo temporale successivo
        self.current_step += 1
        self.current_price = self.path[self.current_step]
        # Aggiorna la cassa con l'interesse maturato durante l'intervallo
        self.cash *= math.exp(self.r * self.dt)
        # Tempo rimanente alla scadenza
        remaining_time = max(self.T - self.current_step * self.dt, 0.0)
        # Controlla se la simulazione è terminata (opzione scaduta)
        done = (self.current_step >= self.n_steps)
        reward = 0.0
        info = {}
        if done:
            # Calcola il payoff dell'opzione alla scadenza
            if self.option_type == 'call':
                payoff = max(self.current_price - self.strike, 0.0)
            else:
                payoff = max(self.strike - self.current_price, 0.0)
            # Valore finale del portafoglio = cassa + valore posizione in underlying
            final_portfolio = self.cash + self.holding * self.current_price
            # P&L finale = portafoglio finale - payoff dell'opzione (importo pagato all'acquirente)
            pnl = final_portfolio - payoff
            # Ricompensa: negativo dell'errore quadratico (si massimizza quando P&L ~ 0)
            reward = - (pnl ** 2)
            info['pnl'] = pnl
        # Stato successivo
        next_state = np.array([self.current_price, remaining_time], dtype=np.float32)
        return next_state, reward, done, info
