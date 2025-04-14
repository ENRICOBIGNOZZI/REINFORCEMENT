import math
import numpy as np

class PriceSimulator:
    """
    Simulatore di prezzi per l'asset sottostante.
    Supporta diversi modelli: Black-Scholes (GBM), Random Walk, Regime Switching.
    """
    def __init__(self, model='gbm', S0=100.0, mu=0.0, sigma=0.2, r=0.0, T=1.0, n_steps=100,
                 sigma2=None, mu2=None, switch_prob=0.05):
        """
        Inizializza il simulatore di prezzi.
        :param model: Modello da utilizzare ('gbm' per Black-Scholes, 'random_walk' per Random Walk, 'regime_switching' per regime alternato).
        :param S0: Prezzo iniziale dell'asset sottostante.
        :param mu: Drift (tasso di crescita atteso) usato per i modelli di prezzo.
        :param sigma: Volatilità usata per il modello (o per il primo regime se regime switching).
        :param r: Tasso privo di rischio (usato per simulazioni in misura risk-neutral e calcolo prezzi teorici).
        :param T: Orizzonte temporale complessivo (es: 1.0 rappresenta un anno).
        :param n_steps: Numero di passi temporali (discretizzazioni) in [0, T].
        :param sigma2: (opzionale) Volatilità per il secondo regime (richiesto se model='regime_switching').
        :param mu2: (opzionale) Drift per il secondo regime (default = mu).
        :param switch_prob: Probabilità di commutare regime ad ogni passo (usato se model='regime_switching').
        """
        self.model = model
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.T = T
        self.n_steps = n_steps
        # Passo temporale discreto
        self.dt = T / n_steps if n_steps > 0 else 0.0
        # Parametri per regime switching
        self.sigma2 = None
        self.mu2 = None
        self.switch_prob = switch_prob
        if model == 'regime_switching':
            if sigma2 is None:
                sigma2 = sigma
            self.sigma2 = sigma2
            if mu2 is None:
                mu2 = mu
            self.mu2 = mu2
            self.switch_prob = switch_prob

    @staticmethod
    def black_scholes_price(S, K, r, sigma, T, option_type="call"):
        """Calcola il prezzo di un'opzione europea (call/put) usando il modello Black-Scholes."""
        if T <= 0:
            # Alla scadenza, il prezzo coincide con il payoff
            if option_type == "call":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)
        # Funzione di ripartizione della Normale standard
        N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        # Parametri d1 e d2 della formula di Black-Scholes
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            # Prezzo call: S * N(d1) - K * e^{-rT} * N(d2)
            return S * N(d1) - K * math.exp(-r * T) * N(d2)
        else:
            # Prezzo put: K * e^{-rT} * N(-d2) - S * N(-d1)
            return K * math.exp(-r * T) * N(-d2) - S * N(-d1)

    @staticmethod
    def black_scholes_delta(S, K, r, sigma, T, option_type="call"):
        """Calcola la delta (sensibilità al prezzo dell'underlying) per un'opzione call/put nel modello Black-Scholes."""
        if T <= 0:
            # Alla scadenza: delta ~ 1 per call ITM, 0 per call OTM (simile per put con segno opposto)
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -0.0 if S < K else 0.0
        N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if option_type == "call":
            return N(d1)
        else:
            # Delta put = N(d1) - 1 (valido per asset senza dividendi)
            return N(d1) - 1.0

    def simulate_path(self, n_steps=None):
        """
        Simula un singolo percorso del prezzo dell'asset secondo il modello specificato.
        :param n_steps: (opzionale) numero di passi da simulare (se None usa self.n_steps).
        :return: Lista dei prezzi simulati (incluso il prezzo iniziale come primo elemento).
        """
        if n_steps is None:
            n_steps = self.n_steps
        dt = self.T / n_steps if n_steps > 0 else 0.0
        path = [self.S0]
        S = self.S0
        if self.model == 'gbm':
            # Modello Black-Scholes: Geometric Brownian Motion
            for t in range(n_steps):
                z = np.random.normal()
                S = S * math.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * math.sqrt(dt) * z)
                path.append(S)
        elif self.model == 'random_walk':
            # Random walk additivo (approssimazione di GBM per piccoli dt)
            for t in range(n_steps):
                z = np.random.normal()
                S = S + self.mu * S * dt + self.sigma * S * math.sqrt(dt) * z
                # Evita valori negativi per il prezzo
                if S < 0:
                    S = 0.0
                path.append(S)
        elif self.model == 'regime_switching':
            # Modello con due regimi di volatilità (e drift) che si alternano
            state = 0 if np.random.rand() < 0.5 else 1  # stato iniziale (0 o 1)
            vol = self.sigma if state == 0 else (self.sigma2 if self.sigma2 is not None else self.sigma)
            mu = self.mu if state == 0 else (self.mu2 if self.mu2 is not None else self.mu)
            for t in range(n_steps):
                z = np.random.normal()
                S = S * math.exp((mu - 0.5 * vol ** 2) * dt + vol * math.sqrt(dt) * z)
                path.append(S)
                # Cambia regime con probabilità switch_prob ad ogni passo
                if np.random.rand() < self.switch_prob:
                    state = 1 - state
                    vol = self.sigma if state == 0 else (self.sigma2 if self.sigma2 is not None else self.sigma)
                    mu = self.mu if state == 0 else (self.mu2 if self.mu2 is not None else self.mu)
        else:
            raise ValueError(f"Modello sconosciuto: {self.model}")
        return path
