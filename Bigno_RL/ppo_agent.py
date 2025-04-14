import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class PPOAgent:
    """
    Agente PPO (Proximal Policy Optimization) con architettura Actor-Critic per l'hedging di opzioni.
    Usa una rete neurale in PyTorch per apprendere la policy (attore) e il valore (critico).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.99, eps_clip=0.2, value_coeff=0.5, entropy_coeff=0.01, ppo_epochs=4):
        """
        Inizializza l'agente PPO con i parametri dati.
        :param state_dim: Dimensione del vettore di stato in ingresso.
        :param action_dim: Dimensione dell'azione (numero di parametri dell'azione continua).
        :param hidden_dim: Dimensione dei layer nascosti della rete neurale.
        :param lr: Tasso di apprendimento per l'ottimizzatore.
        :param gamma: Fattore di sconto per il calcolo dei ritorni cumulati.
        :param eps_clip: Epsilon per il clipping del rapporto di probabilità (PPO).
        :param value_coeff: Coefficiente di peso per il loss del Critic (valore).
        :param entropy_coeff: Coefficiente di peso per il termine di entropia (esplorazione).
        :param ppo_epochs: Numero di epoch di aggiornamento da eseguire ad ogni iterazione PPO.
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs
        # Rete neurale Actor-Critic (condivisione dei layer hidden)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        # Parametro (logσ) per la deviazione standard della politica gaussiana (inizializzato a 0 -> σ=1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Ottimizzatore (unico per attore e critico)
        self.optimizer = optim.Adam(
            list(self.fc1.parameters()) + list(self.fc2.parameters()) +
            list(self.actor_head.parameters()) + list(self.critic_head.parameters()) +
            [self.log_std], lr=lr
        )

    def _forward(self, state):
        """
        Esecuzione forward della rete neurale dato lo stato.
        :param state: Tensor di dimensione (batch_size, state_dim)
        :return: (mean, value) - tensori per la media dell'azione (output actor) e il valore V(s) stimato (output critic).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.actor_head(x)   # output della politica (mean dell'azione)
        value = self.critic_head(x) # output del Critic (stima V(s))
        return mean, value

    def act(self, state):
        """
        Seleziona un'azione (posizione di hedging) dato lo stato corrente, utilizzando la policy attuale.
        Questa funzione viene usata durante l'interazione/valutazione (azione deterministica = media).
        :param state: Stato corrente (array numpy).
        :return: Azione scelta (float se l'azione è scalare, altrimenti array numpy).
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self._forward(state_t)
        # Usa la media della distribuzione gaussiana come azione deterministica
        action = mean[0].numpy()
        if action.shape == ():
            # Se l'azione è scalare, restituisce come float
            return float(action.item())
        else:
            return action

    def train(self, env, episodes):
        """
        Addestra l'agente PPO interagendo con l'ambiente per un dato numero di episodi.
        Raccoglie le traiettorie, calcola i vantaggi (advantages) e aggiorna la policy e la stima del valore.
        :param env: Istanza di OptionHedgingEnv su cui eseguire il training.
        :param episodes: Numero di episodi di training da eseguire.
        """
        for episode in range(episodes):
            state = env.reset()
            # Liste per memorizzare la traiettoria dell'episodio
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            values = []
            done = False
            # Interagisci con l'ambiente fino alla scadenza dell'opzione (fine episodio)
            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mean, value = self._forward(state_t)
                # Crea la distribuzione gaussiana della policy corrente
                std = torch.exp(self.log_std).expand_as(mean)
                dist = Normal(mean, std)
                # Campiona un'azione dalla distribuzione (esplorazione)
                action_tensor = dist.sample()
                log_prob_tensor = dist.log_prob(action_tensor).sum(dim=-1)
                # Estrai valori scalari per memorizzarli
                action_val = action_tensor.detach().squeeze(0).numpy()
                log_prob_val = log_prob_tensor.item()
                value_val = value.item()
                # Applica l'azione nell'ambiente
                next_state, reward, done, info = env.step(action_val)
                # Memorizza transizione
                states.append(state)
                actions.append(action_val)
                rewards.append(reward)
                old_log_probs.append(log_prob_val)
                values.append(value_val)
                # Passa allo stato successivo
                state = next_state

            # Episodio terminato: calcola i ritorni (discounted rewards) e gli advantage
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            values_tensor = torch.tensor(values, dtype=torch.float32)
            # Advantage = Return - Value (stima del vantaggio)
            advantages = returns - values_tensor
            # Normalizzazione degli advantage (migliora stabilità)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Converte le liste in tensori per l'update
            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.float32)
            if actions_tensor.dim() == 1:
                # Aggiungi dimensione se l'azione è scalare
                actions_tensor = actions_tensor.unsqueeze(1)
            old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32)

            # Aggiorna policy (Actor) e Critic usando PPO con K epoche
            for _ in range(self.ppo_epochs):
                mean, value = self._forward(states_tensor)
                std = torch.exp(self.log_std).expand_as(mean)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
                # Calcola il rapporto tra nuove e vecchie probabilità
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                # Loss Actor con clipping (PPO objective)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # Loss Critic (MSE tra valore stimato e return effettivo)
                critic_loss = F.mse_loss(value.squeeze(-1), returns)
                # Entropia media della policy (per incentivare l'esplorazione)
                entropy = dist.entropy().sum(dim=-1).mean()
                # Loss totale (somma pesata delle componenti, da minimizzare)
                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
                # Backpropagation e aggiornamento dei pesi
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
