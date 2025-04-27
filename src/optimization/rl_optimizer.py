# src/optimization/rl_optimizer.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

# 강화학습에 사용할 상태 정의
@dataclass
class TradingState:
    """트레이딩 환경 상태"""
    price: float
    volume: float
    volatility: float
    bid_ask_spread: float
    position_size: float
    unrealized_pnl: float
    time_held: int

class TradingEnvironment:
    """트레이딩 강화학습 환경 (리스크 최적화용)
       - 실제 환경에서는 state_manager 등 외부 시장 데이터를 활용할 수 있음.
    """
    def __init__(self, initial_balance: float = 1000000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.position_price = 0
        self.done = False
        self.history = []

    def reset(self) -> TradingState:
        """환경 초기화"""
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.done = False
        self.history = []
        return self._get_state()

    def step(self, action: Dict) -> Tuple[TradingState, float, bool, Dict]:
        """한 스텝 실행 (action: {'type': 'buy'/'sell'/'hold', 'size': float})"""
        order_type = action['type']
        size = action['size']
        
        # 주문 실행
        if order_type == 'buy':
            cost = size * self._get_current_price() * (1 + self._get_slippage())
            if cost <= self.balance:
                self.balance -= cost
                self.position += size
                self.position_price = self._get_current_price()
        elif order_type == 'sell':
            if self.position >= size:
                revenue = size * self._get_current_price() * (1 - self._get_slippage())
                self.balance += revenue
                self.position -= size

        # 보상 계산 (포지션 유지 비용 및 위험 조정 반영)
        reward = self._calculate_reward()
        
        new_state = self._get_state()
        
        # 종료 조건: 잔고 소진 또는 2배 증가
        if self.balance <= 0 or self.balance >= self.initial_balance * 2:
            self.done = True

        return new_state, reward, self.done, {}

    def _get_state(self) -> TradingState:
        """현재 상태 반환"""
        return TradingState(
            price=self._get_current_price(),
            volume=self._get_current_volume(),
            volatility=self._calculate_volatility(),
            bid_ask_spread=self._get_spread(),
            position_size=self.position,
            unrealized_pnl=self._calculate_unrealized_pnl(),
            time_held=len(self.history)
        )

    def _calculate_reward(self) -> float:
        """보상 계산: 미실현 손익에서 포지션 유지 비용을 차감"""
        pnl = self._calculate_unrealized_pnl()
        # 추가로 위험 노출에 따른 페널티(예: 높은 포지션 크기에 대해 페널티)를 줄 수 있음
        risk_penalty = self.position * 0.0001
        return pnl - risk_penalty

    def _get_current_price(self) -> float:
        """현재가 조회 (예시)"""
        return 50000000

    def _get_current_volume(self) -> float:
        """현재 거래량 조회 (예시)"""
        return 100

    def _calculate_volatility(self) -> float:
        """변동성 계산 (히스토리 기반)"""
        if len(self.history) < 2:
            return 0.0
        returns = np.diff(np.log(self.history))
        return np.std(returns)

    def _get_spread(self) -> float:
        """스프레드 조회 (예시)"""
        return 0.001

    def _get_slippage(self) -> float:
        """슬리피지 추정 (예시)"""
        return 0.0002

    def _calculate_unrealized_pnl(self) -> float:
        """미실현 손익 계산"""
        if self.position == 0:
            return 0
        return self.position * (self._get_current_price() - self.position_price)

class PPONetwork(nn.Module):
    """PPO 신경망 (Discrete 액션과 연속적 위험 조정 출력 포함)"""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Discrete 액션을 위한 actor 네트워크
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        # 상태 가치(value) 예측
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # 연속적 위험 조정을 위한 head: 평균 및 학습 가능한 로그 표준편차 사용
        self.risk_mean = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.risk_log_std = nn.Parameter(torch.zeros(1))  # 학습 가능한 파라미터

    def forward(self, state):
        """
        입력 state에 대해 discrete 액션 확률, 상태 가치, 연속적 위험 조정값과 해당 로그확률 반환.
        위험 조정값은 Sigmoid를 적용하여 0~1 범위로 제한함.
        """
        action_probs = self.actor(state)
        value = self.critic(state)
        risk_mean = self.risk_mean(state)
        risk_std = self.risk_log_std.exp().expand_as(risk_mean)
        # 정규분포로부터 샘플링 (reparameterization)
        dist = torch.distributions.Normal(risk_mean, risk_std)
        risk_action = dist.rsample()  # 샘플링 (reparameterized)
        risk_action = torch.sigmoid(risk_action)  # 0~1 범위
        risk_log_prob = dist.log_prob(risk_action)  # (변환 보정은 단순화를 위해 생략)
        return action_probs, value, risk_action, risk_log_prob

class ParameterOptimizer:
    """강화학습 기반 리스크 최적화 (PPO 적용)
       - 환경(TradingEnvironment)에서 얻은 상태와 시장 데이터를 학습하여
         최적의 포트폴리오 밸런싱 및 손실 제어 정책을 도출함.
    """
    def __init__(self, env: TradingEnvironment, learning_rate: float = 0.0003):
        self.env = env
        state_dim = 7  # TradingState의 속성 수
        action_dim = 3  # buy, sell, hold
        
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 배치 데이터 저장용 리스트
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_values = []
        self.batch_probs = []
        self.batch_risk_actions = []
        self.batch_risk_log_probs = []

    def optimize(self, episodes: int = 1000) -> Dict:
        """강화학습 최적화 실행"""
        best_reward = float('-inf')
        best_params = None
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 상태를 텐서로 변환
                state_tensor = torch.FloatTensor([
                    state.price, state.volume, state.volatility,
                    state.bid_ask_spread, state.position_size,
                    state.unrealized_pnl, state.time_held
                ])
                
                # 네트워크 실행: discrete 액션, 상태 가치, 연속적 위험 조정값 및 해당 로그 확률 산출
                action_probs, value, risk_action, risk_log_prob = self.network(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
                
                # 주문 크기는 기본 크기 0.1에 위험 조정 계수를 곱하여 결정
                adjusted_size = 0.1 * risk_action.item()
                action_dict = {
                    'type': ['buy', 'sell', 'hold'][action],
                    'size': adjusted_size
                }
                
                next_state, reward, done, _ = self.env.step(action_dict)
                
                # 배치 데이터 저장
                self.batch_states.append(state_tensor)
                self.batch_actions.append(action)
                self.batch_rewards.append(reward)
                self.batch_values.append(value)
                self.batch_probs.append(action_probs[action])
                self.batch_risk_actions.append(risk_action)
                self.batch_risk_log_probs.append(risk_log_prob)
                
                state = next_state
                episode_reward += reward
                
                # 배치가 충분히 쌓이면 네트워크 업데이트
                if len(self.batch_states) >= 128:
                    self._update_network()
            
            # 에피소드별 결과 기록
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_params = self.network.state_dict()
            
            if episode % 100 == 0:
                logging.info(f"Episode {episode}, Reward: {episode_reward:.2f}")
        
        return {
            'best_reward': best_reward,
            'best_params': best_params
        }

    def _update_network(self):
        """신경망 업데이트 (PPO 업데이트)"""
        states = torch.stack(self.batch_states)
        actions = torch.tensor(self.batch_actions)
        rewards = torch.tensor(self.batch_rewards, dtype=torch.float32)
        values = torch.cat(self.batch_values)
        old_probs = torch.stack(self.batch_probs)
        old_risk_log_probs = torch.stack(self.batch_risk_log_probs)
        
        # Advantage 계산 (여기서는 단순히 rewards 사용)
        returns = rewards
        advantages = returns - values.detach()
        
        clip_epsilon = 0.2
        
        # 네트워크 재실행하여 새로운 확률 및 위험 로그확률 산출
        new_action_probs, new_values, new_risk_actions, new_risk_log_probs = self.network(states)
        new_probs = new_action_probs.gather(1, actions.unsqueeze(1))
        
        # discrete 액션에 대한 비율 및 PPO 손실
        ratio_disc = new_probs / old_probs.unsqueeze(1)
        surr1_disc = ratio_disc * advantages.unsqueeze(1)
        surr2_disc = torch.clamp(ratio_disc, 1-clip_epsilon, 1+clip_epsilon) * advantages.unsqueeze(1)
        actor_loss_disc = -torch.min(surr1_disc, surr2_disc).mean()
        
        # 연속적 위험 조정에 대한 비율 및 PPO 손실
        ratio_cont = torch.exp(new_risk_log_probs - old_risk_log_probs)
        surr1_cont = ratio_cont * advantages.unsqueeze(1)
        surr2_cont = torch.clamp(ratio_cont, 1-clip_epsilon, 1+clip_epsilon) * advantages.unsqueeze(1)
        actor_loss_cont = -torch.min(surr1_cont, surr2_cont).mean()
        
        actor_loss = actor_loss_disc + actor_loss_cont
        
        critic_loss = nn.MSELoss()(new_values, returns)
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 배치 데이터 초기화
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_values = []
        self.batch_probs = []
        self.batch_risk_actions = []
        self.batch_risk_log_probs = []

    def save_model(self, path: str):
        """모델 저장"""
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str):
        """모델 불러오기"""
        self.network.load_state_dict(torch.load(path))
