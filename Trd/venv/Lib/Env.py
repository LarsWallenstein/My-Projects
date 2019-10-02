from gym.utils import seeding
from gym import spaces
import numpy as np
import gym
from Common import Actions, load_data

DEFAULT_BARS_COUNT = 3
COMISSION = 0.1
RESET = False
OFFSET = True
ON_CLOSE = False
DISCOUNT = 0.97
DRAWDOWN = 0.9

LENGTH=10


class State:
    def __init__(self, bars_count, comission, reset_on_close, reward_on_close, discount_hold, drawdown, length):

        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(comission, float)
        assert comission >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        assert isinstance(discount_hold, float)
        assert discount_hold >= 0.0
        self.bars_count = bars_count
        self.comission = comission
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.discount_hold = discount_hold
        self.drawdown = drawdown
        self.length = length
        self.iter = 0

    def reset(self, data, prices, offset):
        assert offset >= self.bars_count  # _____________WARNING_____________________
        assert data.shape[0] == prices.shape[0]
        self.long_position = False
        self.short_position = False
        self.open_price = 0.0
        self._data = data
        self._prices = prices
        self._offset = offset
        self._real_reward = 1
        self.iter = 0

    @property
    def shape(self):
        # tmp = self._prices.shape[1]*self.bars_count + 1 + 1 + 1
        return (1, 41 * self.bars_count + 1 + 1 + 1)

    def encode(self):
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = self._data.shape[1]
        j = 0
        for i in range(self.bars_count - 1, -1, -1):
            res[0, j * shift:(j + 1) * shift] = self._data[self._offset - i,:]
            j += 1
        res[0, -3] = float(self.long_position)
        res[0, -2] = float(self.short_position)
        if not (self.long_position or self.short_position):
            res[0, -1] = 0.0
        elif self.long_position:
            res[0, -1] = (self._cur_close() - self.open_price) / self.open_price
        else:
            res[0, -1] = (self.open_price - self._cur_close()) / self.open_price
        return res

    def _cur_close(self):
        return self._prices.loc[self._offset, 'Close']

    def set_length(self,length):
        self.length=length

    def step(self, action):
        assert isinstance(action, Actions)
        self.iter+=1
        reward = 0.00000
        done = False
        close = self._cur_close()
        # ТУТ МОЖНО В ПРИНЦИПЕ РЕАЛИЗОВАТЬ ДОКУПКУ АКЦИЙ, НО НЕ ЗАБУДЬ ДОБАВИТЬ УСРЕДНЕНИЕ
        # Награда в процентах, учти
        # В первую очередь обрати внимание на награды, с ними может быть хрень

        if self.long_position and action == Actions.Sell:
            self.long_position = False
            reward -= self.comission
            self._real_reward *= (1 - self.comission / 100)
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
                self._real_reward *= abs(1 + (close - self.open_price) / self.open_price)
            # else:
            #  reward += 100.0 * (close - self.prev_close) / self.prev_close
            # self._real_reward*=abs(1+(close - self.prev_close) / self.prev_close)
            # print("Long Trade, Bought: "+str(self.open_price)+", sold: "+str(close)+", income: "+str(-self.open_price+close))
            # print("Real_reward: "+str(self._real_reward))
            self.open_price = 0.0

        elif self.short_position and action == Actions.Buy:
            self.short_position = False
            reward -= self.comission
            self._real_reward *= (1 - self.comission / 100)
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (self.open_price - close) / self.open_price
                self._real_reward *= abs(1 + (self.open_price - close) / self.open_price)
            # else:
            #  reward += 100.0 * (self.prev_close - close) / self.prev_close
            # self._real_reward*=abs(1+(self.prev_close - close) / self.prev_close)
            # print("Short Trade, sold: "+str(self.open_price)+" bought: "+str(close)+", income: "+str(self.open_price-close))
            # print("Real_reward: "+str(self._real_reward))
            self.open_price = 0.0

        elif not (self.short_position or self.long_position) and (action != Actions.Skip):
            if action == Actions.Buy:
                self.long_position = True
            else:
                self.short_position = True
            self.open_price = close
            reward -= self.comission
            self._real_reward *= (1 - self.comission / 100)

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.shape[0] - 1

        if (self.long_position or self.short_position) and not self.reward_on_close:
            if self.long_position:
                reward += self.discount_hold * 100.0 * (close - prev_close) / prev_close
                # reward+=100.0*(close-prev_close)/prev_close
                self._real_reward *= abs(1 + (close - prev_close) / prev_close)
            else:
                reward += self.discount_hold * 100.0 * (prev_close - close) / prev_close
                # reward += 100.0 * (prev_close - close) / prev_close
                self._real_reward *= abs(1 + (prev_close - close) / prev_close)

        if self._real_reward < self.drawdown or self.iter==self.length:
            done = True
        return reward, done


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, net, bars_count=DEFAULT_BARS_COUNT,
                 comission=COMISSION, reset_on_close=RESET,
                 random_ofs_on_reset=OFFSET, reward_on_close=ON_CLOSE,
                 discount_hold=DISCOUNT, drawdown=DRAWDOWN, length=LENGTH):

        self._state = State(bars_count=bars_count,
                            comission=comission,
                            reset_on_close=reset_on_close,
                            reward_on_close=reward_on_close,
                            discount_hold=discount_hold,
                            drawdown=drawdown, length=length)

        self.action_space = spaces.Discrete(n=len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.res = net.reset
        self.seed()

    def reset(self):
        data, prices = load_data()
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(int(0.5 * prices.shape[0])) + bars
            # print('kek')
        else:
            offset = bars
        self._state.reset(data, prices, offset)
        self.res()
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"real_reward": self._state._real_reward, "offset": self._state._offset}
        return obs, reward, done, info

    def set_name(self,i):
        self.name=i

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]