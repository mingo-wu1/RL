import gym
import numpy as np
import random

def main():
    # 创建 Taxi environment
    env = gym.make('Taxi-v3')

    # Step - 2 - -创建 - Q - 表并初始化
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size)) # 初始化 Q 表

    print(f'出租车问题状态数量为:{state_size}，动作数量为:{action_size}')
    #
    #   动作：0-5对应出租车在当前状态下的动作：(南，北，东，西，接乘客，放下乘客)。
    #   有6个离散的确定性动作：
    #   0：向南移动
    #   1：向北移动
    #   2：向东移动
    #   3：向西移动
    #   4：乘客上车
    #   5：乘客下车
    #
    #   奖励：
    #   每次行动奖励-1，解除乘客安全奖励+20。非法执行“载客/落客”行为的，奖励-10。
    #
    #   颜色：
    #   蓝色：乘客
    #   洋红：目的地
    #   黄色：空出租车
    #   绿色：出租车满座
    #
    #   状态空间：
    #   状态空间表示为：
    #   （出租车行、出租车列、乘客位置、目的地）
    #
    print(qtable)
    print(np.shape(qtable))

    # Step - 3 - -超参数设置
    learning_rate = 0.9  # 学习率
    discount_rate = 0.8  # 未来奖励折扣率# 探索相关参数
    epsilon = 1.0  # 探索概率
    decay_rate = 0.005  # 探索概率的指数衰减概率
    num_episodes = 1000  # 一共玩多少局游戏
    max_steps = 100  # 每一局游戏最多走几步

    # total_test_episodes = 100  # 测试中一共走几步
    # max_epsilon = 1.0  # 一开始的探索概率
    # min_epsilon = 0.01  # 最低的探索概率

    # Step - 4 - -Q - learning - 算法
    for episode in range(num_episodes):
        state = env.reset()  # reset() important function # 重置环境
        done = False

        for s in range(max_steps):  # 每一局游戏最多 99 步
            if random.uniform(0, 1) < epsilon:
                # 否则，进行探索（选择随机动作) 方法自动从所有可能的操作中选择一个随机操作
                action = env.action_space.sample()  # 传入参数 action vip important function
            else:
                # 如果这个数字大于 探索概率（开始时为 1），则进行开发（选择最大 Q 的动作）
                action = np.argmax(qtable[state, :])  # 传入参数 action

            # 这个动作与环境交互后，获得奖励，环境变成新的状态
            new_state, reward, done, info = env.step(action)  # 核心函数

            # 按照公式 Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)] 更新 Q 表
            qtable[state, action] = qtable[state, action] + \
                                    learning_rate * (reward + discount_rate * (np.max(qtable[new_state, :])
                                    - qtable[state, action]))

            # 迭代环境状态
            state = new_state
            if done == True:  # 如果游戏结束，则跳出循环
                break

        # 减小探索概率（由于不确定性越来越小）
        # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp (-decay_rate * episode)
        epsilon = np.exp(-decay_rate * episode)

    print(f'Trained over {num_episodes} episodes.')
    print('press Enter to start test...')

    state = env.reset()
    rewards = 0
    done = False

    # Step-5--使用-Q--表来玩-Taxi-- 使用上面得到的模型qtable
    for s in range(max_steps):
        # 测试中我们就不需要探索了，只要选择最优动作
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)  # action 是策略，是qtable模型
        env.render()

        rewards += reward
        print(f'step:{s}, rewards:{rewards}')
        state = new_state
        if done == True:
            break

    env.close()


if __name__ == '__main__':
    main()


# import gym
# import numpy as np
#
# env = gym.make('Taxi-v3')
# Q = np.zeros((env.observation_space.n,env.action_space.n))
#
# def trainQ():
#     for _ in range(10000):
#         observation = env.reset()
#         while True:
#             action = env.action_space.sample()
#             observation_,reward, done,info = env.step(action)
#             Q[observation,action] = reward + 0.75 * Q[observation_].max()
#             observation = observation_
#             if done:break
#     return Q
#
# def findway():
#     observation = env.reset()
#     rewards = 0
#     while True:
#         action = Q[observation].argmax()
#         observation_,reward, done,info = env.step(action)
#         print(observation_,reward, done,info)
#         rewards += reward
#         observation = observation_
#         env.render()
#         if done:
#             print(rewards)
#             break
#
# Q = trainQ()
# findway()