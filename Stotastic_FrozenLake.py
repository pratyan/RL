import numpy as np
import gym



class MDP:
    def __init__(self,env):
        self.env = gym.make(env)
        self.env = self.env.unwrapped
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.policy, self.vf = np.zeros(self.n_states), np.zeros(self.n_states) #Initially taking the state Values zeros
        self.vf_converged = False # Flag to see whaether the state values have converged after iteartions
        self.gamma = 0.99 # discount variable

        

    # For the State Value 
    def state_value(self,state,action):
        return np.sum([prob*(rew+self.gamma*self.vf[next_state]) for prob,next_state,rew,terminal in self.env.P[state][action]])



    #Updating the State Values untill it converges
    def value_iteration(self):
        n_iterations = 0 #keeping track of the no. iterations
        eps = 0.00001 # least diff

        while not self.vf_converged:
            dell=0 #Diff between the previous and current state value
            for state in range(self.n_states):
                old_vf = self.vf[state]
                #Updating the State Value
                # State Value = max(state_values of all the possible actions in that state)
                self.vf[state] = np.max([self.state_value(state,action) for action in range(self.n_actions)])
                dell = max(dell,np.abs(old_vf - self.vf[state]))

            #when the diff < than our threshold
            if dell<eps:
                self.vf_converged=True #Means we have reached Value convergence

            n_iterations+= 1
        return self.vf_converged,n_iterations



    def play_games(self,num_games):
        games_won = 0
        current_state = self.env.reset()
        for g in range(num_games):
            while True:
                action = np.argmax([self.state_value(current_state,action) for action in range(self.n_actions)])
                next_state,rew,done,_ = self.env.step(action)
                current_state = next_state
                games_won += rew

                if done:
                    current_state=self.env.reset()
                    if rew:
                        print("game won")
                    else:
                        print("game lost")
                    break
        print("total_games won %d"%games_won)

        

if __name__ == '__main__':
    lake = MDP('FrozenLake-v1')
    lake.env.reset()
    vf_converged,iterations = lake.value_iteration()
    print("vf_converged after {} iterations" .format(iterations))
    #print(lake.env.P) # To print the whole dictionary of the game, consisting all the posible states and thier respective reward and transition probability.
    lake.play_games(100)
    print(lake.policy.reshape(4,4))
    print(lake.vf.reshape(4,4))
    lake.env.close()
