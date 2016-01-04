import numpy as np
import sys

class ISGD:
    def __init__(self, n_user, n_item, k, r_pos=5.0, l2_reg=0.01, learn_rate=0.01):
        self.k = k
        self.r_pos = r_pos
        self.l2_reg = l2_reg
        self.learn_rate = learn_rate
        self.known_users = np.array([])
        self.known_items = np.array([])
        self.n_user = n_user
        self.n_item = n_item
        self.A = np.random.normal(0., 0.1, (n_user, self.k))
        self.B = np.random.normal(0., 0.1, (n_item, self.k))

    def update(self, u_index, i_index):
        if u_index not in self.known_users: self.known_users = np.append(self.known_users, u_index)
        u_vec = self.A[u_index]

        if i_index not in self.known_items: self.known_items = np.append(self.known_items, i_index)
        i_vec = self.B[i_index]

        err = self.r_pos - np.inner(u_vec, i_vec)
        self.A[u_index] = u_vec + self.learn_rate * (err * i_vec - self.l2_reg * u_vec)
        self.B[i_index] = i_vec + self.learn_rate * (err * u_vec - self.l2_reg * i_vec)

    def recommend(self, u_index, N, history_vec):
        """
        Recommend Top-N items for the user u
        """

        if u_index not in self.known_users: raise ValueError('Error: the user is not known.')

        recos = []
        scores = np.abs(self.r_pos - np.dot(np.array([self.A[u_index]]), self.B.T)).reshape(self.B.shape[0])

        cnt = 0
        for i_index in np.argsort(scores):
            if history_vec[i_index] == 1: continue
            recos.append(i_index)
            cnt += 1
            if cnt == N: break

        return recos
