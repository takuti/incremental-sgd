import numpy as np
import sys

class ISGD:
    def __init__(self, k, l2_reg=0.01, learn_rate=0.01):
        self.k = k
        self.l2_reg = l2_reg
        self.learn_rate = learn_rate
        self.known_users = np.array([])
        self.known_items = np.array([])

    def update(self, u, i):
        if u not in self.known_users:
            u_vec = np.random.normal(0., 0.1, self.k)
            self.known_users = np.append(self.known_users, u)
            u_index = len(self.known_users) - 1
            self.A = np.vstack((self.A, u_vec)) if hasattr(self, 'A') else np.array([u_vec])
        else:
            u_index = np.where(self.known_users==u)[0][0]
            u_vec = self.A[u_index]

        if i not in self.known_items:
            i_vec = np.random.normal(0., 0.1, self.k)
            self.known_items = np.append(self.known_items, i)
            i_index = len(self.known_items) - 1
            self.B = np.vstack((self.B, i_vec)) if hasattr(self, 'B') else np.array([i_vec])
        else:
            i_index = np.where(self.known_items==i)[0][0]
            i_vec = self.B[i_index]

        err = 1. - np.inner(u_vec, i_vec)
        self.A[u_index] = u_vec + self.learn_rate * (err * i_vec - self.l2_reg * u_vec)
        self.B[i_index] = i_vec + self.learn_rate * (err * u_vec - self.l2_reg * i_vec)

    def recommend(self, u, N):
        """
        Recommend Top-N items for the user u
        """

        if u not in self.known_users: raise ValueError('Error: Given user_id is not known.')
        u_index = np.where(self.known_users==u)[0][0]

        recos = []
        scores = np.array([])
        for j in range(len(self.known_items)):
            scores = np.append(scores, np.inner(self.A[u_index], self.B[j]))
        for j in np.argsort(scores)[::-1][:N]:
            recos.append(self.known_items[j])
        return recos
