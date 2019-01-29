#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Using Genetic Programming to Solve Symbolic Regression Problem"

__author__ = 'Hailong Feng'

import sys
import random
import math
import copy

# some paramemters
max_depth = 5  # for both tree growth depth & mutation depth
c_rate = 1  # corssover rate
m_rate = 0.2  # mutation rate
file_name = 'A4_trainingSamples.txt'  # should be in the same directory

# global variables
t_set = ['x', 'y', '1', '2', '3', '4', '5']  # terminal set
f_set = ['+', '-', '*', '/']  # function set


def draw_val(scope='tf'):
    # todo: to be modified with constant rate
    if scope == 'tf':
        draw = random.randint(0, 39)
        # draw from both terminal set & function set
        if draw <= 4:
            value = t_set[0]
        elif draw <= 9:
            value = t_set[1]
        elif draw <= 11:
            value = t_set[2]
        elif draw <= 13:
            value = t_set[3]
        elif draw <= 15:
            value = t_set[4]
        elif draw <= 17:
            value = t_set[5]
        elif draw <= 19:
            value = t_set[6]
        elif draw <= 24:
            value = f_set[0]
        elif draw <= 29:
            value = f_set[1]
        elif draw <= 34:
            value = f_set[2]
        elif draw <= 39:
            value = f_set[3]
    elif scope == 't':
        # draw from terminal set only
        draw = random.randint(0, 19)
        if draw <= 4:
            value = t_set[0]
        elif draw <= 9:
            value = t_set[1]
        elif draw <= 11:
            value = t_set[2]
        elif draw <= 13:
            value = t_set[3]
        elif draw <= 15:
            value = t_set[4]
        elif draw <= 17:
            value = t_set[5]
        elif draw <= 19:
            value = t_set[6]
    elif scope == 'f':
        # draw from function set only
            draw = random.randint(0, 3)
            value = f_set[draw]
    return value


def read_data(file_name):
    '''Read traning samples'''
    x = []
    y = []
    rf = []
    with open(file_name, 'r') as f:
        for line in f:
            sample = line.split()
            x.append(float(sample[0]))
            y.append(float(sample[1]))
            rf.append(float(sample[2]))
    return x, y, rf


def inital(pop_size):
    '''Initialize a population with pop_size
    Initialization method is ramped half-and-half
    pop is a list-like object with each member a tree-class-like object
    '''
    pop = []
    for i in range(pop_size):
        if i < (pop_size / 2):
            # for half of the pop_size create full tree
            t = Tree()
            t.grow_tree(method='full')
            pop.append(t)
        else:
            t = Tree()
            while t.root.left is None:
                # avoid one node tree, which is useless in this case
                t.grow_tree(method='grow')
            pop.append(t)
    # shuffle the population
    random.shuffle(pop)
    return pop


def crossover(par1, par2):
    '''given two parents of trees, return two offsprings
    par1 and par2 are two tree-like instances'''
    if random.uniform(0, 1) <= c_rate:
        # choose swaping point from off1
        off1 = copy.deepcopy(par1)
        off2 = copy.deepcopy(par2)
        swap_point1 = random.randint(0, off1.ncount-1)
        # choose replace point from par2
        repl_point1 = random.randint(0, par2.ncount-1)
        # replace the swap_node with repl_node
        off1.get_node(swap_point1).left = copy.deepcopy(
            par2.get_node(repl_point1).left)
        off1.get_node(swap_point1).right = copy.deepcopy(
            par2.get_node(repl_point1).right)
        off1.get_node(swap_point1).value = copy.deepcopy(
            par2.get_node(repl_point1).value)
        swap_point2 = random.randint(0, off2.ncount-1)
        repl_point2 = random.randint(0, par1.ncount-1)
        off2.get_node(swap_point2).left = copy.deepcopy(
            par1.get_node(repl_point2).left)
        off2.get_node(swap_point2).right = copy.deepcopy(
            par1.get_node(repl_point2).right)
        off2.get_node(swap_point2).value = copy.deepcopy(
            par1.get_node(repl_point2).value)
    else:
        off1 = copy.deepcopy(par1)
        off2 = copy.deepcopy(par2)
    return off1, off2


def mutate(off):
    if random.uniform(0, 1) <= m_rate:
        # scan the offsping to get the node count
        off.scan()
        # choose mutation point
        mut_point = random.randint(0, off.ncount-1)
        # grow a sub-tree with max_depth
        sub_t = Tree()
        sub_t.grow_tree(method='grow')
        # replace the mutation point with the growed sub-tree
        off.get_node(mut_point).left = copy.deepcopy(sub_t.root.left)
        off.get_node(mut_point).right = copy.deepcopy(sub_t.root.right)
        off.get_node(mut_point).value = copy.deepcopy(sub_t.root.value)
    return off


def tournament(pop, tour_size, x, y, rf):
    # randomly draw tour_size individuals from pop
    tour_ind = []  # store indexes of selected individual
    cur_depth = 0  # store average of tree depth
    for i in random.sample(range(len(pop)), tour_size):
        # scan the selected tree and compute its error
        pop[i].scan()
        cur_depth += pop[i].height
        pop[i].compute_error(x, y, rf)
        tour_ind.append(i)
    max1, max2, min1, min2, cur_fit = find_ind(pop, tour_ind)
    # compute avgerage depth
    cur_depth = cur_depth/tour_size
    # replace max1, max2 with crossover offsprings between min1 and min2
    pop[max1], pop[max2] = crossover(pop[min1], pop[min2])
    # mutate off1 and off2
    pop[max1] = mutate(pop[max1])
    pop[max2] = mutate(pop[max2])
    return cur_fit, min1, cur_depth


def find_ind(pop, tour_ind):
    '''Return the indexes of the largest two & the smallest two'''
    max_value1, max_value2 = float('-inf'), float('-inf')
    min_value1, min_value2 = float('inf'), float('inf')
    max_index1, max_index2, min_index1, min_index2 = None, None, None, None
    n = len(tour_ind)
    for i in range(n):
        # i denotes the index of tour_ind
        if pop[tour_ind[i]].error <= min_value1:
            min_value1, min_value2 = pop[tour_ind[i]].error, min_value1
            min_index1, min_index2 = i, min_index1
        elif pop[tour_ind[i]].error < min_value2:
            min_value2 = pop[tour_ind[i]].error
            min_index2 = i
        if pop[tour_ind[i]].error >= max_value1:
            max_value1, max_value2 = pop[tour_ind[i]].error, max_value1
            max_index1, max_index2 = i, max_index1
        elif pop[tour_ind[i]].error > max_value2:
            max_value2 = pop[tour_ind[i]].error
            max_index2 = i
    # convert to index of pop from index of tour_ind
    return tour_ind[max_index1], tour_ind[max_index2], \
        tour_ind[min_index1], tour_ind[min_index2], min_value1


def evlove(pop, tour_size, gen, x, y, rf):
    best_fit = []  # store fitness for ploting the figure
    avg_fit = []
    worst_fit = []
    tree_depth = []
    temp_fit = []  # temporary fitness value for 100 generation
    min_ind = None  # the index of best individual in current generation
    for i in range(1, gen+1):
        cur_fit, min_ind, cur_depth = tournament(pop, tour_size, x, y, rf)
        temp_fit.append(cur_fit)
        if i % 100 == 0:
            best = min(temp_fit)
            avg = sum(temp_fit)/len(temp_fit)
            worst = max(temp_fit)
            print('Generation %s:' % i)
            print('Best fitness is', best)
            print('Avg fitness is', avg)
            print('Worst fitnes is', worst)
            print()
            best_fit.append(best)  # store fitness for ploting the figure
            avg_fit.append(avg)
            worst_fit.append(worst)
            tree_depth.append(cur_depth)
            temp_fit = []  # clean temporary fitness
        if i == gen:
            # print best tree in the final generation
            print('Evolution ends...')
            print('Best tree representation in the final generation is:')
            pop[min_ind].print_tree()
    return best_fit, avg_fit, worst_fit, tree_depth


def plot_figure(b, a, w, depth, gen):
    import matplotlib.pyplot as plt
    axis = range(100, gen+1, 100)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(axis, w, 'b', label='worst')
    ax1.plot(axis, a, 'g', label='avg')
    ax1.plot(axis, b, 'r', label='best')
    ax1.set_ylabel('error')
    ax1.set_xlabel('generation')
    ax1.legend(loc=2)
    ax2 = ax1.twinx()
    ax2.plot(axis, depth, 'c', label='depth')
    ax2.set_ylabel('avg tree depth')
    ax2.legend(loc=1)
    plt.show()


class Node(object):
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
        self.depth = None
        self.ind = None


class Tree(object):
    def __init__(self):
        self.root = Node()
        self.height = None
        self.ncount = None
        self.error = None

    def grow_tree(self, method='grow'):
        self.root.depth = 0
        if method == 'full':
            self._full(self.root)
        elif method == 'grow':
            self._grow(self.root)

    def _full(self, cur_node):
            if cur_node.depth < max_depth:
                # if current depth < max_depth, insert value from f_set
                cur_node.left = Node()
                cur_node.left.depth = cur_node.depth + 1
                cur_node.right = Node()
                cur_node.right.depth = cur_node.depth + 1
                self._full(cur_node.left)
                cur_node.value = draw_val(scope='f')
                self._full(cur_node.right)
            elif cur_node.depth == max_depth:
                # if current depth == max_depth, insert value from t_set
                cur_node.value = draw_val(scope='t')

    def _grow(self, cur_node):
            if cur_node.depth < max_depth:
                # if current depth < max_depth, insert value from f or t
                value = draw_val(scope='tf')
                if value in t_set:
                    # if the value is in terminal set, do not create children
                    cur_node.value = value
                elif value in f_set:
                    # if the value is in function set, create children
                    cur_node.left = Node()
                    cur_node.left.depth = cur_node.depth + 1
                    cur_node.right = Node()
                    cur_node.right.depth = cur_node.depth + 1
                    self._grow(cur_node.left)
                    cur_node.value = value
                    self._grow(cur_node.right)
            elif cur_node.depth == max_depth:
                # if current depth == max_depth, insert value from t_set
                cur_node.value = draw_val(scope='t')

    def print_tree(self):
        self._print_tree(self.root)

    def _print_tree(self, cur_node):
            if cur_node is not None:
                if cur_node.value not in t_set:
                    print('(', end=" ")
                self._print_tree(cur_node.left)
                print(str(cur_node.value), end=" ")
                self._print_tree(cur_node.right)
                if cur_node.value not in t_set:
                    print(')', end=" ")

    def scan(self):
        '''scan the tree, index each node, store tree depth and node count'''
        self.root.ind = 0
        self.ncount = self._scan(self.root, 0) + 1
        self.height = self._height(self.root, 0)

    def _scan(self, cur_node, cur_ind):
        '''index each node by Preorder and return total numbers of nodes(-1)'''
        if cur_node.left is not None:
            cur_node.left.ind = cur_ind + 1
            cur_ind = self._scan(cur_node.left, cur_ind+1)
            cur_node.right.ind = cur_ind + 1
            cur_ind = self._scan(cur_node.right, cur_ind+1)
            return cur_ind
        if cur_node.left is None:
            return cur_ind

    def _height(self, cur_node, cur_height):
        if cur_node.left is None:
            return cur_height
        else:
            left_height = self._height(cur_node.left, cur_height+1)
            right_height = self._height(cur_node.right, cur_height+1)
            return max(left_height, right_height)

    def get_node(self, ind):
        'return the node given an index'
        find_node = None

        def _get_node(cur_node, ind):
            nonlocal find_node
            if cur_node.ind == ind:
                find_node = cur_node
                return
            else:
                if cur_node.left is None:
                    return
                _get_node(cur_node.left, ind)
                _get_node(cur_node.right, ind)

        _get_node(self.root, ind)
        return find_node

    def compute_tree(self, x, y):
        '''given the value of x and y, compute the expression tree'''
        if self.root.value in t_set:
            return eval(self.root.value)
        else:
            return self._compute_tree(self.root, x, y)

    def _compute_tree(self, cur_node, x, y):
        if cur_node.left is not None:
            left_value = self._compute_tree(cur_node.left, x, y)
            right_value = self._compute_tree(cur_node.right, x, y)
            if right_value == 0 and cur_node.value == '/':
                # divsion protection
                return 1
            else:
                return eval(str(left_value) + cur_node.value + str(right_value))
        else:
            return eval(cur_node.value)

    def compute_error(self, x, y, rf):
        '''Given the training set, return fitness.
        The fitness function is defined as the root-mean-squared error.
        x, y, and rf to be array-like objects,
        where as rf denotes the real funtion value '''
        sum_square = 0
        n = len(x)
        for i in range(n):
            sum_square += math.pow((self.compute_tree(x[i], y[i]) - rf[i]), 2)
        self.error = math.sqrt(sum_square / n)


def main(pop_size, tour_size, gen, graph=0):
    print('\nEvolution starts...\n')
    x, y, rf = read_data(file_name)
    pop = inital(pop_size)
    best, avge, worst, depth = evlove(pop, tour_size, gen, x, y, rf)
    if graph == 1:
        plot_figure(best, avge, worst, depth, gen)
    print('\n')
    return None


if __name__ == "__main__":
    pop_size = int(sys.argv[1])
    tour_size = int(sys.argv[2])
    gen = int(sys.argv[3])
    main(pop_size, tour_size, gen)
