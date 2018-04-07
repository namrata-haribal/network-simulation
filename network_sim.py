import networkx as nx
import matplotlib.pyplot as plt
from random import uniform, randint, shuffle, choice, sample

class SocialDynamicsSimulation:
    '''
    Simulate social dynamics.
    '''
    def __init__(self, network_size, num_iterations, num_topics=10):
        '''
        :param network_size (int): the number of nodes in a graph.
        :param num_topics (int): the number of possible conversation topics. Default is 10.
        '''
        self.network_size = network_size
        self.conversation_topics = [i for i in range(1,num_topics+1)]

        # core opinions would be stickier, which is why the maximum alpha value is limited to 0.09.
        self.alpha_core_limit = 0.09

        # non core opinions less sticky than core opinons, thus the maximum alpha value is higher than  alpha_core_limit.
        self.alpha_non_core_limit = 0.19

        # nodes change edge weight quickly when differing on core opinions.
        self.beta_core_limit = 0.5

        # nodes change edge weight slowly when differing on core opinions.
        self.beta_non_core_limit =  0.3

        # for core opinions, nodes have a lower tolerance for differing opinions, which is why higher gamma values are chosen.
        self.gamma_core = [i for i in range(5,8)]

        # for non-core opinions, nodes have a higher tolerance for differing opinions, which is why lower gamma values are chosen.
        self.gamma_non_core = [i for i in range(2,5)]

        # number of times we run the simulation
        self.iterations = num_iterations

        # this list will store all the opinion values to be visualized:
        self.op_list = [0 for i in range(network_size*self.iterations*2)] # a list of zeros

    def initialize(self):
        # initialize a watts strogatz graph
        self.graph = nx.watts_strogatz_graph(self.network_size, 5, 0.5)
        # self.graph.add_node(1)
        # self.graph.add_node(2)
        # self.graph.add_edge(1,2)
        # print(list(self.graph.nodes))

        # initialize edge weights to 0.5 for all edges
        for edge in self.graph.edges:
            self.graph.edges[edge]['weight'] = 0.5

        # initialize node attributes for all nodes: topics node feels strongly & weekly about, associated opinion values, & personalized parameters.
        for node in self.graph.nodes:
            conv = self.conversation_topics.copy()
            num_strong = randint(0, len(self.conversation_topics))
            shuffle(conv)

            # opinions = {topic#, opinion, True if core topic, alpha value, beta value, gamma value} for each opinion.

            # add opinions and associated values for all topics that a node feels strongly about. True indicates it is a core opinion.
            opinions = {topic:[round(uniform(0,1),3), True,
                               round(uniform(0,self.alpha_core_limit),3),
                               round(uniform(self.beta_non_core_limit,self.beta_core_limit), 3),
                               choice(self.gamma_core)] for topic in conv[:num_strong]}

            # add opinions and associated values for all topics that a node feels weakly about. False indicates it is a non core opinion.
            opinions.update({topic:[round(uniform(0,1),3),False,
                                    round(uniform(0.09, 0.19), 3),
                                    round(uniform(0, 0.3), 3),
                                    choice(self.gamma_non_core)] for topic in conv[num_strong:]})


            #hash the dictionary so it can be added as an attribute to the node.
            def __hash__():
                return hash(opinions)


            self.graph.nodes[node]['opinions'] = opinions
            # print(self.graph.nodes[node]['opinions'])

        self.layout = nx.spring_layout(self.graph)  # Initial visual layout
        self.step = 0

    def observe(self):
        '''
        Draw the state of the network.
        '''
        self.layout = nx.spring_layout(self.graph, pos=self.layout, iterations=5)
        plt.clf()
        nx.draw(
            self.graph, pos=self.layout, with_labels=True,
            node_color=[self.op_list[i] for i in self.graph.nodes],
            edge_color=[self.graph.edges[i, j]['weight'] for i, j in self.graph.edges],
            edge_cmap=plt.cm.binary, edge_vmin=0, edge_vmax=1,
            alpha=0.7, vmin=0, vmax=1)
        plt.title('Step: ' + str(self.step))
        pass

    def update(self):
        if uniform(0, 1) < 0.01:
            additions = ["node", "graph"]
            to_add = "node" #choice(additions) #"node"
            nodes = list(self.graph.nodes)
            if to_add == "node":
                while True:
                    new_edge = sample(nodes,2)
                    if new_edge not in self.graph.edges:
                        break
                    self.graph.add_edge(new_edge[0], new_edge[1], weight=0.5)
            else:
                choices = ["ba", "er", "ws"]
                graph_type = choice(choices)
                choose_node = sample(nodes,1)
                if graph_type=="ba":
                    new_graph = nx.barabasi_albert_graph(5, 4)
                elif graph_type=="er":
                    new_graph = nx.erdos_renyi_graph(5,0)
                else:
                    new_graph = nx.watts_strogatz_graph(5,2,0.3)

                for node in new_graph.nodes:
                    conv = self.conversation_topics.copy()
                    num_strong = randint(0, len(self.conversation_topics))
                    shuffle(conv)
                    opinions_new = {topic: [round(uniform(0, 1), 3), True,
                                        round(uniform(0, self.alpha_core_limit), 3),
                                        round(uniform(self.beta_non_core_limit, self.beta_core_limit), 3),
                                        choice(self.gamma_core)] for topic in conv[:num_strong]}

                    # add opinions and associated values for all topics that a node feels weakly about. False indicates it is a non core opinion.
                    opinions_new.update({topic: [round(uniform(0, 1), 2), False,
                                             round(uniform(self.alpha_core_limit, self.alpha_non_core_limit), 3),
                                             round(uniform(0, self.beta_non_core_limit), 3),
                                             choice(self.gamma_non_core)] for topic in conv[num_strong:]})

                    # hash the dictionary so it can be added as an attribute to the node.
                    def __hash__():
                        return hash(opinions_new)

                    new_graph.nodes[node]['opinions'] = opinions_new
                    self.graph.add_node(new_graph)
                    for i in range(len(list(new_graph.nodes))):
                        self.graph.add_edge(choose_node,list(new_graph)[i], weight=0.5)

        else:
            # choose an edge
            edge = choice(list(self.graph.edges))
            #choose a conversation topic
            curr_topic = choice(self.conversation_topics)
            weight = self.graph.edges[edge]['weight']
            opinion_val = []
            opinion_status = []
            alpha = []
            beta = []
            gamma = []
            for n in edge:
                lst = list(self.graph.nodes[n]['opinions'][curr_topic])
                opinion_val.append(lst[0])
                opinion_status.append(lst[1])
                alpha.append(lst[2])
                beta.append(lst[3])
                gamma.append(lst[4])


            # How to write this in a more elegant way
            # opinion_val, opinion_status, alpha, beta, gamma = [self.graph.nodes[n]['opinions'][curr_topic] for n in edge]

            for i in [0,1]:
                op = round((opinion_val[i] + alpha[i] * weight * (opinion_val[1 - i] - opinion_val[i])),3) # new opinion value.]
                self.op_list[self.op_list.index(0)] = op
                self.graph.nodes[edge[i]]['opinions'][curr_topic] = [op,opinion_status[i], alpha[i], beta[i], gamma[i]] #updating the entire curr topic key

            self.graph.edges[edge]['weight'] = (weight +(sum(beta)/len(beta)) * weight * (1 - weight) * (1 - round((sum(gamma)/len(gamma)),0) * abs(opinion_val[0] - opinion_val[1])))

            if self.graph.edges[edge]['weight'] < 0.05:
                self.graph.remove_edge(*edge)

        self.step += 1




num_iter = 20
sim = SocialDynamicsSimulation(100, num_iter)
sim.initialize()
plt.figure()
sim.observe()
for i in range(num_iter):
    for i in range(num_iter):
        sim.update()
        plt.figure()
        sim.observe()
        plt.show()
