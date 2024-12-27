import random

OPERATIONS = ["nor_conv_3x3", "avg_pool_3x3", "skip_connect", "none", "nor_conv_1x1", ]

class Architecture:
    def __init__(self, api, arch_str=None):
        self.api=api
        if arch_str is None:
            self._init_arch()
        else:
            self.arch_str = arch_str
        
    def _init_arch(self):
        base_str = "|"
        for i in range(3):
            for j in range(i+1):
                op = random.choice(OPERATIONS)
                base_str = base_str + op + "~" + str(j) +"|"
            base_str = base_str + "+" + "|"
        base_str = base_str[:-2]
        self.arch_str = base_str

    def get_child(self):
        indexes = [1, 3, 4, 6, 7, 8]
        splitted = self.arch_str.split('|')
        to_mutate = random.choice(indexes)
        new_op = random.choice(OPERATIONS)
        op_link = splitted[to_mutate].split('~')
        op_link[0]=new_op
        new_op_link = "~".join(op_link)
        splitted[to_mutate] = new_op_link
        child_str = "|".join(splitted)
        return Architecture(arch_str=child_str, api=self.api)
    
    def get_acc(self):
        index = self.api.query_index_by_arch(self.arch_str)
        acc = self.api.get_more_info(index, "cifar10", hp="200")['test-accuracy']
        self.acc = acc
        return acc