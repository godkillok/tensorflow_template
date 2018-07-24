# https://blog.csdn.net/l153097889/article/details/48310739
class TreeNode:
    def __init__(self, value=None, leftNode=None, rightNode=None):
        self.value = value
        self.leftNode = leftNode
        self.rightNode = rightNode


class Tree:
    def __init__(self, root=None):
        self.root = root

    def preOrder_1(self):
        if root is not None:
            stackNode = []
            node = self.root
            stackNode.append(node)
            while stackNode != []:
                node = stackNode.pop(0)
                print()

    def preOrder_2(self):
        '''
        stack is a queue that first in later out
         #1st put root
        2nd put right
        3nd put left
        :return:
        '''
        if self.root is not None:
            stackNode = []
            node = self.root
            stackNode.append(node)
            while stackNode != []:
                node = stackNode.pop()
                print(node.value, )
                if node.rightNode:
                    stackNode.append(node.rightNode)
                if node.leftNode:
                    stackNode.append(node.leftNode)

    def midOrder(self):
        '''
        :return:
        '''
        if self.root is not None:
            stack_node = []
            second = []
            second.append(self.root)
            node=self.root

            while stack_node!=[] or node:

                while node!=0:
                    stack_node.append(node)
                    node=node.leftNode
                node = stack_node.pop()
                print(node.value)

                node=node.rightNode

#对于一个节点而言，要实现访问顺序为左儿子-右儿子-根节点，可以利用后进先出的栈，
# 在节点不为空的前提下，依次将根节点，右儿子，左儿子压栈。
# 故我们需要按照根节点-右儿子-左儿子的顺序遍历树，而我们已经知道先序遍历的顺序是根节
# 点-左儿子-右儿子，故只需将先序遍历的左右调换并把访问方式打印改为压入另一个栈即可。
# 最后一起打印栈中的元素。

    def aftOrder(self):
        if not self.root:
            return
        stackNode = []
        flag=[]
        node = self.root
        while stackNode or node:
            while node:
                stackNode.append(node)
                flag.append(0)
                node=node.leftNode
            node=stackNode[-1]
            if flag[-1]==0 and node.rightNode:
                node =node.rightNode
                flag.append(1)
            else:
                flag.pop()
                node=stackNode.pop()
                print(node.value)
                node = 0

    def BFSOrder(self):
        '''
        https://www.cnblogs.com/simplepaul/p/6721687.html
        http://www.cnblogs.com/LZYY/p/3454778.html
        1.首先将根节点放入队列中。
       2.当队列为非空时，循环执行步骤3到步骤5，否则执行6；
       3.出队列取得一个结点，访问该结点；
       4.若该结点的左子树为非空，则将该结点的左子树入队列；
       5.若该结点的右子树为非空，则将该结点的右子树入队列；
       6.结束。
        :return:
        '''


        if not self.root:
            return
        stack=[]
        stack.append(self.root)
        while stack:
            node=stack.pop(0)
            if node.leftNode:
                stack.append(node.leftNode)
            if  node.rightNode:
                stack.append(node.rightNode)


##    def midOrder(self):
##        if not self.root:
##            return
##        stackNode = []
##        stackNode.append(self.root)
##        while stackNode:
##            node = stackNode.pop()
##            if node:
##                stackNode.append(node)
##                node = node.leftNode
##                stackNode.append(node)
##            elif stackNode:
##                node = stackNode.pop()
##                print node.value,
##                stackNode.append(node.rightNode)

if __name__ is '__main__':
    n10 = TreeNode(10, 0, 0)
    n9 = TreeNode(9, 0, 0)
    n3 = TreeNode(3, n9, n10)
    n8 = TreeNode(8, 0, 0)
    n14 = TreeNode(14, 0, 0)
    n7 = TreeNode(7, 0, 0)
    n16 = TreeNode(16, n7, 0)
    n2 = TreeNode(2, n14, n8)
    n1 = TreeNode(1, n2, n16)
    root = TreeNode(4, n1, n3)
    tree = Tree(root)

    tree.BFSOrder()
