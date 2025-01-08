import anytree #import * //Node, RenderTree, PreOrderIter, findall, search

# Define a function to find the ancestor of a node with more than one child
def find_ancestor_with_multiple_children(node):
    lengthOfthePath = 0
    while node.parent is not None:
        lengthOfthePath += 1
        parent = node.parent
        if len(parent.children) > 1:
            return parent, lengthOfthePath
        node = parent
    return None, lengthOfthePath



# Create some nodes
A = anytree.Node("A")
A_0 = anytree.Node("A_0", parent=A)
A_1 = anytree.Node("A_1", parent=A)
A_0_0 = anytree.Node("A_0_0", parent=A_0)
A_0_0_1 = anytree.Node("A_0_0_1", parent=A_0_0)
A_1_2 = anytree.Node("A_1_2", parent=A_1)
A_0_2 = anytree.Node("A_0_2", parent=A_0)

currentNode = "A"
print(f"{currentNode} has #childrent => ", len(anytree.Node(currentNode).children))
currentNode = anytree.find(A, filter_=lambda node: node.name==currentNode)
print(f"{currentNode.name} has #childrent => ", len(currentNode.children))
# Print the tree structure
print("Before any insertion")
for pre, fill, node in anytree.RenderTree(A):
    print("%s%s" % (pre, node.name))

currentNode = "A_0_0_1_2"
currentNodeParentName = "A_0_0_1"
currentNodeParent = anytree.find(A, filter_=lambda node: node.name==currentNodeParentName)

currentNode = anytree.Node(currentNode, parent=currentNodeParent)

print("After insertion of A_0_0_1_2 ")
for pre, fill, node in anytree.RenderTree(A):
    print("%s%s" % (pre, node.name))


currentNode = "A_0_0_1_2_1"
currentNodeParentName = "A_0_0_1_2"
currentNodeParent = anytree.find(A, filter_=lambda node: node.name==currentNodeParentName)
currentNode = anytree.Node(currentNode, parent=currentNodeParent)
print("After insertion of A_0_0_1_2_1 ")
for pre, fill, node in anytree.RenderTree(A):
    print("%s%s" % (pre, node.name))


# Find the ancestor of Node 3
ancestor, lengthOfPath = find_ancestor_with_multiple_children(A_0_0_1)



# Print the ancestor with multiple children
if ancestor is not None:
    print(f"The ancestor of {A_0_0_1.name} with multiple children is {ancestor.name} and path length is {lengthOfPath}")
else:
    print(f"{A_0_0_1.name} does not have an ancestor with multiple children")
    
# Print the tree structure using RenderTree
for pre, fill, node in anytree.RenderTree(A):
    print("%s%s" % (pre, node.name))

# Print the tree structure using PreOrderIter
for node in anytree.PreOrderIter(A):
    print(node.name)

# myString = "node2"
# node = Node(name=myString)

# print(node.parent)







# Find the parent of a node given as a string
node_name = "A_0"

# f = anytree.Node(node_name)
# anytree.findall(f)
result = anytree.find(A, filter_=lambda node: node.name==ancestor.name)
print(result)
if result.parent:
    print('parent name:', result.name)

# parent = node.parent

# Print the tree structure
# for pre, fill, node in RenderTree(A):
#     print("%s%s" % (pre, node.name))

# Print the parent of the node
# if result is not None:
#     print(f"The node of name =  {node_name} is {result}")
# else:
#     print(f"{node_name} does not have a parent")
