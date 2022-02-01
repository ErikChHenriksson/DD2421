import monkdata as m
import dtree
import drawtree_qt5 as drawtree
import random


""" Assignment 1 """
def calc_entropy():
    print(dtree.entropy(m.monk1))
    print(dtree.entropy(m.monk2))
    print(dtree.entropy(m.monk3))


""" Assignment3 """
def calc_average_gain():
    s = 1
    for set in [m.monk1, m.monk2, m.monk3]:
        string = ""
        for i in range(6):
            string += str(round(dtree.averageGain(set, m.attributes[i]), 4)) + " & "
        s += 1
        print(string)

""" Assignment pre-5 """
def monk1_subset():
    subset_1 = dtree.select(m.monk1, m.attributes[4], 1)
    subset_2 = dtree.select(m.monk1, m.attributes[4], 2)
    subset_3 = dtree.select(m.monk1, m.attributes[4], 3)
    subset_4 = dtree.select(m.monk1, m.attributes[4], 4)
    # No data points have value 3 or 4 for a_5 (?)

    for dp in subset_1:
        print(str(dp.attribute[m.attributes[0]]) + ", " +str(dp.attribute[m.attributes[1]]) + ", " + str(dp.attribute[m.attributes[2]]) + ", " + str(dp.attribute[m.attributes[3]]) + ", " + str(dp.attribute[m.attributes[4]]) + ", " + str(dp.attribute[m.attributes[5]]))


    print("best attr subset_1 " + str(dtree.bestAttribute(subset_1, m.attributes)))
    print("best attr subset_2 " + str(dtree.bestAttribute(subset_2, m.attributes)))
    print("best attr subset_3 " + str(dtree.bestAttribute(subset_3, m.attributes)))
    print("best attr subset_4 " + str(dtree.bestAttribute(subset_4, m.attributes)))

    s = 1
    for set in [subset_1, subset_2, subset_3, subset_4]:
        string = ""
        for i in range(6):
            string += str(round(dtree.averageGain(set, m.attributes[i]), 4)) + " & "
        print("set " + str(s) + ": " + string)
        s += 1
        print(dtree.mostCommon(set))


def draw_tree():
    tree = dtree.buildTree(m.monk1, m.attributes, 2)
    drawtree.drawTree(tree)


""" Assignment 5 """
def performances():
    s = 1
    for train, test in [(m.monk1, m.monk1test), (m.monk2, m.monk2test), (m.monk3, m.monk3test)]:
        t=dtree.buildTree(train, m.attributes)
        print("Performance monk"+str(s), dtree.check(t, test))
        s += 1


""" Assignment 6 """

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruning():
    monk1train, monk1val = partition(m.monk1, 0.6)
    t=dtree.buildTree(monk1train, m.attributes)
    prev_best = dtree.check(t, monk1val)
    while (True):
        best = 0
        for pruned in dtree.allPruned(t):
            if dtree.check(pruned, monk1val) > best:
                best = dtree.check(pruned, monk1val)
                new_t = pruned
        if best <= prev_best:
            break
        prev_best = best
        t = new_t
    # Found best
    print("Performance best", dtree.check(t, m.monk1test))


if __name__ == "__main__":
    pruning()

