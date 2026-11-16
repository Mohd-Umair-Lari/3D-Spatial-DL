import numpy as np

def inbox(point, box):
    for i in range(len(point)):
        if not (box[i][0] <= point[i] <= box[i][1]):
            return False
    return True


class KDTree:
    def __init__(self, P, d=0):
        n=len(P)
        m=n//2
        P.sort(key=lambda x:x[d])
        self.point=P[m]
        self.d=d
        d=(d+1)%len(P[0])
        self.left=self.right=None
        if m>0:
            self.left=KDTree(P[:m],d)
        if n-(m+1)>0:
            self.right=KDTree(P[m+1:],d)

    def rangesearch(self, box):
        p=self.point
        if inbox(p, box):
            yield p
        min,max=box[self.d]
        split=p[self.d]
        if self.left is not None and split>=min:
            yield from self.left.rangesearch(box)
        if self.right is not None and split<=max:
            yield from self.right.rangesearch(box) 

P=[(1,2),(3,4),(5,6)]
T=KDTree(P)
print(list(T.rangesearch(np.array([[3,4],[0,9]]))))