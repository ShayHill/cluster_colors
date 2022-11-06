class AAA:

    def __init__(self, x):
        self.x = x
    def __repr__(self):
        return str(self.x)


class ChildA(AAA):
    pass


class ChildB(AAA):
    pass


class ChildC(AAA):
    pass
         

class BBB(AAA):

    def __init__(self):
        self.x = [1, 2, 3] 
        self.A = ChildA(self.x)
        self.B = ChildB(self.x)
        self.C = ChildC(self.x)
        breakpoint()

if __name__ == '__main__':
    bbb = BBB()
