from ._rank import Rank


class WithoutRank(Rank):
    def __init__(self):
        super().__init__()
        self.name = "Without"

    def __call__(self, x, y, rules, measures):
        return rules
