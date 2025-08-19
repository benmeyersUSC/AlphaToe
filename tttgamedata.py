class GameState:
    def __init__(self):
        self.state = [" "] * 9

    def __getitem__(self, key):
        return self.state[key]

    def __setitem__(self, key, value):
        if key > 8 or key < 0:
            return

        if value in ["X", "O", " "]:
            self.state[key] = value
