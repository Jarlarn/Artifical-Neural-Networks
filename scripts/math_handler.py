class MathHandler:

    def signum_func(self, x: float) -> int:
        if x > 0:
            return 1

        if x < 0:
            return -1

        return 0
