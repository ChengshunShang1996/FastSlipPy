def bisection(f,
              lb,
              ub,
              target=0.0,
              tolX=1e-6,
              tolFun=0.0,
              maxiter=1000):

    # shift function by target
    def g(x):
        return f(x) - target

    flb = g(lb)
    fub = g(ub)

    # root must be bracketed
    if flb * fub > 0:
        return np.nan, np.nan, -2

    for _ in range(maxiter):

        x = 0.5 * (lb + ub)

        fx = g(x)

        outsideTolX = abs(ub - x) > tolX
        outsideTolFun = abs(fx) > tolFun

        # convergence
        if (not outsideTolX) and (not outsideTolFun):
            return x, fx + target, 3

        if not outsideTolX:
            return x, fx + target, 1

        if not outsideTolFun:
            return x, fx + target, 2

        # keep bracket
        if np.sign(fx) != np.sign(fub):
            lb = x
            flb = fx
        else:
            ub = x
            fub = fx

    return x, fx + target, -1