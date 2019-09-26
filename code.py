# za poizvedbo F in podatkovno bazo x odgovori z
# $\varepsilon$-diferencirano zasebnim odgovorom z uporabo Laplacevega mehanizma
def laplace(F, x, eps):
    result = F @ np.transpose(x)
    b = 2 / eps
    return result + np.random.laplace(0,b, size = result.shape)

# reši problem $min \norm{Cx-d}_2$
# pri pogojih $Ax >= b$
def least_squares(C, d, A, b):
    C = C.astype('float64')
    P = C.T @ C
    q = - C.T @ d
    G = -A
    h = -b
    sol = qpsolvers.solve_qp(P, q, G, h, solver='cvxopt')
    return np.linalg.norm(C @ sol - d, ord=2)

# vrne ali je $a \in FB_1^n + B_1^d$
def oracle(F, a):
    d, n = F.shape
    if np.linalg.norm(a, ord=1) <= 1:
        return True

    F_double = np.hstack((F, -F))
    A = np.vstack((np.identity(2*n),-np.ones(2*n)))
    b = np.zeros(2*n+1)
    b[2*n] = -1
    res = least_squares(F_double, a, A, b)
    
    return res < 1e-5

# vrne vzorec enakomerne porazdelitve na $B_p^n$ krogli
def uniform_ball(d, delta, p=2):
    normals = stats.gennorm(beta=p).rvs(size=d)
    exp = np.random.exponential()
    denom = (np.sum(abs(normals)**p)+exp)**(1/p)
    return (normals / denom) * delta
    
# naredi en korak sprehoda po kroglah velikosti $\delta$ iz točke $a$ po telesu $FB_1^n + B_1^d$
def ballwalk(F, a, delta):
    d, n = F.shape
    move = uniform_ball(d, delta)    
    while not oracle(F, a + move):
        print('fail')
        move = uniform_ball(d, delta)
    print(end - start)
    return move + a

# za poizvedbo $F$ in podatkovno bazo $x$ odgovori z
# $\varepsilon$-diferencirano zasebnim odgovorom z uporabo K-normnega mehanizma
def knorm(F, x, eps):
    d, n = F.shape

    r = np.random.gamma(d+1, scale=(1/eps))
    a = uniform_ball(d, 1, p=1)
    
    for i in range(n^2):
        a = ballwalk(F, a, 1/np.sqrt(d))
        
    return F @ x + r * a