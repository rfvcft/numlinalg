import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import matplotlib.animation as anim
from matplotlib.patches import Circle

#========== TASK 1 ===========

def offdiag_scale(A, p):
    return p*A + (1-p)*np.diag(A.diagonal())

def task1():
    A = np.array([[5, 0, 0, -1],
                  [1, 0, -1, 1],
                  [-1.5, 1, -2, 1],
                  [-1, 1, 3, -3]])

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_xlim((-9, 6))
    ax.set_ylim((-6, 6))
    ax.grid(True)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    annotation = ax.annotate("p=0.00", (-8,5))
    dots, = plt.plot([], [], "b.", label="Eigenvalues of A(p)")
    circles = []
    for _ in range(A.shape[0]):
        circles.append(Circle((0, 0), 0, fill=False))

    full_eigs, _ = sl.eig(A)
    full_re, full_im = full_eigs.real, full_eigs.imag
    ax.plot(full_re, full_im, 'ro', label="Eigenvalues of A")
    ax.plot(A.diagonal(), np.zeros(A.shape[0]), 'gx', label="Values on A's diagonal")
    ax.legend()
    MAX_FRAMES = 100

    def init():
        for circle in circles:
            ax.add_patch(circle)
        return []

    def animate(frame):
        p = (frame+1)/MAX_FRAMES
        A_p = offdiag_scale(A, p)
        for i in range(A_p.shape[0]):
            radius = sum(np.abs(A_p[i])) - np.abs(A_p[i, i])
            center = (A_p[i,i], 0)
            circles[i].center = center
            circles[i].radius = radius
        eigs, _ = sl.eig(A_p)
        re = eigs.real
        im = eigs.imag
        dots.set_data(re, im)
        annotation.set_text(f"p={p:.2f}")
        return dots

    animation = anim.FuncAnimation(fig=fig, func=animate, init_func=init, frames=MAX_FRAMES, interval=100)
    writergif = anim.PillowWriter(fps=60)
    animation.save('gershgorin_discs.gif', writer=writergif, progress_callback=lambda i, n: print(f"Saving frame {i}/{n}"))

    # plt.show()
    plt.close()

def print_matrix(A):
    for i in range(A.shape[0]):
        print("[", end="")
        for j in range(A.shape[1]):
            print(f"{A[i,j]:.3f}", end=" ")
        print("]")

#========== TASK 2 ===========

def qr_rayleigh(A : np.ndarray) -> tuple[np.ndarray, int]:
    m = A.shape[0]
    if m == 1:
        return A, 0
    tol = 1e-5
    A = sl.hessenberg(A)
    I = np.eye(m)
    count = 0
    done = False
    while not done:
        count += 1
        mu = A[m-1, m-1]
        Q, R = sl.qr(A - mu*I)
        A = R@Q + mu*I
        for j in range(m-1):
            if abs(A[j, j+1]) < tol:
                A[j, j+1] = A[j+1, j] = 0
                A_1, c1 = qr_rayleigh(A[:j+1, :j+1])
                A_2, c2 = qr_rayleigh(A[j+1:, j+1:])
                A = np.block([[A_1, np.zeros((j+1, m-j-1))],
                              [np.zeros((m-j-1, j+1)), A_2]])
                count += (c1 + c2)
                done = True
                break
    return A, count

def task2():
    n = 10
    A = np.array(np.random.random((n,n)))
    A = 1/2*(A.T + A)
    # Q, R = sl.qr(A)
    # A = Q @ Q.T
    eigs = sl.eig(A)[0]
    D, count = qr_rayleigh(A)
    print(count)
    print(np.allclose(sorted(eigs.real), sorted(D.diagonal()), rtol=1e-8))

#========== TASK 4 ===========

def p(A : np.ndarray, x : float) -> list[float]:
    ps = [1, A[0,0]-x]
    for i in range(1, A.shape[0]):
        ps.append((A[i,i]-x) * ps[i] - A[i,i-1]**2 * ps[i-1])
    return ps

def num_zeros(A : np.ndarray, x : float) -> int:
    ps = p(A, x)
    zeros = 0
    # Ugly way to check for sign changes, but I guess it works...?
    for i in range(len(ps)-1):
        s1 = np.sign(ps[i])
        s2 = np.sign(ps[i+1])
        if s2 != 0 and s1 != s2:
            zeros += 1
    return zeros

def bisection(A : np.ndarray, i : int, tol : float = 1e-8) -> float:
    """ Find the i:th eigenvalue of A """
    # First, find an interval which contains the i:th eigenvalue
    # Do it by exponentially expanding the size of the interval until we have
    # z_1 < i and z_2 >= i
    a = -1
    b = 1
    z_1 = num_zeros(A, a)
    z_2 = num_zeros(A, b)
    k = 1
    while z_1 >= i or z_2 < i:
        if z_1 >= i:
            a -= 2**k
            k += 1
            z_1 = num_zeros(A, a)
        else:
            b += 2**k
            k += 1
            z_2 = num_zeros(A, b)

    # Now we know that the i:th eigenvalue is in the interval [a,b]
    # so we can start bisecting the interval
    while b-a > tol:
        c = (b+a)/2
        z = num_zeros(A, c)
        if z >= i:
            b = c
        else:
            a = c

    return (b+a)/2

def task4():
    n = 100
    index = 3
    A = np.array(np.random.random((n,n)))
    A = 1/2*(A.T + A)
    A_tridag = sl.hessenberg(A)
    true_eigs = sl.eig(A_tridag)[0]
    eig = bisection(A_tridag, i=index)
    print(sorted(true_eigs.real)[index-1])
    print(eig)


# task1()
task2()
# task4()
