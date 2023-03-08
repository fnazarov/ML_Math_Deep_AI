def newton(x,n):
    x_n = x
    for i in range(1,n):

        x_n_1 = x_n - (x_n**2 -2 )/ (2*x_n)
        x_n = x_n_1
        print(x_n)