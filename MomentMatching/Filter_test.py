



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from MomentMatching.baseMomentMatch import MomentMatching, UnscentedTransform, TaylorTransform
    from MomentMatching.TimeSeriesModel import TimeSeriesModel, UniformNonlinearGrowthModel
    from MomentMatching.StateModels import GaussianState
    from MomentMatching.ExpectationPropagation import EPNodes, TopEP
    import numpy as np

    np.random.seed(seed=100)


    N = 50
    demo = UniformNonlinearGrowthModel()
    data = demo.system_simulation(N)
    x_true, x_noisy, y_true, y_noisy = zip(*data)

    transform = TaylorTransform(dimension_of_state=1)
    Nodes = EPNodes(dimension_of_state=1, N=N)
    EP = TopEP(system_model=demo, moment_matching=transform.moment_matching_KF)

    plt.plot(x_true)

    plt.scatter(list(range(N)), x_noisy)
    plt.plot(y_noisy)
    plt.show()