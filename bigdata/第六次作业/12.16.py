def find_decision_boundary(density, degree, theta, threshold, cord_bounds):
    t1 = np.linspace(cord_bounds[0], cord_bounds[1], density)
    t2 = np.linspace(cord_bounds[2], cord_bounds[3], density)

    coordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*coordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, degree)
    mapped_cord.insert(0, 'Ones', 1)
    print(mapped_cord.shape)
    print(theta.shape)

    inner_product = np.matrix(mapped_cord) * theta.T
    decision = mapped_cord[np.abs(inner_product) < threshold]
    print(decision)

    return decision.F10, decision.F11