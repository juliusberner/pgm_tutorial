import networkx as nx
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork, MarkovNetwork


def student_model():
    # Defining the model structure. We can define the network by just passing a list of edges.
    model = BayesianNetwork([("Difficulty", "Grade"), ("Intel", "Grade"), ("Grade", "Letter"), ("Intel", "SAT")], latents={"Grade", "Letter", "Intel"})

    # Defining individual CPDs.
    cpd_d = TabularCPD(variable="Difficulty", variable_card=2, values=[[0.6], [0.4]], state_names={"Difficulty": ["Easy", "Hard"]})
    cpd_i = TabularCPD(variable="Intel", variable_card=2, values=[[0.7], [0.3]], state_names={"Intel": ["Average", "High"]})

    # In pgmpy the colums are the evidences and rows are the states of the variable.
    cpd_g = TabularCPD(
        variable="Grade",
        variable_card=3,
        values=[[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        evidence=["Intel", "Difficulty"],
        evidence_card=[2, 2],
        state_names={"Grade": ["A", "B", "C"], "Intel": ["Average", "High"], "Difficulty": ["Easy", "Hard"]},
    )

    cpd_l = TabularCPD(
        variable="Letter",
        variable_card=2,
        values=[[0.1, 0.4, 0.99], [0.9, 0.6, 0.01]],
        evidence=["Grade"],
        evidence_card=[3],
        state_names={"Letter": ["Bad", "Good"], "Grade": ["A", "B", "C"]},
    )

    cpd_s = TabularCPD(
        variable="SAT",
        variable_card=2,
        values=[[0.95, 0.2], [0.05, 0.8]],
        evidence=["Intel"],
        evidence_card=[2],
        state_names={"SAT": ["Bad", "Good"], "Intel": ["Average", "High"]},
    )

    # Associating the CPDs with the network
    model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

    # check_model checks for the network structure and CPDs and verifies that the CPDs are correctly
    # defined and sum to 1.
    assert model.check_model()

    return model


def monty_hall_model():
    # Defining the network structure
    model = BayesianNetwork([("Choice", "Host"), ("Prize", "Host")], latents={"Prize"})

    # Defining the CPDs:
    cpd_c = TabularCPD("Choice", 3, [[0.33], [0.33], [0.33]])
    cpd_p = TabularCPD("Prize", 3, [[0.33], [0.33], [0.33]])
    cpd_h = TabularCPD(
        "Host",
        3,
        [
            [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
            [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
            [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0],
        ],
        evidence=["Choice", "Prize"],
        evidence_card=[3, 3],
    )

    # Associating the CPDs with the network structure.
    model.add_cpds(cpd_c, cpd_p, cpd_h)

    # Check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
    model.check_model()
    return model


def voting_model():
    model = MarkovNetwork([("A", "C"), ("C", "B"), ("B", "D"), ("D", "A")])

    # Define Factors
    factors = []
    for edge in model.edges():
        factors.append(DiscreteFactor(variables=edge, cardinality=[2, 2], values=[[5, 1], [1, 10]]))

    model.add_factors(*factors)
    model.check_model()

    return model


def to_daft(
    model,
    node_pos="circular",
    latex=True,
    pgm_params={},
    edge_params={},
    node_params={},
):
    """
    Returns a daft (https://docs.daft-pgm.org/en/latest/) object which can be rendered for
    publication quality plots. The returned object's render method can be called to see the plots.

    Parameters
    ----------
    node_pos: str or dict (default: circular)
        If str: Must be one of the following: circular, kamada_kawai, planar, random, shell, sprint,
            spectral, spiral. Please refer: https://networkx.org/documentation/stable//reference/drawing.html#module-networkx.drawing.layout for details on these layouts.

        If dict should be of the form {node: (x coordinate, y coordinate)} describing the x and y coordinate of each
        node.

        If no argument is provided uses circular layout.

    latex: boolean
        Whether to use latex for rendering the node names.

    pgm_params: dict (optional)
        Any additional parameters that need to be passed to `daft.PGM` initializer.
        Should be of the form: {param_name: param_value}

    edge_params: dict (optional)
        Any additional edge parameters that need to be passed to `daft.add_edge` method.
        Should be of the form: {(u1, v1): {param_name: param_value}, (u2, v2): {...} }

    node_params: dict (optional)
        Any additional node parameters that need to be passed to `daft.add_node` method.
        Should be of the form: {node1: {param_name: param_value}, node2: {...} }

    Returns
    -------
    Daft object: daft.PGM object
        Daft object for plotting the DAG.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> dag = DAG([('a', 'b'), ('b', 'c'), ('d', 'c')])
    >>> dag.to_daft(node_pos={'a': (0, 0), 'b': (1, 0), 'c': (2, 0), 'd': (1, 1)})
    <daft.PGM at 0x7fc756e936d0>
    >>> dag.to_daft(node_pos="circular")
    <daft.PGM at 0x7f9bb48c5eb0>
    >>> dag.to_daft(node_pos="circular", pgm_params={'observed_style': 'inner'})
    <daft.PGM at 0x7f9bb48b0bb0>
    >>> dag.to_daft(node_pos="circular",
    ...             edge_params={('a', 'b'): {'label': 2}},
    ...             node_params={'a': {'shape': 'rectangle'}})
    <daft.PGM at 0x7f9bb48b0bb0>
    """
    try:
        from daft import PGM
    except ImportError as e:
        raise ImportError("Package daft required. Please visit: https://docs.daft-pgm.org/en/latest/ for installation instructions.")

    if isinstance(node_pos, str):
        supported_layouts = {
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "planar": nx.planar_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "spring": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "spiral": nx.spiral_layout,
        }
        if node_pos not in supported_layouts.keys():
            raise ValueError("Unknown node_pos argument. Please refer docstring for accepted values")
        else:
            node_pos = supported_layouts[node_pos](model)
    elif isinstance(node_pos, dict):
        for node in model.nodes():
            if node not in node_pos.keys():
                raise ValueError(f"No position specified for {node}.")
    else:
        raise ValueError("Argument node_pos not valid. Please refer to the docstring.")

    daft_pgm = PGM(**pgm_params)
    for node in model.nodes():
        try:
            extra_params = node_params[node]
        except KeyError:
            extra_params = dict()

        if latex:
            daft_pgm.add_node(
                node,
                rf"${node}$",
                node_pos[node][0],
                node_pos[node][1],
                observed=node not in model.latents,
                **extra_params,
            )
        else:
            daft_pgm.add_node(
                node,
                f"{node}",
                node_pos[node][0],
                node_pos[node][1],
                observed=node not in model.latents,
                **extra_params,
            )

    for u, v in model.edges():
        try:
            extra_params = edge_params[(u, v)]
        except KeyError:
            extra_params = dict()
        daft_pgm.add_edge(u, v, **extra_params)

    return daft_pgm


def render(model, node_pos="circular", grid_unit=None, node_unit=None):
    if grid_unit is None:
        grid_unit = max(map(len, model.nodes)) * 0.14 + 0.5
    if node_unit is None:
        node_unit = max(map(len, model.nodes)) * 0.11 + 0.6
    pgm_params = {"node_unit": node_unit, "grid_unit": grid_unit}
    if isinstance(model, MarkovNetwork):
        pgm_params["directed"] = False
    to_daft(model, node_pos=node_pos, pgm_params=pgm_params).render()


def print_cpds(model):
    for cpd in model.cpds:
        evidence = ", ".join(map(str, cpd.variables[1:]))
        if evidence:
            evidence = "|" + evidence
        print(f"P({cpd.variable}{evidence}), {cpd.values.size} values")


def print_cpd(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup
