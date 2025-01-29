use polars_graph::expressions::{build_graph, get_cycles};

#[test]
fn test_build_graph() {
    let sources = vec![1, 2, 3];
    let destinations = vec![2, 3, 1];
    let graph = build_graph(&sources, &destinations);

    // Test number of nodes
    assert_eq!(graph.node_count(), 3);
    // Test number of edges
    assert_eq!(graph.edge_count(), 3);
}

#[test]
fn test_get_cycles_int32() {
    let sources = vec![1, 2, 3, 4];
    let destinations = vec![2, 3, 1, 5];
    let graph = build_graph(&sources, &destinations);

    let cycles = get_cycles(&graph);
    assert_eq!(cycles.len(), 1); // Should find one cycle
    assert!(cycles[0].contains(&1)); // Cycle should contain these nodes
    assert!(cycles[0].contains(&2));
    assert!(cycles[0].contains(&3));
}